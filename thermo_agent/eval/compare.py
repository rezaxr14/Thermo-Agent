"""Paired comparison of two reports from model_eval (same examples): e.g. harness on vs off, SFT vs GRPO.
    python -m thermo_agent.eval.compare A.json B.json [--task repair] [--holm]
    python -m thermo_agent.eval.compare A.json B.json --family uplift=repair/test_id/success,...
Binary metrics: exact McNemar test. Other metrics: paired bootstrap interval and a sign-flip permutation test on the differences.
Examples are matched by id; examples missing on either side or with a missing value are dropped (the count is reported).

Holm correction is applied *within named comparison families*, never across unrelated tests. The four
headline families of the frozen protocol (§5) are named in ``HEADLINE_FAMILIES``. Use ``compare_families``
to pass each family's tests explicitly; ``compare`` keeps the old "all rows are one family" behaviour.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from scipy.stats import binomtest

BINARY = {"repair": ["success", "first_turn_success"], "design": ["valid", "within_1pct"], "ledger": ["all_correct"], "audit": []}
CONTINUOUS = {"repair": ["solves"], "design": ["gap", "evals"], "audit": ["format", "rank", "valid", "gain", "pred", "grounded"], "ledger": ["exact", "closure", "nonneg"]}

# The four Gate 9 headline families. A family is a dict key passed to compare_families; each key
# scopes one Holm correction. Keeping the names here makes the scope explicit instead of implicit.
HEADLINE_FAMILIES = {
    "harness_uplift_h_minus_z": "H - Z per model",
    "sft_over_base_same_mode": "SFT - base in the same mode",
    "grpo_over_sft_by_k": "GRPO - SFT by k",
    "final_4b_vs_baselines": "final 4B vs base 9B / other LLMs / frontier Z/H",
}


def paired(a: list[dict], b: list[dict], metric: str, binary: bool, seed: int = 0, n_boot: int = 4000, n_perm: int = 20000) -> dict | None:
    ib = {e["id"]: e for e in b}; pairs = [(float(e[metric]), float(ib[e["id"]][metric])) for e in a if e["id"] in ib and e.get(metric) is not None and ib[e["id"]].get(metric) is not None]
    if not pairs: return None
    x, y = np.array(pairs).T; d = x - y; rng = np.random.default_rng(seed); n = len(d)
    boot = rng.choice(d, size=(n_boot, n)).mean(axis=1); lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {"n": n, "mean_a": round(float(x.mean()), 4), "mean_b": round(float(y.mean()), 4), "diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    if binary:
        b01, c10 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())          # a only, b only
        out.update(test="mcnemar_exact", only_a=b01, only_b=c10, p=1.0 if b01 + c10 == 0 else float(binomtest(b01, b01 + c10, 0.5).pvalue))
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n)); perm = np.abs((signs * d).mean(axis=1))
        out.update(test="sign_flip_permutation", p=float((np.sum(perm >= abs(d.mean()) - 1e-15) + 1) / (n_perm + 1)))
    out["p"] = round(out["p"], 5); return out


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return [round(p, 5) for p in adj]


def _one(rep_a: dict, rep_b: dict, task: str, split: str, metric: str) -> dict | None:
    """One raw paired row (no correction) for an explicitly chosen cell/metric."""
    cell_a = (rep_a.get(task) or {}).get(split)
    cell_b = (rep_b.get(task) or {}).get(split)
    if not cell_a or not cell_b or "examples" not in cell_a or "examples" not in cell_b:
        return None
    r = paired(cell_a["examples"], cell_b["examples"], metric,
               metric in BINARY.get(task, []))
    if not r:
        return None
    return {"task": task, "split": split, "metric": metric, **r}


def _load_reports(value):
    """Accept an already-parsed report dict or a path to a report JSON."""
    if isinstance(value, dict):
        return value
    return json.loads(pathlib.Path(value).read_text(encoding="utf-8"))


def compare_rows(rep_a: dict, rep_b: dict, tasks=None) -> list[dict]:
    """All raw paired rows between two reports, without any Holm correction."""
    rows = []
    for task in (tasks or BINARY):
        for split, va in (rep_a.get(task) or {}).items():
            vb = (rep_b.get(task) or {}).get(split)
            if not va or not vb or "examples" not in va or "examples" not in vb: continue
            for metric, binary in [(m, True) for m in BINARY[task]] + [(m, False) for m in CONTINUOUS[task]]:
                r = paired(va["examples"], vb["examples"], metric, binary)
                if r: rows.append({"task": task, "split": split, "metric": metric, **r})
    return rows


def compare(rep_a: dict, rep_b: dict, tasks=None) -> list[dict]:
    """Legacy: treat every row as one family and Holm-correct across all of them."""
    rows = compare_rows(rep_a, rep_b, tasks)
    ps = holm([r["p"] for r in rows]) if rows else []
    for r, p in zip(rows, ps): r["p_holm"] = p
    return rows


def compare_families(families: dict[str, list], rep_a=None, rep_b=None) -> list[dict]:
    """Named, family-scoped Holm correction.

    ``families`` maps a family name to a list of test specs. Each spec is either
    ``(rep_a, rep_b, task, split, metric)`` or a dict with those keys (``a``/``rep_a`` and
    ``b``/``rep_b`` accept report dicts or paths). ``rep_a``/``rep_b`` are defaults for specs
    that omit the reports, e.g. several families over the same A/B pair.

    Every returned row records ``family``, its raw ``p`` and ``p_holm`` corrected ONLY against
    the other p-values of that same family -- never across families.
    """
    out = []
    for name, specs in families.items():
        fam = []
        for spec in specs:
            if isinstance(spec, dict):
                a = _load_reports(spec.get("a", spec.get("rep_a", rep_a)))
                b = _load_reports(spec.get("b", spec.get("rep_b", rep_b)))
                task, split, metric = spec["task"], spec["split"], spec["metric"]
            else:
                a, b, task, split, metric = spec
                a, b = _load_reports(a), _load_reports(b)
            row = _one(a, b, task, split, metric)
            if row: fam.append(row)
        ps = holm([r["p"] for r in fam]) if fam else []
        for row, p in zip(fam, ps):
            row["family"] = name
            row["p_holm"] = p
        out.extend(fam)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--task", nargs="*")
    ap.add_argument("--family", action="append", default=[],
                    help="named Holm family NAME=task/split/metric,... applied to A and B (repeatable); "
                         "p_holm is corrected within each family only")
    a = ap.parse_args()
    ra, rb = (_load_reports(p) for p in (a.a, a.b))
    if a.family:
        families = {}
        for block in a.family:
            name, _, selectors = block.partition("=")
            families[name.strip()] = [(ra, rb, *sel.split("/")) for sel in selectors.split(",") if sel.strip()]
        rows = compare_families(families, ra, rb)
        print(f"A = {a.a}\nB = {a.b}\n(diff = A - B; p_holm corrects within each named family only)")
    else:
        rows = compare(ra, rb, a.task)
        print(f"A = {a.a}\nB = {a.b}\n(diff = A - B; p_holm corrects across every comparison listed)")
    for r in rows:
        fam = f"{r['family']:28s} " if "family" in r else ""
        print(f"{fam}{r['task']:7s} {r['split']:18s} {r['metric']:20s} n={r['n']:<4d} A={r['mean_a']:<8} B={r['mean_b']:<8} diff={r['diff']:<8} ci={r['ci95']} p={r['p']} p_holm={r['p_holm']} ({r['test']})")


if __name__ == "__main__":
    main()
