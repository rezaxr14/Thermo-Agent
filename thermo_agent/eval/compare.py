"""Paired comparison of two reports from model_eval (same examples): e.g. harness on vs off, SFT vs GRPO.
    python -m thermo_agent.eval.compare A.json B.json [--task repair] [--holm]
Binary metrics: exact McNemar test. Other metrics: paired bootstrap interval and a sign-flip permutation test on the differences.
Examples are matched by id; examples missing on either side or with a missing value are dropped (the count is reported)."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from scipy.stats import binomtest

BINARY = {"repair": ["success", "first_turn_success"], "design": ["valid", "within_1pct"], "ledger": ["all_correct"], "audit": []}
CONTINUOUS = {"repair": ["solves"], "design": ["gap", "evals"], "audit": ["format", "rank", "valid", "gain", "pred", "grounded"], "ledger": ["exact", "closure", "nonneg"]}


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


def compare(rep_a: dict, rep_b: dict, tasks=None) -> list[dict]:
    rows = []
    for task in (tasks or BINARY):
        for split, va in (rep_a.get(task) or {}).items():
            vb = (rep_b.get(task) or {}).get(split)
            if not va or not vb or "examples" not in va or "examples" not in vb: continue
            for metric, binary in [(m, True) for m in BINARY[task]] + [(m, False) for m in CONTINUOUS[task]]:
                r = paired(va["examples"], vb["examples"], metric, binary)
                if r: rows.append({"task": task, "split": split, "metric": metric, **r})
    ps = holm([r["p"] for r in rows]) if rows else []
    for r, p in zip(rows, ps): r["p_holm"] = p
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--task", nargs="*")
    a = ap.parse_args(); ra, rb = (json.loads(pathlib.Path(p).read_text()) for p in (a.a, a.b))
    print(f"A = {a.a}\nB = {a.b}\n(diff = A - B; p_holm corrects across every comparison listed)")
    for r in compare(ra, rb, a.task):
        print(f"{r['task']:7s} {r['split']:18s} {r['metric']:20s} n={r['n']:<4d} A={r['mean_a']:<8} B={r['mean_b']:<8} diff={r['diff']:<8} ci={r['ci95']} p={r['p']} p_holm={r['p_holm']} ({r['test']})")


if __name__ == "__main__":
    main()
