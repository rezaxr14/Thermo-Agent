"""Anytime curves for Design: how good is the best valid design after b design evaluations?
    python -m thermo_agent.design.curves DATA_DIR --split test_id --limit 10 [--report reports/x.json ...] --out paper/generated/design_anytime.png
Baselines are enumeration of the default catalog, random search and the reference procedure (enumerate then polish); a model run
contributes its curve when its report was made with `--refine-turns`."""
from __future__ import annotations
import argparse, json, pathlib, random, statistics as st
from ..core.schema import Site
from .search import anytime_enumeration, anytime_reference, random_search


def to_gaps(curve, ref):
    return [None if f is None else f / ref - 1 for f in curve]


def at_budget(curve: list, b: int):
    """Best gap using at most b evaluations (the last entry within budget); None if nothing valid yet."""
    if not curve: return None
    return curve[min(b, len(curve)) - 1]


def summarize(curves: list[list], budgets: list[int], tol: float = 0.02) -> list[dict]:
    out = []
    for b in budgets:
        v = [at_budget(c, b) for c in curves]; ok = [g for g in v if g is not None]
        out.append({"budget": b, "valid": round(len(ok) / len(v), 3), "within_tol": round(sum(g <= tol for g in ok) / len(v), 3), "mean_gap_pct": round(100 * st.mean(ok), 2) if ok else None})
    return out


def baseline_curves(rows: list[dict], seed: int = 0, random_budget: int = 80) -> dict[str, list[list]]:
    rng = random.Random(seed); out = {"enumerate_defaults": [], "random": [], "enumerate_then_polish": []}
    for r in rows:
        site = Site.model_validate_json(r["site"]); ref = r["ref_fuel_w"]
        out["enumerate_defaults"].append(to_gaps(anytime_enumeration(site), ref))
        out["random"].append(to_gaps(random_search(site, rng, random_budget), ref))
        out["enumerate_then_polish"].append(to_gaps(anytime_reference(site, max_evals=40), ref))
    return out


def model_curves(report: dict, split: str) -> list[list]:
    """Per-example anytime curves as (evals, gap) pairs -> per-evaluation gap list (carry forward)."""
    curves = []
    for ex in (report.get("design", {}).get(split) or {}).get("examples", []):
        pts = ex.get("anytime") or []; n = max([p[0] for p in pts], default=0); c = [None] * n
        for evals, gap in pts:
            for i in range(max(evals, 1) - 1, n): c[i] = gap if c[i] is None or (gap is not None and gap < c[i]) else c[i]
        curves.append(c)
    return curves


def plot(series: dict[str, list[dict]], path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for name, rows in series.items():
        axes[0].plot([r["budget"] for r in rows], [r["within_tol"] for r in rows], label=name)
        pts = [(r["budget"], r["mean_gap_pct"]) for r in rows if r["mean_gap_pct"] is not None]
        if pts: axes[1].plot(*zip(*pts), label=name)
    axes[0].set_xlabel("design evaluations"); axes[0].set_ylabel("share within 2 % of best known"); axes[1].set_xlabel("design evaluations"); axes[1].set_ylabel("mean gap among valid (%)")
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--split", default="test_id"); ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--report", nargs="*", default=[]); ap.add_argument("--max-budget", type=int, default=60); ap.add_argument("--out", default="paper/generated/design_anytime.png")
    a = ap.parse_args(); rows = [json.loads(l) for l in open(pathlib.Path(a.data) / f"design_{a.split}.jsonl")][: a.limit]
    budgets = [1, 2, 3, 5, 8, 12, 20, 32, 48, a.max_budget]; series = {k: summarize(v, budgets) for k, v in baseline_curves(rows).items()}
    for p in a.report:
        rep = json.loads(pathlib.Path(p).read_text()); mc = model_curves(rep, a.split)
        if mc: series[f"{pathlib.Path(p).stem}"] = summarize(mc, budgets)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); plot(series, a.out)
    print(json.dumps(series, indent=1))


if __name__ == "__main__":
    main()
