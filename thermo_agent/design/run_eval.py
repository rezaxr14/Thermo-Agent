"""python -m thermo_agent.design.run_eval DATA_DIR [DATA_DIR ...] --split test_ood_climate --limit 10 : search baselines on a split."""
from __future__ import annotations
import argparse, json, pathlib, random, statistics as st
from ..core.schema import Site
from ..eval.rows import load_split
from .search import enumerate_designs, reference_design, random_search


def score(rows: list[dict], seed: int = 0) -> dict:
    rng = random.Random(seed)
    acc = {"enum_defaults": [], "enum_polish": [], "random_12": []}
    for r in rows:
        site = Site.model_validate_json(r["site"]); ref = r["ref_fuel_w"]
        tab = enumerate_designs(site); f = [e.fuel_w for _, e in tab if e.fuel_w]
        acc["enum_defaults"].append((min(f) / ref - 1 if f else None, len(tab)))
        ref2 = reference_design(site); acc["enum_polish"].append((ref2["fuel_w"] / ref - 1, ref2["evals"]))
        rs = random_search(site, rng, 12); acc["random_12"].append((rs[-1] / ref - 1 if rs[-1] else None, 12))
    out = {"n": len(rows)}
    for k, v in acc.items():
        ok = [g for g, _ in v if g is not None]
        out[k] = {"valid": f"{len(ok)}/{len(v)}", "mean_gap_pct": round(100 * st.mean(ok), 2) if ok else None, "evals": round(st.mean(e for _, e in v), 1) if v else 0}
    return out


def run(path, limit=None, seed=0) -> dict:
    return score([json.loads(l) for l in open(path)][:limit], seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data", nargs="+", help="ground-truth dirs, e.g. data/processed data/processed_arch")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args(); rows = load_split(a.data, "design", a.split)[: a.limit]
    print(a.split, json.dumps(score(rows), indent=1))
