"""python -m thermo_agent.audit.run_eval DATA_DIR --split test_id --limit 40 : non-LLM baselines for Audit."""
from __future__ import annotations
import argparse, json, pathlib, random, statistics as st
from ..core.schema import Plant
from ..core.solve import solve
from ..eval.rows import load_split
from .rewards import ndcg
from .truth import first_law_ranking


def _rule_change_gain(plant, r):
    """Fixed rule: raise the combustor outlet temperature 10 percent, else cut the engine loss fraction 10 percent."""
    from ..core.schema import Site
    from . import changes as ch
    from .truth import change_effect
    site = Site.model_validate_json(r["site"]) if r["site"] else None
    c = next((c for c in plant.components if c.type == "combustor"), None) or next((c for c in plant.components if c.type == "gas_engine"), None)
    cid, p, v = (c.id, "T_out", c.params["T_out"] * 1.1) if c.type == "combustor" else (c.id, "f_loss", c.params["f_loss"] * 0.9)
    e = change_effect(plant, site, cid, p, float(f"{v:.6g}"), r["net_w"], r["fuel_w"])
    return (max(0.0, -e["delta_w"] / r["best_gain_w"]), e["delta_w"]) if e["valid"] else None


def score(rows: list[dict], seed: int = 0) -> dict:
    """Ranking is scored on the components other than the combustor / gas engine (always the largest destroyer)."""
    rng = random.Random(seed)
    nd = {"random": [], "first_law": []}; top1 = {"random": [], "first_law": []}; ncand = []; rule_gain = []
    for r in rows:
        plant = Plant.model_validate_json(r["plant"]); ed = json.loads(r["ed_json"])
        pm = next(c.id for c in plant.components if c.type in ("combustor", "gas_engine"))
        ed_r = {k: v for k, v in ed.items() if k != pm}; best = max(ed_r, key=ed_r.get)
        fl = [c for c, _ in first_law_ranking(plant, solve(plant)) if c != pm]
        rnd = [rng.sample(list(ed_r), len(ed_r)) for _ in range(20)]
        nd["random"].append(st.mean(ndcg(x, ed_r) for x in rnd)); nd["first_law"].append(ndcg(fl, ed_r))
        top1["random"].append(st.mean(x[0] == best for x in rnd)); top1["first_law"].append(fl[0] == best)
        ncand.append(r["n_candidates"]); rule_gain.append(_rule_change_gain(plant, r))
    n = len(rows); rg = [g[0] for g in rule_gain if g is not None]
    pairs = [(-r["best_gain_w"], r["fuel_w"]) for r in rows]                 # effect of the best allowed change on each plant
    score = lambda pred, act, fuel: max(0.0, 1.0 - abs(pred - act) / max(abs(act), 0.002 * fuel))          # the a_pred reward
    zero = [score(0.0, d, f) for d, f in pairs]
    loo = [score(f * (sum(x / y for x, y in pairs) - d / f) / max(len(pairs) - 1, 1), d, f) for d, f in pairs]   # mean relative change of the other plants
    return {"n": n, "ranking_ndcg3_excluding_prime_mover": {k: round(st.mean(v), 3) for k, v in nd.items()},
            "ranking_top1_excluding_prime_mover": {k: round(st.mean(v), 3) for k, v in top1.items()},
            "rule_change (raise turbine inlet T 10%, else cut engine loss 10%)": {"mean_gain_vs_best": round(st.mean(rg), 3) if rg else None, "valid": f"{len(rg)}/{n}"},
            "prediction_baselines_for_the_best_change (a_pred score)": {"predict_zero": round(st.mean(zero), 3) if zero else None, "predict_mean_relative_change": round(st.mean(loo), 3) if loo else None},
            "change_search_evaluations": round(st.mean(ncand), 1), "model_evaluations": 1}


def run(path, limit=None, seed=0) -> dict:
    """Baselines for one split file; kept for callers that already have a file path."""
    return score([json.loads(l) for l in open(path)][:limit], seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data", nargs="+", help="ground-truth dirs, e.g. data/processed data/processed_arch")
    ap.add_argument("--split", default="test_id"); ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args(); rows = load_split(a.data, "audit", a.split)[: a.limit]
    print(a.split, json.dumps(score(rows)))
