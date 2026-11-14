"""python -m thermo_agent.eval.run_eval DATA_DIR [--split test_ood_fault] [--limit 50] : non-LLM baselines on a split."""
from __future__ import annotations
import argparse, json, pathlib, statistics as st
from ..core.schema import Plant, Site
from ..synth.faults import apply_patch, _dump
from ..train.rewards import evaluate
from .baselines import search_repair
from .repair import score_patch
from ..train.prompts import describe_failure


def run(path, limit=None, by_fault=False) -> dict:
    rows = [json.loads(l) for l in open(path)][:limit]
    if by_fault:
        out = {}
        for ft in sorted({r["fault_type"] for r in rows}):
            sub = [r for r in rows if r["fault_type"] == ft]
            import tempfile, os
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
                f.write("\n".join(json.dumps(r) for r in sub))
            out[ft] = run(f.name); os.unlink(f.name)
        return out
    out = {"n": len(rows), "noop": 0, "search_success": 0, "search_exact": 0, "search_solves": [], "expert_success": 0}
    for r in rows:
        broken = Plant.model_validate_json(r["plant_broken"]); orig = Plant.model_validate_json(r["plant_original"])
        site = Site.model_validate_json(r["site"]) if r["site"] else None
        out["noop"] += int(score_patch(broken, orig, [], site).get("success", False))
        patch, n = search_repair(broken, site, hint=r["prompt"][1]["content"].split("RESULT")[-1])
        if patch is not None:
            sc = score_patch(broken, orig, patch, site); out["search_success"] += int(sc["success"]); out["search_exact"] += int(sc["exact"]); out["search_solves"].append(n)
        ev = evaluate(r["completion"][0]["content"], r["plant_broken"], r["plant_original"], r["site"], bool(r["deducible"]))
        out["expert_success"] += int(ev["physics_ok"] and ev["site_ok"] is not False)
    s = out.pop("search_solves"); out["search_mean_solves_when_solved"] = round(st.mean(s), 1) if s else None
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--split", default="test_id"); ap.add_argument("--limit", type=int, default=40); ap.add_argument("--by-fault", action="store_true")
    a = ap.parse_args(); res = run(pathlib.Path(a.data) / f"repair_{a.split}.jsonl", a.limit, a.by_fault)
    print(a.split); [print(" ", k, v) for k, v in (res.items() if a.by_fault else [("all", res)])]
