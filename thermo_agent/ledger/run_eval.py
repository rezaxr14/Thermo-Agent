"""python -m thermo_agent.ledger.run_eval DATA_DIR --split test_id --limit 200 : non-model baselines for ledger repair.

 no_op        change nothing
 regenerate   rebuild every line from the topology and the standard stream template (what a program that already knows the
              balance form would do; solves the task by construction, so the benchmark measures whether a model can apply that
              rule from text, not whether the rule is hard)
 symptoms     how often the second-law checks reveal an equation fault: closure residual != 0, a negative ED, or either.
              Parameter-range checks (V6) see none of these."""
from __future__ import annotations
import argparse, collections, json, pathlib
from . import program as P


def symptoms(row: dict) -> dict:
    L = json.loads(row["ledger"]); data, lines = L["data"], L["lines"]; r = P.run(lines, data); ef = max(data["e_fuel_mw"], 1e-6)
    closure = r["closure"] is not None and abs(r["closure"]) > 1e-6 * ef
    negative = any(v < -1e-5 * ef for v in r["ED"].values())
    return {"closure": closure, "negative_ed": negative, "either": closure or negative}


def run(path, limit=None) -> dict:
    rows = [json.loads(l) for l in open(path)][:limit]; by = collections.defaultdict(lambda: collections.Counter()); regen = 0
    for r in rows:
        L = json.loads(r["ledger"]); data = L["data"]
        fixed = P.run(data["lines"], data); regen += all(P.matches_truth(fixed, data, k) for k in data["lines"])       # regenerate = the correct ledger
        s = symptoms(r)
        for t in r["fault_types"]:
            by[t]["n"] += 1
            for k, v in s.items(): by[t][k] += v
    n = len(rows)
    return {"n": n, "no_op_solved": 0.0, "regenerate_solved": round(regen / max(n, 1), 3),
            "symptom_rate_by_fault": {t: {k: round(c[k] / c["n"], 2) for k in ("closure", "negative_ed", "either")} | {"n": c["n"]} for t, c in sorted(by.items())}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--split", default="test_id"); ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(); print(a.split, json.dumps(run(pathlib.Path(a.data) / f"ledger_{a.split}.jsonl", a.limit), indent=1))
