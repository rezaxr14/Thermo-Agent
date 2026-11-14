"""python -m thermo_agent.eval.summarize reports/*.json : one markdown table to paste back."""
from __future__ import annotations
import json, sys, pathlib

KEYS = {"repair": ["n", "success", "first_turn_success", "exact_on_deducible", "mean_solves"], "design": ["n", "valid", "mean_gap_pct", "within_1pct", "mean_evals"],
        "audit": ["n", "format", "rank", "valid", "gain", "pred", "grounded"], "ledger": ["n", "format", "closure", "nonneg", "exact", "all_correct"]}


def _cell(v: dict, k: str) -> str:
    ci = v.get(k + "_ci95")
    return f"{v.get(k)} [{ci[0]}-{ci[1]}]" if ci else str(v.get(k))


def table(paths) -> str:
    lines = []
    for task, keys in KEYS.items():
        lines += [f"\n### {task}", "| report | split | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        for p in paths:
            rep = json.loads(pathlib.Path(p).read_text()); m = rep["meta"]
            name = f"{pathlib.Path(p).stem} ({m.get('adapter') or 'base'}, harness={m['harness']})"
            for sp, v in (rep.get(task) or {}).items():
                if v: lines.append(f"| {name} | {sp} | " + " | ".join(_cell(v, k) for k in keys) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(sys.argv[1:]))
