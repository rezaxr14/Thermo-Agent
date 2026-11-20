#!/usr/bin/env python3
"""Rebuild the per-model bulk plan so every task/tier cell is reachable.

The hosted runner needs an explicit ``tier`` on each shard row, and the datagen
gate counts traces per ``<model>/<task>/<tier>`` cell. The earlier plan files held
only the few cells that happened to be missing at the time (180-200 rows), so the
overnight loop would run out of rows before reaching thousands of traces.

This script plans EVERY prompt in ``prompts_all.jsonl`` for both budget models and
stamps the tier with the same rule the gate uses
(``thermo_agent.synth.tiers.assign_tiers``), so the shard tiers and the reported
coverage cannot drift apart.

Design rows need ``ref_spec`` to be measurable; ``prompts_all.jsonl`` carries only
the compact prompt, so the reference specs are joined in from
``data/frontier_export/design_train.jsonl`` for SCORING ONLY. They are never
written into a shard (that would leak the answer).

Usage
    python scripts/build_budget_plan.py --dry-run
    python scripts/build_budget_plan.py
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "frontier_tools_cohort_prompts"
DESIGN_SOURCE = ROOT / "data" / "frontier_export" / "design_train.jsonl"
MODELS = ("gpt-6-luna", "claude-haiku-5-5")
TASKS = ("repair", "design", "audit", "ledger")


def load_prompts(path: pathlib.Path = PROMPTS / "prompts_all.jsonl") -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    seen, unique = set(), []
    for row in rows:
        rid = str(row.get("id") or "")
        if rid and rid not in seen:
            seen.add(rid)
            unique.append(row)
    return unique


def design_specs(path: pathlib.Path = DESIGN_SOURCE) -> dict[str, str]:
    """id -> ref_spec JSON, used only to measure design complexity."""
    specs: dict[str, str] = {}
    if not path.is_file():
        return specs
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rid = str(row.get("id") or "")
        spec = row.get("ref_spec")
        if rid and spec and rid not in specs:
            specs[rid] = spec if isinstance(spec, str) else json.dumps(spec)
    return specs


def scored_row(task: str, row: dict, specs: dict[str, str]) -> dict:
    if task != "design":
        return row
    spec = specs.get(str(row.get("id") or ""))
    if spec is None:
        return row
    return {**row, "ref_spec": spec}


def plan_model(rows: list[dict], specs: dict[str, str], model: str) -> tuple[list[dict], list[str]]:
    """Return (plan rows, ids that could not be tiered)."""
    from thermo_agent.synth.tiers import TIERS, assign_tiers

    planned: list[dict] = []
    unscored: list[str] = []
    for task in TASKS:
        task_rows = [scored_row(task, r, specs) for r in rows if r.get("task") == task]
        tiers = assign_tiers(task, task_rows)
        for row in task_rows:
            rid = str(row.get("id") or "")
            tier = tiers.get(rid)
            if tier not in TIERS:
                unscored.append(rid)
                continue
            planned.append({"id": rid, "base": rid, "cell": f"{task}/{tier}",
                            "task": task, "tier": tier})
    planned.sort(key=lambda r: (TASKS.index(r["task"]), r["tier"], r["id"]))
    return planned, unscored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompts", default=str(PROMPTS / "prompts_all.jsonl"))
    args = parser.parse_args(argv)

    rows = load_prompts(pathlib.Path(args.prompts))
    specs = design_specs()
    print(f"prompts: {len(rows)} unique; design ref_specs: {len(specs)}")
    if not rows:
        raise SystemExit("no prompts found")

    for model in MODELS:
        planned, unscored = plan_model(rows, specs, model)
        counts = collections.Counter(r["cell"] for r in planned)
        missing = [f"{t}/{tier}" for t in TASKS
                   for tier in ("easy", "medium", "hard") if not counts[f"{t}/{tier}"]]
        payload = {"model": model, "phase": "bulk", "policy": "all-cells-equal-thirds",
                   "rows": planned}
        out = PROMPTS / f"plan_bulk_{model}.json"
        if not args.dry_run:
            out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{model}: {len(planned)} rows, {len(counts)} cells, unscored={len(unscored)}"
              + (f" missing={missing}" if missing else " (all 12 task/tier cells planned)")
              + ("" if args.dry_run else f" -> {out.name}"))
        per_tier = collections.Counter(r["tier"] for r in planned)
        print(f"  tiers: {dict(per_tier)}")
        if unscored:
            print(f"  unscored ids (cannot be tiered, excluded): {unscored[:5]}"
                  + (" ..." if len(unscored) > 5 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
