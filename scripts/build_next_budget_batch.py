"""Build the next 100-row luna/haiku shards from the unique-diagram plans.

Reads the live gate coverage, skips ids already attempted by a model, and picks
rows from that model's missing cells first. Writes per-model shards and an
updated harness_config.json into data/frontier_tools_cohort_prompts/; the caller
then syncs that dir to HF and refreshes the Kaggle dataset via the bridge kernel.
No Kaggle calls, no model calls, no local data upload to Kaggle.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/frontier_tools_cohort_prompts"
COHORT = ROOT / "data/frontier_tools_cohort"
JOURNALS = ROOT / "data/teacher_runs/kaggle_hosted/budget_batches"
MODELS = ("gpt-6-luna", "claude-haiku-5-5")
BATCH_SIZE = 100
TASKS = ("repair", "design", "audit", "ledger")
TIERS = ("easy", "medium", "hard")


def gate_missing_cells() -> list[str]:
    """Parse the explicit 'missing cells' list from the live gate detail."""
    proc = subprocess.run([sys.executable, "scripts/check_datagen_complete.py"],
                          cwd=ROOT, capture_output=True, text=True)
    report = json.loads(proc.stdout)
    item = next(i for i in report["items"] if i["name"].startswith("frontier CoT"))
    # The gate detail appends coverage after the list ("] ...; coverage=..."),
    # so do not require a semicolon immediately after the bracket.
    match = re.search(r"missing cells: (\[.*?\])", item["detail"])
    if not match:
        raise SystemExit("gate detail has no missing-cell list")
    return list(ast.literal_eval(match.group(1)))


def used_ids(model: str) -> set[str]:
    """Only ACCEPTED traces count as used; a failed question may be retried.

    Journals contain every attempt, including failures that never enter training.
    Counting them would permanently exclude exactly the cells we still need.
    """
    used: set[str] = set()
    for path in (COHORT / model).glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                used.add(str(json.loads(line).get("record_id") or json.loads(line).get("id")))
    return {u.split("#v", 1)[0] for u in used if u}


def batch_number() -> int:
    n = 1
    while list(PROMPTS.glob(f"prompts_batch{n}_*.jsonl")):
        n += 1
    return n


def interleave(tier_rows, quota):
    """Pick `quota` rows, taking one per (priority, cell) in turn.

    Sorting by id alone starves whole cells: audit ids sort before ledger ids, so a
    100-row batch could spend every easy slot on audit/easy and never touch
    ledger/easy. Missing cells (priority 0) are still served before present ones.
    """
    buckets = {}
    for priority, rid, entry in tier_rows:
        key = (priority, str(entry.get("cell") or ""))
        buckets.setdefault(key, []).append((priority, rid, entry))
    for key in buckets:
        buckets[key].sort(key=lambda x: x[1])
    keys = sorted(buckets)
    idx = {k: 0 for k in keys}
    take = []
    while len(take) < quota and any(idx[k] < len(buckets[k]) for k in keys):
        for k in keys:
            if len(take) >= quota:
                break
            if idx[k] < len(buckets[k]):
                take.append(buckets[k][idx[k]])
                idx[k] += 1
    return take


def stamp_rows(chosen, all_rows):
    """Give every chosen row an explicit tier/cell, or refuse to build the shard.

    The hosted runner filters on `tier`; a shard without it loads fine and then
    selects 0 rows, which is how two batches produced 0-byte journals.
    """
    lines = []
    counts = {t: 0 for t in TIERS}
    for _, rid, entry in chosen:
        row = dict(all_rows[rid])
        cell = str(entry.get("cell") or "")
        tier = cell.split("/")[-1]
        if tier not in TIERS or "/" not in cell:
            raise SystemExit(f"row {rid} has no usable cell/tier ({cell!r})")
        row["cell"] = cell
        row["tier"] = tier
        counts[tier] += 1
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    return lines, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=BATCH_SIZE,
                    help="rows per model (default 100; use 40 for the reasoning pilot)")
    ap.add_argument("--name", default=None,
                    help="batch tag for shard/journal names (default: auto batchN)")
    a = ap.parse_args(argv)
    size = a.size
    n = batch_number()
    tag = a.name or f"batch{n}"
    missing_all = gate_missing_cells()
    all_rows = {}
    for line in (PROMPTS / "prompts_all.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            all_rows[str(row["id"])] = row
    model_shards = {}
    for model in MODELS:
        plan = json.loads((PROMPTS / f"plan_bulk_{model}.json").read_text(encoding="utf-8"))
        used = used_ids(model)
        # Gate keys are "<model>/<task>/<tier>"; plan cells are "<task>/<tier>".
        missing = {cell.split("/", 1)[1] for cell in missing_all
                   if cell.startswith(model + "/")}
        # Equal rows per task AND equal thirds across easy/medium/hard. A plain
        # cell-sorted interleave starves tasks whose names sort late (audit gets
        # 12 of a 40-row batch), so allocate per task first, then per tier.
        by_task_tier: dict[tuple[str, str], list] = {
            (t, tier): [] for t in TASKS for tier in TIERS}
        for entry in plan["rows"]:
            rid = str(entry["id"])
            base = str(entry.get("base") or rid).split("#v", 1)[0]
            if base in used or rid not in all_rows:
                continue
            cell = str(entry.get("cell") or "")
            task, tier = cell.split("/")[-2:] if "/" in cell else ("", "")
            if (task, tier) not in by_task_tier:
                continue
            priority = 0 if cell in missing else 1
            by_task_tier[(task, tier)].append((priority, rid, entry))
        per_task = size // len(TASKS)
        base = per_task // 3
        extra = per_task - 3 * base
        chosen = []
        for ti, task in enumerate(TASKS):
            # Rotate which tier gets the rounding remainder. Each task keeps
            # exactly `per_task` rows, and the shard-wide tier counts differ by
            # at most one (13/13/14 for a 40-row pilot), so the hosted runner's
            # select_rows keeps ALL rows instead of trimming to the min tier.
            quotas = {tier: base + (1 if extra and tier == TIERS[ti % len(TIERS)] else 0)
                      for tier in TIERS}
            for tier, quota in quotas.items():
                chosen.extend(interleave(by_task_tier[(task, tier)], quota))
        if not chosen:
            raise SystemExit(f"{model}: no unused rows left in plan")
        shard = PROMPTS / f"prompts_{tag}_{model}.jsonl"
        lines, counts = stamp_rows(chosen, all_rows)
        shard.write_text("".join(lines), encoding="utf-8")
        model_shards[model] = shard.name
        print(f"{model}: {tag} {len(chosen)} rows tiers={counts}, "
              f"{sum(1 for p, _, _ in chosen if p == 0)} from missing cells -> {shard.name}")
    config = {
        "mode": "harness", "shard": "prompts_proof.jsonl", "limit": size,
        "rounds": 6, "k_memory": 3, "deadline_seconds": 0, "max_cost_usd": 0.0,
        "journal": f"{tag}_completions.jsonl",
        "manifest": f"{tag}_manifest.json",
        "memory_dir": "", "export_dir": "",
        "model_shards": model_shards,
        "tiers": ["easy", "medium", "hard"],
        "equal_tiers": True,
        "tier_field": "tier",
        "cell_field": "cell",
        "pipeline": {"gate_tokens": 8192, "ingest": True,
                     "memory": True, "mix_refresh": True,
                     "dynamic_tools": True},
        "allowed_models": list(MODELS),
        "code_dataset": "rezanadimi/thermo-agent-data",
        "data_dataset": "rezanadimi/thermo-agent-data",
    }
    (PROMPTS / "harness_config.json").write_text(json.dumps(config, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"{tag} config written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
