"""Build the next 100-row luna/haiku shards from the unique-diagram plans.

Reads the live gate coverage, skips ids already attempted by a model, and picks
rows from that model's missing cells first. Writes per-model shards and an
updated harness_config.json into data/frontier_tools_cohort_prompts/; the caller
then syncs that dir to HF and refreshes the Kaggle dataset via the bridge kernel.
No Kaggle calls, no model calls, no local data upload to Kaggle.
"""
from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/frontier_tools_cohort_prompts"
COHORT = ROOT / "data/frontier_tools_cohort"
JOURNALS = ROOT / "data/teacher_runs/kaggle_hosted/budget_batches"
MODELS = ("gpt-6-luna", "claude-haiku-5-5")
BATCH_SIZE = 100


def gate_coverage() -> dict[str, dict[str, int]]:
    proc = subprocess.run([sys.executable, "scripts/check_datagen_complete.py"],
                          cwd=ROOT, capture_output=True, text=True)
    report = json.loads(proc.stdout)
    item = next(i for i in report["items"] if i["name"].startswith("frontier CoT"))
    detail = item["detail"]
    if "; coverage=" not in detail:
        raise SystemExit("gate detail has no coverage map")
    return ast.literal_eval(detail.split("; coverage=", 1)[1])


def used_ids(model: str) -> set[str]:
    used: set[str] = set()
    for path in (COHORT / model).glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                used.add(str(json.loads(line).get("record_id") or json.loads(line).get("id")))
    if JOURNALS.is_dir():
        for path in JOURNALS.rglob("*.jsonl"):
            if model.replace("/", "-") not in str(path).replace("\\", "/").lower():
                continue
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    used.add(str(json.loads(line).get("id") or ""))
    return {u.split("#v", 1)[0] for u in used if u}


def batch_number() -> int:
    n = 1
    while list(PROMPTS.glob(f"prompts_batch{n}_*.jsonl")):
        n += 1
    return n


def main() -> int:
    coverage = gate_coverage()
    all_rows = {}
    for line in (PROMPTS / "prompts_all.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            all_rows[str(row["id"])] = row
    n = batch_number()
    model_shards = {}
    for model in MODELS:
        plan = json.loads((PROMPTS / f"plan_bulk_{model}.json").read_text(encoding="utf-8"))
        used = used_ids(model)
        # Gate keys are "<model>/<task>/<tier>"; plan cells are "<task>/<tier>".
        missing = {key.split("/", 1)[1] for key, counts in coverage.items()
                   if key.startswith(model + "/") and any(v == 0 for v in counts.values())}
        # Equal thirds across easy/medium/hard for every 100-row batch; within
        # each tier prefer cells that are still missing in the gate.
        quotas = {"easy": BATCH_SIZE // 3, "medium": BATCH_SIZE // 3,
                  "hard": BATCH_SIZE - 2 * (BATCH_SIZE // 3)}
        by_tier = {"easy": [], "medium": [], "hard": []}
        for entry in plan["rows"]:
            rid = str(entry["id"])
            base = str(entry.get("base") or rid).split("#v", 1)[0]
            if base in used or rid not in all_rows:
                continue
            cell = str(entry.get("cell") or "")
            tier = cell.split("/")[-1]
            if tier not in by_tier:
                continue
            priority = 0 if cell in missing else 1
            by_tier[tier].append((priority, rid, entry))
        chosen = []
        for tier, quota in quotas.items():
            rows = sorted(by_tier[tier], key=lambda x: (x[0], x[1]))
            chosen.extend(rows[:quota])
        if not chosen:
            raise SystemExit(f"{model}: no unused rows left in plan")
        shard = PROMPTS / f"prompts_batch{n}_{model}.jsonl"
        shard.write_text("".join(json.dumps(all_rows[rid], ensure_ascii=False) + "\n"
                                 for _, rid, _ in chosen), encoding="utf-8")
        model_shards[model] = shard.name
        print(f"{model}: batch {n} {len(chosen)} rows, "
              f"{sum(1 for p, _, _ in chosen if p == 0)} from missing cells -> {shard.name}")
    config = {
        "mode": "harness", "shard": "prompts_proof.jsonl", "limit": BATCH_SIZE,
        "rounds": 6, "k_memory": 3, "deadline_seconds": 0, "max_cost_usd": 0.0,
        "journal": f"batch{n}_completions.jsonl",
        "manifest": f"batch{n}_manifest.json",
        "memory_dir": "", "export_dir": "",
        "model_shards": model_shards,
        "tiers": ["easy", "medium", "hard"],
        "equal_tiers": True,
        "tier_field": "tier",
        "cell_field": "cell",
        "pipeline": {"gate_tokens": 8192, "ingest": True,
                     "memory": True, "mix_refresh": True},
        "allowed_models": list(MODELS),
        "code_dataset": "rezanadimi/thermo-agent-data",
        "data_dataset": "rezanadimi/thermo-agent-data",
    }
    (PROMPTS / "harness_config.json").write_text(json.dumps(config, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"batch {n} config written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
