"""Build additive prompt shards for the big-model hosted teacher pass.

Prompts are the same 1,600 ground-truth prompts already exported for the frontier pass,
minus base ids accepted by any earlier model. Old journals and corpora are untouched.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "frontier_export"
FRONTIER = ROOT / "data" / "frontier"
OUT = ROOT / "data" / "kaggle_datasets" / "teacher_big_prompts"
SHARD_SIZE = 200


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    accepted = set()
    for path in sorted(FRONTIER.glob("frontier_*_train.jsonl")):
        for row in read_jsonl(path):
            accepted.add(str(row.get("record_id") or str(row["id"]).split("-t-")[0]))
    rows = []
    for path in sorted(EXPORT.glob("*_train.jsonl")):
        for row in read_jsonl(path):
            if str(row["id"]) in accepted:
                continue
            messages = row["prompt"]
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
            rows.append({
                "id": row["id"], "task": row.get("task"), "topology": row.get("topology"),
                "harness": row.get("harness", "none"),
                "system": system, "user": user,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest(),
                "est_input_tokens": int(row.get("tokens") or 0),
            })
    rows.sort(key=lambda r: (-{"audit": 3, "ledger": 2, "design": 2, "repair": 1}.get(r["task"], 0), r["id"]))
    OUT.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(0, len(rows), SHARD_SIZE):
        name = f"prompts_s{i // SHARD_SIZE:02d}.jsonl"
        chunk = rows[i:i + SHARD_SIZE]
        (OUT / name).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in chunk), encoding="utf-8")
        shards.append(name)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "thermo-agent big-model teacher prompts",
        "id": "rezanadimi/thermo-teacher-prompts-big",
        "licenses": [{"name": "MIT"}],
        "files": [{"name": name, "description": "compact prompt shard (no ground truth)"} for name in shards],
    }, indent=1) + "\n", encoding="utf-8")
    from collections import Counter
    print(json.dumps({
        "already_accepted": len(accepted), "available": len(rows), "shards": shards,
        "by_task": dict(Counter(r["task"] for r in rows)),
    }, indent=1))


if __name__ == "__main__":
    main()
