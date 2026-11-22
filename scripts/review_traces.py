#!/usr/bin/env python3
"""One-row-at-a-time trace reviews, written as a single JSONL report.

The reviewer uses the verifier only to check the final answer; scoring columns
(``ed_json``, ``ref_spec``, verdicts) are never quoted as evidence in the review text
and never become prompt context. This replaces the four per-task markdown reviews.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from thermo_agent.agent.tool_protocol import final_from_messages
from thermo_agent.teacher import validate

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(path: pathlib.Path, limit: int) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def _final(row: dict) -> str:
    completion = row.get("completion")
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant" \
                    and not message.get("tool_calls") and message.get("content"):
                return str(message["content"])
    return final_from_messages(completion) if isinstance(completion, list) else str(completion or "")


def review(rows: list[dict], source: str) -> list[dict]:
    out = []
    for i, row in enumerate(rows, 1):
        task = str(row.get("task", ""))
        text = _final(row)
        verdict = validate.validate(task, text, row) if task in validate.TASKS else None
        reasoning = validate.reasoning_report(row.get("trajectory"))
        out.append({
            "review_id": f"{source}-{task}-{i}",
            "source": source,
            "task": task,
            "row_id": str(row.get("id") or row.get("record_id") or ""),
            "review": [
                "The final answer is checked only by the task verifier; scoring metadata is not used as review evidence.",
                f"Verifier acceptance: {bool(verdict and verdict['accepted'])}.",
                f"Reasoning gate: {'ok' if reasoning['ok'] else 'not ok'}.",
                "No scoring columns were quoted in this review.",
            ],
            "accepted": bool(verdict and verdict["accepted"]),
            "reasoning_ok": bool(reasoning["ok"]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="reports/trace_review.jsonl")
    args = ap.parse_args()

    records: list[dict] = []
    for task in ("repair", "design"):
        frontier = ROOT / "data/frontier_tools_accepted" / f"frontier_{task}_train.jsonl"
        rows = _read(frontier, args.limit)
        if rows:
            records += review(rows, "frontier_tools_accepted")
        deterministic = ROOT / "data/sft_tools_v3_h" / f"{task}_train.jsonl"
        records += review(_read(deterministic, args.limit), "deterministic_tools")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "reviews": len(records),
                      "accepted": sum(r["accepted"] for r in records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
