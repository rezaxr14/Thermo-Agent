#!/usr/bin/env python3
"""Reasoning-quality gate report, per model x task (step 5).

Applies ``validate.reasoning_report`` to the stored native-pilot journals and to the
deterministic tool corpora. The gate is the same one ``transfer.ingest`` enforces:
every tool turn has >= 40 characters of reasoning, every reason is distinct within the
trace, and at least half of the non-first tool turns cite an identifier or number from
an earlier tool result.

    python scripts/reasoning_quality_report.py

Writes ``reports/reasoning_quality_step5.json`` and ``.md``. No model calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re

from thermo_agent.agent import tool_protocol
from thermo_agent.teacher import transfer, validate

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "data/teacher_runs/kaggle_hosted/budget_batches/native_pilot/thermo-teacher-cotproof-s00/23"
JOURNALS = {
    "gpt-6-luna": (BASE / "gpt-6-luna/7332012/native_pilot_completions.jsonl", "7332012"),
    "claude-haiku-5-5": (BASE / "claude-haiku-5-5/7332013/native_pilot_completions.jsonl", "7332013"),
}
EXTRA_EXPORTS = ("data/processed_arch", "data/frontier_prov", "data/a_slice", "data/frontier_export")


def _read(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _turns_from_messages(messages: list[dict]) -> list[dict]:
    """Turn a stored SFT completion (assistant/tool messages) back into gate turns."""
    turns: list[dict] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            content = str(message.get("content") or "")
            think = re.search(r"<think[^>]*>(.*?)</think>", content, re.S)
            turns.append({"assistant": content,
                          "reasoning": think.group(1).strip() if think else "",
                          "tool_calls": message.get("tool_calls"),
                          "tool_results": []})
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                turns[-1]["tool_results"].append(str(messages[index].get("content") or ""))
                index += 1
        else:
            index += 1
    return turns


def _bucket() -> dict:
    return {"rows": 0, "ok": 0, "tool_turns": 0, "reasoned_turns": 0, "distinct": 0,
            "non_first_turns": 0, "cited_turns": 0,
            "failures": collections.Counter()}


def _add(bucket: dict, report: dict) -> None:
    bucket["rows"] += 1
    bucket["ok"] += int(report["ok"])
    for key in ("tool_turns", "reasoned_turns", "distinct", "non_first_turns", "cited_turns"):
        bucket[key] += report[key]
    for failure in report.get("failures") or []:
        bucket["failures"][failure] += 1


def journal_report() -> dict:
    exports = transfer.load_exports(ROOT / "data/processed", transfer.TASKS, "train",
                                    [ROOT / extra for extra in EXTRA_EXPORTS])
    groups: dict[tuple[str, str], dict] = collections.defaultdict(_bucket)
    for model, (path, _run) in JOURNALS.items():
        for record in _read(path):
            row = exports.get(str(record.get("id", "")))
            text = str(record.get("completion") or record.get("sft_completion") or "")
            if row is None or not text or tool_protocol.is_legacy_row(record):
                continue
            task = row.get("task", "repair")
            trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), dict) else {}
            tool_protocol.normalize_tool_turns(trajectory.get("turns") or [])
            _add(groups[(model, task)], validate.reasoning_report(trajectory))
    return {f"{model}/{task}": bucket for (model, task), bucket in sorted(groups.items())}


def deterministic_report(directory: pathlib.Path) -> dict:
    groups: dict[tuple[str, str], dict] = collections.defaultdict(_bucket)
    for path in sorted(directory.glob("*_train.jsonl")):
        task = path.name.split("_", 1)[0]
        for row in _read(path):
            turns = _turns_from_messages(row.get("completion") or [])
            _add(groups[("deterministic", task)], validate.reasoning_report({"turns": turns}))
    return {f"{model}/{task}": bucket for (model, task), bucket in sorted(groups.items())}


def _row(label: str, bucket: dict) -> dict:
    return {"model": label.split("/", 1)[0], "task": label.split("/", 1)[1],
            "rows": bucket["rows"], "gate_ok": bucket["ok"], "gate_fail": bucket["rows"] - bucket["ok"],
            "tool_turns": bucket["tool_turns"], "reasoned_turns": bucket["reasoned_turns"],
            "distinct": bucket["distinct"], "non_first_turns": bucket["non_first_turns"],
            "cited_turns": bucket["cited_turns"],
            "failures": dict(sorted(bucket["failures"].items()))}


def _markdown(title: str, rows: list[dict]) -> str:
    lines = [f"### {title}", "",
             "| Model | Task | Rows | Gate OK | Gate fail | short | repeated | ungrounded | cited / non-first |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        failures = row["failures"]
        lines.append("| {model} | {task} | {rows} | {gate_ok} | {gate_fail} | {short} | {repeated} | "
                     "{ungrounded} | {cited}/{non_first} |".format(
                         short=failures.get("short_reason", 0),
                         repeated=failures.get("repeated_reason", 0),
                         ungrounded=failures.get("ungrounded_reason", 0),
                         cited=row["cited_turns"], non_first=row["non_first_turns"], **row))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/reasoning_quality_step5.json")
    ap.add_argument("--markdown", default="reports/reasoning_quality_step5.md")
    args = ap.parse_args()

    frontier = journal_report()
    deterministic = deterministic_report(ROOT / "data/sft_tools_v3")
    frontier_rows = [_row(label, bucket) for label, bucket in frontier.items()]
    deterministic_rows = [_row(label, bucket) for label, bucket in deterministic.items()]
    payload = {"gate": {"min_chars": validate.MIN_COT_CHARS,
                        "all_reasons_distinct": True,
                        "citation": "at least half of non-first tool turns cite an earlier identifier/number"},
               "frontier_journals": frontier_rows,
               "deterministic_corpus": deterministic_rows}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    markdown = ROOT / args.markdown
    markdown.write_text(_markdown("Native-pilot journals (frontier models)", frontier_rows) + "\n"
                        + _markdown("Deterministic tool corpus (`data/sft_tools_v3`)", deterministic_rows),
                        encoding="utf-8")
    total = sum(row["rows"] for row in frontier_rows)
    ok = sum(row["gate_ok"] for row in frontier_rows)
    det_total = sum(row["rows"] for row in deterministic_rows)
    det_ok = sum(row["gate_ok"] for row in deterministic_rows)
    print(json.dumps({"frontier_rows": total, "frontier_gate_ok": ok,
                      "deterministic_rows": det_total, "deterministic_gate_ok": det_ok,
                      "out": str(out), "markdown": str(markdown)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
