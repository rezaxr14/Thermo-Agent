"""Measure what the frontier model actually did with our tools, for the harness-effectiveness claim.

Input is `harness_completions.jsonl` from `kaggle/teacher_harness.ipynb` (or any record list with `task`,
`completion`, `error`, `tool_calls`, `usage`). Output is a small summary: how often tools were called, how many
calls per answer, success rates with and without tool use, and the recorded cost.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics as st

from ..teacher.progress import usage_totals


def load(path) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def summarize(records) -> dict:
    by_task = collections.defaultdict(lambda: {"n": 0, "answered": 0, "with_tools": 0, "calls": 0,
                                               "errors": 0, "reasoning": 0})
    tools = collections.Counter()
    for r in records:
        task = str(r.get("task") or "unknown")
        d = by_task[task]
        d["n"] += 1
        d["answered"] += bool(r.get("completion") and not r.get("error"))
        d["errors"] += bool(r.get("error"))
        calls = r.get("tool_calls") or []
        d["calls"] += len(calls)
        d["with_tools"] += bool(calls)
        d["reasoning"] += bool(r.get("reasoning"))
        for c in calls:
            tools[str(c.get("name", "?"))] += 1
    out = {"n": len(records), "by_task": {}, "tools": dict(tools),
           "usage": usage_totals(records)}
    for task, d in by_task.items():
        out["by_task"][task] = {**d,
                                "calls_per_answer": round(d["calls"] / max(d["answered"], 1), 2),
                                "answer_rate": round(d["answered"] / max(d["n"], 1), 3)}
    return out


def interesting(records, k=3) -> list[dict]:
    """A few concrete interactions: most tool calls, longest reasoning, and a recorded failure."""
    def clean(r):
        return {key: r.get(key) for key in ("id", "task", "model", "harness", "completion", "reasoning",
                                            "tool_calls", "usage", "error", "topology")}
    ranked = sorted(records, key=lambda r: (-len(r.get("tool_calls") or []), str(r.get("id"))))
    reasoned = sorted(records, key=lambda r: (-len(str(r.get("reasoning") or "")), str(r.get("id"))))
    failed = [r for r in records if r.get("error")]
    picked, seen = [], set()
    for r in ranked[:k] + reasoned[:k] + failed[:k]:
        if r.get("id") in seen:
            continue
        seen.add(r.get("id")); picked.append(clean(r))
    return picked[:3 * k]
