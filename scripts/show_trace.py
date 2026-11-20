#!/usr/bin/env python3
"""Print one accepted frontier trace in full, for human quality review.

The accepted cohort stores a structured ``trajectory`` (turns with tool calls and
tool results), the prompt messages, the verifier verdict and the ids of the memory
notes that were served. This reader renders all of it, resolves the memory note ids
back to the note text, and reconstructs the one task skill the harness would have
served.

The served skill is chosen at request time by ``TASK_SKILLS[task]``; newer rows
persist its id and exact system-prompt hash, older rows show the reconstructed id.

Per-turn reasoning: for chat-completions models (including luna) the visible
assistant text IS the reasoning trace, so it is captured in ``turn["assistant"]``
and mirrored to ``turn["reasoning"]``. A hidden provider trace, when present, is
shown instead of the visible text. Absence in an older row means the row predates
this capture fix, not that the model produced no reasoning.

Usage
    python scripts/show_trace.py --id rep-11-00050-t-claude-haiku-5-5
    python scripts/show_trace.py --best                 # richest rewarded trace
    python scripts/show_trace.py --best --task design
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "frontier_tools_cohort"
MEMORY = ROOT / "data" / "memory"
MAX_RESULT_CHARS = 2400


# --------------------------------------------------------------------------- rows
def iter_rows(base: pathlib.Path = COHORT):
    for path in sorted(base.glob("*/*.jsonl")):
        model = path.parent.name
        for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row["_file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            row["_model_dir"] = model
            yield row


def find_row(trace_id: str, base: pathlib.Path = COHORT) -> dict:
    wanted = trace_id.split("#v", 1)[0]
    for row in iter_rows(base):
        if str(row.get("id")) == trace_id:
            return row
    for row in iter_rows(base):
        if str(row.get("record_id") or "").split("#v", 1)[0] == wanted:
            return row
    raise SystemExit(f"trace {trace_id!r} not found under {base}")


def trace_score(row: dict) -> tuple:
    """Rank traces for `--best`: rewarded, more distinct tools, more prose, more notes."""
    traj = row.get("trajectory") or {}
    turns = traj.get("turns") or []
    calls = [c.get("name") for t in turns for c in (t.get("tool_calls") or [])]
    prose = sum(1 for t in turns if (t.get("assistant") or "").strip())
    reward = float((row.get("validation") or {}).get("reward") or 0.0)
    return (reward >= 1.0, len(set(calls)), prose, len(traj.get("notes_used") or []),
            len(turns), float(traj.get("_tokengate", {}).get("tokens") or 0))


def best_row(task: str | None = None, base: pathlib.Path = COHORT) -> dict:
    rows = [r for r in iter_rows(base) if not task or r.get("task") == task]
    if not rows:
        raise SystemExit(f"no accepted traces for task={task!r} under {base}")
    return max(rows, key=trace_score)


# ------------------------------------------------------------------------- memory
def memory_index(base: pathlib.Path = MEMORY) -> dict[str, dict]:
    index: dict[str, dict] = {}
    if not base.is_dir():
        return index
    for path in sorted(base.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                note = json.loads(line)
            except ValueError:
                continue
            if note.get("id"):
                index[str(note["id"])] = note
    return index


def compact_note_line(note: dict) -> str:
    """Mirror the compact memory renderer: class + reusable fix, no ids/models."""
    fault = str(note.get("fault_type") or note.get("task") or "")
    fix = str(note.get("fix") or "").strip()
    lesson = str(note.get("lesson") or "").strip()
    hint = fix or (lesson.split(". Correct", 1)[-1].lstrip() if ". Correct" in lesson else lesson)
    marker = "OK  " if note.get("accepted") in (True, "True", "true", 1) else "FAIL"
    return f"- {marker} {fault}:" + (f" {hint}" if hint else "")


# --------------------------------------------------------------------------- skill
def served_skill(task: str) -> tuple[str, str]:
    """(skill name, compact skill text) the harness would serve for this task."""
    try:
        from thermo_agent.agent import harness as H
        from thermo_agent.agent.tool_prompt import _compact_selected_skill
    except Exception as exc:                      # package not importable
        return "", f"(unavailable: {exc})"
    names = list(H.TASK_SKILLS.get(task) or [])
    if not names:
        return "", ""
    return names[0], _compact_selected_skill(task)


# -------------------------------------------------------------------------- render
def pretty_result(value: str) -> str:
    """Tool results are JSON strings; unwrap ToolInvocationResult reprs when present."""
    text = str(value)
    if text.startswith("ToolInvocationResult("):
        match = re.search(r"output='(.*)'\)\s*$", text, re.S)
        if match:
            text = match.group(1).encode().decode("unicode_escape", "replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} chars trimmed]"


def render(row: dict, notes: dict[str, dict]) -> str:
    traj = row.get("trajectory") or {}
    turns = traj.get("turns") or []
    provenance = row.get("provenance") or {}
    validation = row.get("validation") or {}
    out: list[str] = []

    def head(title: str) -> None:
        out.append("")
        out.append("=" * 78)
        out.append(title)
        out.append("=" * 78)

    head("TRACE")
    out += [
        f"id         : {row.get('id')}",
        f"task       : {row.get('task')}",
        f"record_id  : {row.get('record_id')}",
        f"model      : {row.get('model') or provenance.get('model')}",
        f"provider   : {row.get('provider') or provenance.get('provider')}",
        f"run_id     : {row.get('run_id') or provenance.get('run_id')}",
        f"source     : {row.get('source')}  (provenance.source_family="
        f"{provenance.get('source_family')}, generator={provenance.get('generator')})",
        f"file       : {row.get('_file')}",
        f"token gate : {json.dumps(row.get('_tokengate') or {}, sort_keys=True)}",
    ]

    head("PROMPT SERVED (row system/user messages)")
    for message in row.get("prompt") or []:
        out.append(f"\n[{message.get('role')}]")
        out.append(str(message.get("content") or ""))

    name, skill = served_skill(str(row.get("task") or ""))
    recorded_skill = str(traj.get("served_skill") or "")
    system_hash = str(traj.get("system_prompt_sha256") or "")
    head("SKILL SERVED")
    if recorded_skill:
        out.append(f"skill: {recorded_skill}  (persisted in the trace)")
        if name:
            out.append("")
            out.append(truncate(skill, 4000))
        if system_hash:
            out.append(f"\nsystem prompt sha256: {system_hash}")
    elif not name:
        out.append(f"(no skill is mapped for task {row.get('task')!r} in TASK_SKILLS; none was served)")
    else:
        out.append(f"skill: {name}  (reconstructed from TASK_SKILLS: this row predates capture)")
        out.append("")
        out.append(truncate(skill, 4000))

    head("TOOLS OFFERED TO THE MODEL")
    out.append(", ".join(traj.get("tools_available") or []) or "(none)")

    served = list(traj.get("notes_used") or [])
    head(f"MEMORY SERVED ({len(served)} notes)")
    if not served:
        out.append("(no memory notes were served for this row)")
    for note_id in served:
        note = notes.get(note_id)
        out.append(f"\n[{note_id}]")
        if not note:
            out.append("  (id not found in data/memory/*.jsonl)")
            continue
        out.append(f"  class   : {note.get('fault_type') or note.get('task')}")
        out.append(f"  accepted: {note.get('accepted')}")
        out.append(f"  lesson  : {note.get('lesson')}")
        if note.get("fix"):
            out.append(f"  fix     : {note.get('fix')}")
        out.append(f"  served  : {compact_note_line(note)}")
    memory_block = str(traj.get("memory_block") or "")
    if memory_block:
        out.append("\nExact block sent to the model:")
        out.append(memory_block)

    head(f"TRAJECTORY ({len(turns)} turns)")
    for i, turn in enumerate(turns, 1):
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        out.append(f"\n--- turn {i} ---")
        assistant = str(turn.get("assistant") or "").strip()
        reasoning = str(turn.get("reasoning") or "").strip()
        # Visible assistant prose is the reasoning for chat-completions models.
        shown_reasoning = reasoning or assistant
        out.append(f"assistant: {assistant if assistant else '(empty: tool-only turn)'}")
        out.append("reasoning: " + (shown_reasoning if shown_reasoning
                                    else "(none in this turn)"))
        if not calls:
            out.append("tool calls: none (final answer turn)")
            continue
        for j, call in enumerate(calls):
            out.append(f"\n  call {j + 1}: {call.get('name')}")
            out.append("  args  : " + json.dumps(call.get("arguments") or {}, ensure_ascii=False))
            if j < len(results):
                out.append("  result:\n" + "\n".join(
                    "    " + line for line in truncate(pretty_result(results[j])).splitlines()))

    head("FINAL ANSWER")
    final = traj.get("final")
    out.append(final if isinstance(final, str) else json.dumps(final, indent=2, ensure_ascii=False))

    head("VERIFIER")
    out += [f"reward : {validation.get('reward')}",
            f"reasons: {validation.get('reasons')}",
            f"details: {json.dumps({k: v for k, v in validation.items() if k not in ('reward', 'reasons')})}"]

    head("CAPTURE AUDIT")
    calls = [c.get("name") for t in turns for c in (t.get("tool_calls") or [])]
    # assistant prose is the reasoning trace for chat-completions models.
    def _reason(t):
        return (str(t.get("reasoning") or "").strip()
                or str(t.get("assistant") or "").strip())
    out += [
        f"tool calls captured      : {len(calls)} {sorted(set(calls))}",
        f"assistant prose turns    : {sum(1 for t in turns if (t.get('assistant') or '').strip())}/{len(turns)}",
        f"reasoning turns captured : {sum(1 for t in turns if _reason(t))}/{len(turns)}",
        f"memory notes served      : {len(served)} (ids persisted"
        + (", exact block persisted" if str(traj.get('memory_block') or '') else
           ", text resolved from data/memory") + ")",
        ("skill id persisted      : " + recorded_skill) if recorded_skill
        else ("skill id persisted      : no - reconstructed: " + name if name
              else "skill id persisted      : no skill mapped for this task"),
        ("system prompt sha256     : " + system_hash) if system_hash
        else "served system prompt     : NOT persisted (row predates capture)",
        f"tool results present     : {sum(len(t.get('tool_results') or []) for t in turns)}",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", help="exact trace id (or record_id)")
    group.add_argument("--best", action="store_true",
                       help="pick the richest rewarded trace (optionally within --task)")
    parser.add_argument("--task", choices=["repair", "design", "audit", "ledger"])
    parser.add_argument("--json", action="store_true", help="also dump the raw row as JSON")
    args = parser.parse_args(argv)

    row = best_row(args.task) if args.best else find_row(args.id)
    text = render(row, memory_index())
    sys.stdout.write(text + "\n")
    if args.json:
        sys.stdout.write("\n" + json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
