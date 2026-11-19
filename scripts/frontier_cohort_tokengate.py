"""Real-tokenizer gate for frontier-tool traces (drop, never truncate).

Every accepted trace is measured on the FINAL rendered prompt (the same compact
harness builder training/eval use) + the full SFT completion + the repo's 96-token
reserve, with the real Qwen3.5 tokenizer.  Rows over 8,192 are dropped.  A row
over gate ONLY because of provider bookkeeping wrappers (``call_id`` inside
``ToolInvocationResult``) may be rewritten with the existing additive sanitizer
(``scripts/clean_frontier_traces.rewrite_completion``) and re-measured; the
physics answer and every id are left byte-for-byte unchanged.  Nothing is
truncated and no content is fabricated.

Commands::

    .venv/Scripts/python.exe scripts/frontier_cohort_tokengate.py pilot
    .venv/Scripts/python.exe scripts/frontier_cohort_tokengate.py journal \\
        <model> <downloaded.jsonl> <gated.jsonl> <manifest.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clean_frontier_traces import _rewrite_inner, rewrite_completion  # noqa: E402
from scripts.mix_tool_corpora import is_verifier_accepted  # noqa: E402
from thermo_agent.synth import combine  # noqa: E402
from thermo_agent.synth import tiers as tier_mod  # noqa: E402
from thermo_agent.teacher import memory as mem, validate  # noqa: E402
from thermo_agent.teacher.agent_graph import build_context  # noqa: E402

EXPORT = ROOT / "data" / "frontier_export"
PILOT = ROOT / "data" / "frontier_tools_pilot"
COHORT = ROOT / "data" / "frontier_tools_cohort"
QUARANTINE = ROOT / "data" / "overgate_quarantine"
MEMORY_FILES = ("agent_patterns.jsonl", "frontier_notes.jsonl",
                "agent_success_patterns.jsonl")
MAX_TOKENS = 8192


def _notes() -> list:
    out = []
    for name in MEMORY_FILES:
        path = ROOT / "data" / "memory" / name
        if path.exists():
            out += mem.load(path)
    return mem.dedupe(out)


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(m.get("content", "")) for m in value if isinstance(m, dict))
    return str(value or "")


def _rendered_prompt(row: dict, notes: list) -> list[dict]:
    """Compact harness messages exactly as the hosted harness/training render them."""
    row = dict(row)
    if not isinstance(row.get("prompt"), list):
        row["prompt"] = [
            {"role": "system", "content": row.get("system") or ""},
            {"role": "user", "content": row.get("user") or ""},
        ]
    task = row.get("task") or "repair"
    ctx = build_context({"row": row, "task": task, "notes": notes,
                         "k_memory": 3, "mode": "agents+skills", "max_rounds": 6})
    return ctx["messages"]


def measure(messages: list[dict], completion_text: str, tok) -> dict:
    n = 0
    for m in messages:
        n += len(tok.encode(str(m.get("content", "")), add_special_tokens=False).ids)
    n += len(tok.encode(completion_text, add_special_tokens=False).ids)
    n += combine.TOKEN_RESERVE
    return {"tokens": n, "under_gate": n <= MAX_TOKENS}


def sanitize(text: str, trajectory: dict | None = None) -> tuple[str, dict | None, int]:
    """Remove ONLY call_id wrappers; never touch the final answer or physics."""
    new_text, n_text = rewrite_completion(text)
    new_traj, n_traj = None, 0
    if isinstance(trajectory, dict) and isinstance(trajectory.get("turns"), list):
        new_traj = json.loads(json.dumps(trajectory))
        for turn in new_traj["turns"]:
            results = turn.get("tool_results") or []
            for i, value in enumerate(results):
                if isinstance(value, str) and "call_id=" in value:
                    cleaned = _rewrite_inner(value)
                    if cleaned is not None:
                        results[i] = cleaned
                        n_traj += 1
    return new_text, new_traj, n_text + n_traj


def _export_rows() -> dict[str, dict]:
    out = {}
    for task in tier_mod.TASKS:
        path = EXPORT / f"{task}_train.jsonl"
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("task", task)
                    out[str(row["id"])] = row
    return out


def _has_tool_trace(text: str) -> bool:
    return "<tool_call" in text and "<tool_result" in text


def _one(messages: list[dict], completion_text: str, tok, trajectory=None):
    """Return (accept, measurement, sanitized_text, sanitized_traj, drop_reason)."""
    m0 = measure(messages, completion_text, tok)
    if m0["under_gate"]:
        return True, m0, completion_text, trajectory, None
    # Over gate: retry with bookkeeping-only sanitizer.
    new_text, new_traj, changed = sanitize(completion_text, trajectory)
    if changed == 0:
        return False, m0, None, None, "over_gate_no_bookkeeping_wrapper"
    m1 = measure(messages, new_text, tok)
    if not m1["under_gate"]:
        return False, {"before": m0, "after": m1}, None, None, "over_gate_after_sanitize"
    return True, {"before": m0, "after": m1}, new_text, new_traj, None


def gate_pilot() -> None:
    tok = combine.tokenizer()
    notes = _notes()
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "tokenizer": "Qwen3.5 real tokenizer via thermo_agent.synth.combine",
               "reserve": combine.TOKEN_RESERVE, "max_tokens": MAX_TOKENS,
               "models": {}}
    for model in ("gpt-6-luna", "claude-haiku-5-5"):
        src = PILOT / model
        if not src.exists():
            continue
        dest = COHORT / model
        dest.mkdir(parents=True, exist_ok=True)
        qdir = QUARANTINE / model
        qdir.mkdir(parents=True, exist_ok=True)
        m = {"accepted_rows_seen": 0, "accepted_tool_rows_seen": 0,
             "survived": 0, "dropped_over_gate": 0,
             "recovered_by_sanitizer": 0, "token_max_surviving": 0,
             "token_max_raw": 0, "drop_reasons": {}, "per_task": {}}
        for path in sorted(src.glob("frontier_*_train.jsonl")):
            task = path.name.split("_")[1]
            kept, dropped = [], []
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not is_verifier_accepted(row):
                    continue
                m["accepted_rows_seen"] += 1
                text = _text(row.get("completion"))
                if not _has_tool_trace(text):
                    continue  # zero-shot row: never counted as a tool trace
                m["accepted_tool_rows_seen"] += 1
                messages = _rendered_prompt(row, notes)
                ok, measurement, new_text, new_traj, reason = _one(
                    messages, text, tok, row.get("trajectory"))
                raw_n = measurement.get("before", measurement)["tokens"]
                m["token_max_raw"] = max(m["token_max_raw"], raw_n)
                if not ok:
                    m["dropped_over_gate"] += 1
                    m["drop_reasons"][reason] = m["drop_reasons"].get(reason, 0) + 1
                    row["_tokengate"] = {"tokens": raw_n, "reason": reason}
                    dropped.append(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                if isinstance(measurement, dict) and "after" in measurement:
                    m["recovered_by_sanitizer"] += 1
                    row["completion"] = [{"role": "assistant", "content": new_text}]
                    if new_traj is not None:
                        row["trajectory"] = new_traj
                final_n = measurement["tokens"] if "tokens" in measurement else measurement["after"]["tokens"]
                m["token_max_surviving"] = max(m["token_max_surviving"], final_n)
                row["_tokengate"] = {"tokens": final_n, "under_gate": True}
                m["survived"] += 1
                kept.append(json.dumps(row, ensure_ascii=False) + "\n")
            (dest / path.name).write_text("".join(kept), encoding="utf-8")
            if dropped:
                (qdir / path.name).write_text("".join(dropped), encoding="utf-8")
            m["per_task"][task] = {
                "survived": sum(1 for r in kept if json.loads(r).get("task") == task),
                "dropped": len(dropped),
            }
        summary["models"][model] = m
    (COHORT / "TOKENGATE_PILOT.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def gate_journal(model: str, journal: pathlib.Path, out: pathlib.Path,
                 manifest_path: pathlib.Path) -> None:
    tok = combine.tokenizer()
    notes = _notes()
    exports = _export_rows()
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "model": model, "journal": str(journal), "gated": str(out),
               "tokenizer": "Qwen3.5 real tokenizer via thermo_agent.synth.combine",
               "reserve": combine.TOKEN_RESERVE, "max_tokens": MAX_TOKENS,
               "journal_rows": 0, "accepted_tool_rows": 0, "survived": 0,
               "dropped_over_gate": 0, "recovered_by_sanitizer": 0,
               "token_max_raw": 0, "token_max_surviving": 0,
               "drop_reasons": {}, "failure_taxonomy": {}}
    kept = []
    for line in journal.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        summary["journal_rows"] += 1
        if rec.get("error"):
            summary["failure_taxonomy"][rec["error"].split(":")[0]] = \
                summary["failure_taxonomy"].get(rec["error"].split(":")[0], 0) + 1
            continue
        text = _text(rec.get("sft_completion") or rec.get("completion"))
        if not _has_tool_trace(text):
            continue
        rid = str(rec.get("id", ""))
        row = exports.get(rid)
        if row is None:
            summary["failure_taxonomy"]["unknown_id"] = \
                summary["failure_taxonomy"].get("unknown_id", 0) + 1
            continue
        verdict = validate.validate(row.get("task", "repair"), rec.get("completion", ""), row)
        if not verdict.get("accepted"):
            for reason in verdict.get("reasons", []):
                key = str(reason).split(":")[0]
                summary["failure_taxonomy"][key] = summary["failure_taxonomy"].get(key, 0) + 1
            continue
        summary["accepted_tool_rows"] += 1
        messages = _rendered_prompt(row, notes)
        ok, measurement, new_text, new_traj, reason = _one(
            messages, text, tok, rec.get("trajectory"))
        raw_n = measurement.get("before", measurement)["tokens"]
        summary["token_max_raw"] = max(summary["token_max_raw"], raw_n)
        if not ok:
            summary["dropped_over_gate"] += 1
            summary["drop_reasons"][reason] = summary["drop_reasons"].get(reason, 0) + 1
            continue
        if isinstance(measurement, dict) and "after" in measurement:
            summary["recovered_by_sanitizer"] += 1
            rec["sft_completion"] = new_text
            if new_traj is not None:
                rec["trajectory"] = new_traj
        final_n = measurement["tokens"] if "tokens" in measurement else measurement["after"]["tokens"]
        summary["token_max_surviving"] = max(summary["token_max_surviving"], final_n)
        summary["survived"] += 1
        kept.append(json.dumps(rec, ensure_ascii=False) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(kept), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("pilot", "journal"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.action == "pilot":
        gate_pilot()
    else:
        if len(a.args) != 4:
            raise SystemExit("journal needs: <model> <downloaded.jsonl> <gated.jsonl> <manifest.json>")
        gate_journal(a.args[0], pathlib.Path(a.args[1]), pathlib.Path(a.args[2]),
                     pathlib.Path(a.args[3]))


if __name__ == "__main__":
    main()
