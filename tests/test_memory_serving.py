"""D11: the agent is served a token-efficient "what failed -> what works" memory, never bookkeeping.

The full ledger (model ids, timestamps, cost, latency, token counts, run ids) stays on disk
for our own reports. Only the reasoning-carrying failure class + fix pattern reach the model.
These tests exercise the leak path through Note.lesson / symptoms / fix (not just `source`,
which compact_notes never reads at all).
"""
from __future__ import annotations

import json
import re

from thermo_agent.teacher import memory as mem
from thermo_agent.teacher.agent_graph import build_context
from thermo_agent.teacher.compact_memory import compact_notes

# Every piece of bookkeeping the model must never see.
LEAK_MARKERS = (
    "gemini", "gpt-", "claude", "haiku", "luna",
    "openai/", "anthropic/", "google/",
    "ts=", "cost=", "tokens=", "latency",
)
TIMESTAMP_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")

# A realistic provenance/timing tail that the mining step can append to a note string.
META = (
    "google/gemini-3.1-flash google-gemini-3-1 openai/gpt-4o gpt-4o "
    "anthropic/claude-3-5-haiku claude-3-haiku luna-2 "
    "ts=1 cost=45 tokens=912 latency=1.7 2026-01-14"
)


def _note(i, fault, symptom, lesson_core, fix, *, extra=None):
    """A rejected note whose reasoning fields carry a bookkeeping tail."""
    return mem.Note(
        id=f"n{i}",
        task="repair",
        fault_type=fault,
        architecture="dc_absorption",
        symptoms=f"{symptom} {META}",
        lesson=f"Attempt rejected. Correct {lesson_core}. {META}",
        fix=(f"{fix} {META}" if fix else ""),
        source=f"frontier:google-gemini-3.1-flash-lite-preview ts=1 cost=45 run_id=run-{i}",
        accepted=False,
        record_id=f"rep-{i}",
        extra=extra or {},
    )


NOTES = [
    _note(0, "unit_slip", "verifier: invalid plant",
          "the power of ten, not the digits", "SET CH.T_gen_out = 355.8",
          extra={"model": "google/gemini-3.1-flash", "ts": 1, "cost": 45,
                 "latency": 1.7, "tokens": 912, "run_id": "run-0"}),
    _note(1, "wrong_fluid", "unknown component ids ['DC']",
          "the fluid class at the port", "FLUID @CC.fuel_in = Air"),
    _note(2, "sign_flip", "negative exergy destruction",
          "the sign of the heat duty", ""),  # no fix -> hint comes from lesson
    _note(3, "swap_ports", "E_TEMP_CROSS hot/cold swapped",
          "the hot and cold endpoints", ""),  # no fix -> hint comes from lesson
    _note(4, "eta_limit", "efficiency above reversible limit",
          "eta must be a fraction", "SET CC.eta_s = 0.85"),
]


def _row():
    """Minimal row whose query tokens guarantee retrieval of all NOTES."""
    query = " ".join(mem.note_text(n) for n in NOTES)
    return {
        "id": "row-serving-test",
        "prompt": [
            {"role": "system", "content": "You are a plant engineer."},
            {"role": "user", "content": query},
        ],
    }


# (a) compact is strictly shorter than the verbose ledger format for the same notes.
def test_compact_is_strictly_shorter_than_verbose():
    compact = compact_notes(NOTES, k=5)
    verbose = mem.format_notes(NOTES)
    assert compact and verbose
    assert len(compact) < len(verbose), (len(compact), len(verbose))


# (b) no bookkeeping of any kind reaches the model.
def test_compact_block_leaks_no_bookkeeping():
    out = compact_notes(NOTES, k=5)
    for marker in LEAK_MARKERS:
        assert marker not in out, f"leaked {marker!r} in: {out}"
    assert not TIMESTAMP_RE.search(out), f"leaked a timestamp in: {out}"
    assert "run-" not in out and "record" not in out, f"leaked an id in: {out}"


# (c) the reasoning that matters survives: failure class + numeric fix pattern.
def test_compact_keeps_failure_class_and_fix_pattern():
    out = compact_notes(NOTES, k=5)
    assert "unit_slip" in out
    assert "355.8" in out
    assert "FAILURE MEMORY (what failed" in out
    assert "->" in out  # "what failed -> what works"


# (d) the system prompt carries the compact guidance + memory, not the verbose block.
def test_build_context_serves_guidance_and_compact_memory():
    sys_prompt = build_context({
        "row": _row(), "task": "repair", "notes": NOTES,
        "k_memory": 5, "mode": "agents+skills",
    })["system_prompt"]
    assert "FAILURE MEMORY (what failed" in sys_prompt
    assert "FIRST inventory" in sys_prompt
    assert "UNIT MISMATCH" in sys_prompt
    assert "KNOWN FAILURE MODES" not in sys_prompt
    for marker in LEAK_MARKERS:
        assert marker not in sys_prompt, f"system prompt leaked {marker!r}"


# (e) token efficiency, quantified: compact <= ~60% of verbose for a 5-note set.
def test_token_efficiency_at_most_60_percent():
    compact = compact_notes(NOTES, k=5)
    verbose = mem.format_notes(NOTES)
    ratio = len(compact) / len(verbose)
    print(f"\ncompact={len(compact)} chars, verbose={len(verbose)} chars, "
          f"ratio={ratio:.1%}")
    assert ratio <= 0.60, (len(compact), len(verbose), ratio)


# The full ledger is intentionally kept on disk for our reports; only the served block is lean.
def test_full_ledger_retains_bookkeeping():
    d = NOTES[0].to_dict()
    assert "gemini" in d["source"]
    assert d["extra"]["model"] == "google/gemini-3.1-flash"
    assert d["extra"]["run_id"] == "run-0"
    assert json.dumps(d)  # serialisable for on-disk jsonl
    # The same note contributes reasoning, never bookkeeping, to the agent.
    out = compact_notes([NOTES[0]])
    assert "unit_slip" in out and "355.8" in out
    for marker in LEAK_MARKERS:
        assert marker not in out
