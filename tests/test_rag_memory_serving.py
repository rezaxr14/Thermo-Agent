"""RAG serving contract: the model sees only the task, the artefact, the tool
contract and short success/failure recaps -- never row or journal bookkeeping.

These tests cover the two leak surfaces that matter at serving time:

* row-level bookkeeping (ids, run ids, manifests, journal metadata, model names)
  is never serialised into a model-visible message by any of the builders;
* retrieved memory is rendered as an OK/FAIL recap of what worked or failed,
  with the note's id, record id, source model, timestamp and cost/token/latency
  fields stripped before the text reaches the model.

No network / no model calls: rows come from the on-disk processed corpus.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from thermo_agent.agent.harness import augment_with_memory
from thermo_agent.agent.tool_prompt import (
    build_compact_harness_messages,
    build_harness_tool_messages,
)
from thermo_agent.core.schema import Plant
from thermo_agent.teacher import memory
from thermo_agent.teacher.agent_graph import build_context
from thermo_agent.teacher.compact_memory import compact_notes

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("repair", "design", "audit", "ledger")

# Distinctive values so a leak is unambiguous. These are the *kinds* of fields a
# journal/export layer stamps on a row; none of them is part of the task artefact.
ROW_BOOKKEEPING = {
    "run_id": "SENT_RUN_ROW_9F3A",
    "manifest": "SENT_MANIFEST_ROW",
    "export_manifest": "SENT_EXPORT_ROW",
    "journal": "SENT_JOURNAL_ROW",
    "model": "SENT_MODEL_ROW",
    "ts": "SENT_TS_ROW",
    "source": "SENT_SOURCE_ROW",
    "record_id": "SENT_RECORD_ROW",
}
# Bookkeeping smuggled into a note's free-text reasoning fields (the realistic
# leak path: a mining step appending provenance to a lesson or fix).
NOTE_BOOKKEEPING = (
    "google/gemini-3.1 google-gemini-3-1 openai/gpt-4o "
    "run_id=SENT_RUN_NOTE_GH1 record_id=SENT_RECORD_NOTE "
    "ts=1 cost=45 tokens=912 latency=1.7 2026-01-14"
)
NOTE_LEAK_MARKERS = (
    "gemini", "gpt-4o", "SENT_RUN_NOTE_GH1", "SENT_RECORD_NOTE", "SENT_EXTRA_RUN",
    "run_id", "record_id", "ts=", "cost=", "tokens=", "latency", "2026-01-14",
)


def _row(task: str) -> dict:
    path = ROOT / "data" / "processed" / f"{task}_train.jsonl"
    if not path.is_file():
        pytest.skip(f"no real training row at {path}")
    with path.open(encoding="utf-8") as fh:
        line = fh.readline()
    if not line.strip():
        pytest.skip(f"empty corpus row at {path}")
    return json.loads(line)


def _plant(row: dict) -> Plant | None:
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                return Plant.model_validate_json(row[key])
            except Exception:
                pass
    return None


def _messages_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def _bookkept_row(row: dict) -> dict:
    injected = copy.deepcopy(row)
    injected.update(ROW_BOOKKEEPING)
    return injected


# --------------------------------------------------------------------------- #
# 1. row-level bookkeeping never reaches the model, on any serving path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_row_bookkeeping_never_reaches_the_model(task):
    row = _bookkept_row(_row(task))
    plant = _plant(row)
    rendered = [
        _messages_text(build_harness_tool_messages(
            list(row["prompt"]), task, plant)),                 # training / H eval / GRPO
        _messages_text(build_compact_harness_messages(
            list(row["prompt"]), task, plant)),                 # tracegen compact / Z eval
        _messages_text([{"role": "system", "content": build_context(
            {"row": row, "task": task, "notes": []})["system_prompt"]},
            *[m for m in row["prompt"] if m["role"] != "system"]]),  # hosted tracegen
    ]
    for text in rendered:
        for value in ROW_BOOKKEEPING.values():
            assert value not in text, f"{task}: row bookkeeping {value!r} leaked into the prompt"


# --------------------------------------------------------------------------- #
# 2. retrieved memory is a short recap, not a journal dump
# --------------------------------------------------------------------------- #
def _note(nid: str, *, fault: str, lesson: str, fix: str, accepted: bool,
          record_id: str, with_bookkeeping: bool = False) -> memory.Note:
    tail = NOTE_BOOKKEEPING if with_bookkeeping else ""
    return memory.Note(
        id=nid, task="repair", fault_type=fault, architecture="dc_chp",
        symptoms=f"verifier rejected {tail}".strip(),
        lesson=f"{lesson} {tail}".strip(),
        fix=f"{fix} {tail}".strip(),
        source=f"frontier:{tail}" if with_bookkeeping else "frontier:validator",
        accepted=accepted, record_id=record_id,
        extra=({"run_id": "SENT_EXTRA_RUN", "model": "google/gemini-3.1"}
               if with_bookkeeping else {}),
    )


def test_note_bookkeeping_never_reaches_the_model():
    note = _note("n-1", fault="unit_slip",
                 lesson="Attempt rejected. Correct the power of ten.",
                 fix="SET CH.T_gen_out = 355.8", accepted=False,
                 record_id="SENT_RECORD_NOTE", with_bookkeeping=True)
    served = compact_notes([note], k=3)
    assert served
    for marker in NOTE_LEAK_MARKERS:
        assert marker not in served, f"note bookkeeping {marker!r} leaked into memory block"

    # Same note through the actual hosted serving path (query guarantees retrieval).
    row = {"id": "row-current", "prompt": [
        {"role": "system", "content": "You repair plants."},
        {"role": "user", "content": "unit_slip dc_chp power of ten CH.T_gen_out"},
    ]}
    ctx = build_context({"row": row, "task": "repair", "notes": [note], "k_memory": 3})
    prompt = ctx["system_prompt"] + "\n" + row["prompt"][1]["content"]
    for marker in NOTE_LEAK_MARKERS:
        assert marker not in prompt, f"note bookkeeping {marker!r} leaked into the served prompt"
    assert "355.8" in prompt, "the reusable fix pattern must survive"


def test_success_and_failure_recaps_are_served_with_ok_fail_markers():
    ok = _note("ok-1", fault="wrong_fluid",
               lesson="Attempt accepted. Fixed the fluid class at the port.",
               fix="FLUID @CC.fuel_in = Air", accepted=True, record_id="ok-rec")
    fail = _note("fail-1", fault="unit_slip",
                 lesson="Attempt rejected. Correct the power of ten.",
                 fix="SET CH.T_gen_out = 355.8", accepted=False, record_id="fail-rec")
    query = " ".join(memory.note_text(n) for n in (ok, fail))
    row = {"id": "row-current", "prompt": [
        {"role": "system", "content": "You repair plants."},
        {"role": "user", "content": query},
    ]}
    ctx = build_context({"row": row, "task": "repair", "notes": [ok, fail], "k_memory": 3})
    served = ctx["system_prompt"]

    assert "FAILURE MEMORY (what failed" in served
    assert "- OK   wrong_fluid:" in served, "the success recap must be served"
    assert "- FAIL unit_slip:" in served, "the failure recap must be served"
    assert "FLUID @CC.fuel_in = Air" in served and "355.8" in served
    # The recaps are classes, never this row's answer or the notes' bookkeeping.
    assert "ok-rec" not in served and "fail-rec" not in served
    assert ctx["notes_used"], "retrieved notes must be reported as used"


# --------------------------------------------------------------------------- #
# 3. the harness inside the hosted prompt is the shared text (J2/J3 invariant)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_hosted_prompt_carries_the_shared_harness_text(task):
    row = _row(task)
    plant = _plant(row)
    original = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] == "system")
    shared = build_harness_tool_messages(
        [{"role": "system", "content": original}], task, plant)[0]["content"]
    served = build_context({"row": row, "task": task, "notes": []})["system_prompt"]
    assert shared in served, f"{task}: hosted harness differs from the shared builder"


def test_agent_harness_memory_layer_serves_recaps_not_the_verbose_ledger():
    note = _note("n-2", fault="wrong_fluid",
                 lesson="Attempt rejected. Correct the fluid class.",
                 fix="FLUID @CC.fuel_in = Air", accepted=False,
                 record_id="SENT_RECORD_NOTE", with_bookkeeping=True)
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "wrong_fluid dc_chp fluid class"}]
    out = augment_with_memory(msgs, "none", "repair", None,
                              notes=[note], k=3, exclude_record="row-current")
    served = out[0]["content"]
    assert "FAILURE MEMORY (what failed" in served
    assert "KNOWN FAILURE MODES" not in served
    for marker in NOTE_LEAK_MARKERS:
        assert marker not in served, f"note bookkeeping {marker!r} leaked"
