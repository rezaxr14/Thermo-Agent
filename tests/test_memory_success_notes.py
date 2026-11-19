"""Verifier-accepted attempts must become reusable "what worked" memory, served token-efficiently.

The measured gap: every note in data/memory/*.jsonl was accepted=False, so the harness could only ever see
failure lessons.  These tests pin the four behaviours that close it: success_note, ingest, balanced retrieval,
and a compact renderer that marks OK/FAIL without leaking bookkeeping.

No network / no model calls: rows come from the on-disk processed corpus (or cleanly skip without it).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from thermo_agent.teacher import memory, transfer, validate
from thermo_agent.teacher.compact_memory import compact_notes

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TASKS = ("repair", "ledger", "audit", "design")

# Bookkeeping that must never reach the model through lesson/fix.
BOOKKEEPING = ("record_id", '"source"', "gemini", "claude", "gpt-", "$", "token")
ISO_TS = re.compile(r"20\d{2}-\d{2}-\d{2}")


def _real_row(task: str) -> dict:
    """One real row per task from data/processed; skip cleanly when the corpus is absent."""
    path = PROCESSED / f"{task}_train.jsonl"
    if not path.exists():
        pytest.skip(f"real corpus not present: {path}")
    with path.open(encoding="utf-8") as fh:
        line = fh.readline()
    if not line.strip():
        pytest.skip(f"real corpus is empty: {path}")
    return json.loads(line)


def _completion(row: dict) -> str:
    return row["completion"][0]["content"]


def _ground_truth_payloads(task: str, row: dict) -> list[str]:
    """The full answer payloads (never to be embedded in a class-level note)."""
    if task == "repair":
        return [str(row["plant_original"])]
    if task == "ledger":
        return [json.dumps(row["ledger"], sort_keys=True)]
    if task == "audit":
        return [str(row["ed_json"])]
    if task == "design":
        return [json.dumps(row["ref_spec"], sort_keys=True)]
    return []


@pytest.mark.corpus
@pytest.mark.parametrize("task", TASKS)
def test_success_note_is_a_short_class_level_lesson_from_the_completion(task):
    row = _real_row(task)
    text = _completion(row)
    verdict = validate.validate(task, text, row)
    assert verdict["accepted"], (task, verdict["reasons"])

    note = validate.success_note(task, row, verdict)

    assert note.accepted is True
    assert note.task == task
    assert note.lesson and note.fix
    assert len(note.lesson) <= 240 and len(note.fix) <= 240

    served = f"{note.lesson} {note.fix}"
    # No full ground-truth answer / stream table / spec JSON is ever embedded.
    for payload in _ground_truth_payloads(task, row):
        assert payload not in served, f"{task}: note embeds the ground-truth answer"
    # No bookkeeping of any kind (D11).
    for marker in BOOKKEEPING:
        assert marker not in served, f"{task}: note leaked {marker!r}"
    assert not ISO_TS.search(served), f"{task}: note leaked a timestamp"

    # fix is derived from the candidate's own accepted completion, not from a ground-truth field.
    collapsed_completion = " ".join(text.split())
    for edit in note.fix.split("; "):
        assert edit in collapsed_completion, f"{task}: fix {edit!r} did not come from the completion"


@pytest.mark.corpus
def test_ingest_writes_accepted_and_rejected_notes(tmp_path):
    src, out, mem = tmp_path / "sft", tmp_path / "frontier", tmp_path / "memory"

    def sft_row(row, rid):
        return dict(row, id=rid, task="repair", source="ground_truth", harness="agents+skills",
                    memory_notes=[], tokens=1234)

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    good = _real_row("repair")
    bad = dict(_real_row("repair"), id="row-b", fault_type="unit_slip")
    write(src / "repair_train.jsonl", [sft_row(good, "row-a"), sft_row(bad, "row-b")])
    transfer.export(src, tmp_path / "export", tasks=("repair",), per_task=10)
    comps = tmp_path / "completions.jsonl"
    write(comps, [{"id": "row-a", "completion": _completion(good)},
                  {"id": "row-b", "completion": "<diagnosis>Candidate repair.</diagnosis>"}])

    report = transfer.ingest(tmp_path / "export", comps, out, mem, tasks=("repair",), model="claude-sonnet")
    notes = memory.load(mem / "frontier_notes.jsonl")

    assert any(n.accepted for n in notes) and any(not n.accepted for n in notes)
    assert report["stats"]["memory_success_notes"] >= 1
    assert report["stats"]["memory_notes"] >= 2


def _note(nid, lesson, *, accepted):
    return memory.Note(id=nid, task="repair", fault_type="wrong_fluid", architecture="",
                       symptoms="", lesson=lesson, fix="FLUID @CC.fuel_in = Air",
                       source="x", accepted=accepted, record_id=nid)


def test_retrieve_returns_a_mix_and_keeps_a_failure_slot():
    query = "wrong_fluid fluid class label fix pattern working accepted"
    # Both accepted notes rank above the failure; without the reserved slot k=2 would be all-OK.
    ok1 = _note("ok1", "wrong_fluid fluid class label fix pattern working accepted", accepted=True)
    ok2 = _note("ok2", "wrong_fluid fluid class label fix pattern accepted", accepted=True)
    fail = _note("fail", "wrong_fluid fluid class label", accepted=False)

    got = memory.retrieve(query, [ok1, ok2, fail], k=2)
    assert len(got) == 2
    assert any(n.accepted for n in got) and any(not n.accepted for n in got), "retrieve must return a mix"
    assert sum(not n.accepted for n in got) >= 1, "at least one failure must survive"


def test_retrieve_caps_hand_written_seed_notes_to_one_slot():
    """Seed method notes are recap, not evidence: at most one per served block, mined notes backfill."""
    query = "repair broken plant stream temperature pressure efficiency exergy validity"
    seed1 = _note("seed1", query, accepted=True)
    seed1 = memory.replace(seed1, source="seed:method", extra={"seed": True})
    seed2 = _note("seed2", query, accepted=True)
    seed2 = memory.replace(seed2, source="seed:method", extra={"seed": True})
    mined_ok = _note("mined-ok", "repair broken plant stream", accepted=True)
    mined_fail = _note("mined-fail", "repair broken plant stream pressure", accepted=False)

    got = memory.retrieve(query, [seed1, seed2, mined_ok, mined_fail], k=3)
    assert len(got) == 3
    assert sum(memory._is_seed(n) for n in got) <= 1
    assert any(not n.accepted for n in got), "a verifier-mined failure must survive"


def test_retrieve_treats_missing_accepted_as_failure():
    # A legacy note object with no `accepted` attribute must count as a failure, not an accepted pattern.
    from types import SimpleNamespace
    legacy = SimpleNamespace(id="legacy", task="repair", fault_type="wrong_fluid", architecture="",
                             symptoms="", lesson="wrong_fluid fluid class label", record_id="legacy")
    modern_ok = _note("ok", "wrong_fluid fluid class label fix pattern working accepted", accepted=True)
    got = memory.retrieve("wrong_fluid fluid class label", [modern_ok, legacy], k=1)
    assert got and not getattr(got[0], "accepted", False)


def test_dedupe_lessons_keeps_successes_and_failures_in_separate_classes():
    ok = _note("ok", "same lesson text", accepted=True)
    fail = _note("fail", "same lesson text", accepted=False)
    kept = memory.dedupe_lessons([ok, fail])
    assert len(kept) == 2 and {n.accepted for n in kept} == {True, False}
    assert len(memory.dedupe([ok, fail])) == 2


def test_format_notes_marks_both_kinds():
    block = memory.format_notes([_note("ok", "accepted lesson", accepted=True),
                                 _note("fail", "rejected lesson", accepted=False)])
    assert "KNOWN FAILURE MODES AND WORKING PATTERNS" in block
    assert "- OK   repair/wrong_fluid:" in block
    assert "- FAIL repair/wrong_fluid:" in block


def test_compact_notes_marks_both_kinds_without_bookkeeping():
    def rich(nid, accepted):
        return memory.Note(
            id=nid, task="repair", fault_type="wrong_fluid", architecture="dc_absorption",
            symptoms="google/gemini-3.1 ts=1 cost=45 tokens=912 2026-01-14",
            lesson="Attempt accepted. Correct the fluid class.",
            fix="FLUID @CC.fuel_in = Air cost=45 tokens=9",
            source="frontier:google-gemini-3.1", accepted=accepted, record_id="rec-123",
            extra={"model": "claude-sonnet", "run_id": "run-1"},
        )

    out = compact_notes([rich("ok", True), rich("fail", False)], k=5)
    assert "OK" in out and "FAIL" in out
    assert "wrong_fluid" in out and "FLUID @CC.fuel_in = Air" in out
    for marker in BOOKKEEPING:
        assert marker not in out, f"compact leaked {marker!r}"
    assert not ISO_TS.search(out), "compact leaked a timestamp"
