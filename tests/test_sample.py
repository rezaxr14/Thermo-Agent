"""Prompt construction for the teacher must match what the student will see, and the client must normalise URLs."""
import json

from thermo_agent.teacher import memory, sample

from _rows import make_row


def test_teacher_normalizes_the_base_url():
    t = sample.Teacher("abc.trycloudflare.com", "teacher")
    assert t.base_url == "https://abc.trycloudflare.com/v1"
    t2 = sample.Teacher("https://abc.trycloudflare.com/v1/", "teacher")
    assert t2.base_url == "https://abc.trycloudflare.com/v1"


def test_build_messages_injects_harness_and_memory():
    row = make_row("wrong_fluid")
    note = memory.Note(id="n1", task="repair", fault_type="wrong_fluid", architecture="dc_chp",
                       symptoms="verifier: invalid plant", lesson="copying the broken label back fails",
                       fix="FLUID @chw_ret = Water", source="adversarial", accepted=False, record_id="other")
    msgs = sample.build_messages("repair", row, "agents+skills", [note], k_memory=1)
    system = msgs[0]["content"]
    assert "AGENTS" in system.upper() or "SKILL" in system.upper()
    assert "KNOWN FAILURE MODES" in system
    assert msgs[1]["content"] == row["prompt"][1]["content"], "the user turn must be untouched"


def test_build_messages_without_harness_keeps_the_original_prompt():
    row = make_row("unit_slip")
    msgs = sample.build_messages("repair", row, "none", [], k_memory=0)
    assert msgs == row["prompt"]


def test_load_rows_merges_extra_directories(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for d, rid in ((a, "x"), (b, "y")):
        (d / "repair_train.jsonl").write_text(json.dumps({"id": rid}) + "\n", encoding="utf-8")
    rows = sample.load_rows(a, "repair", extra=[b])
    assert [r["id"] for r in rows] == ["x", "y"]
    assert sample.load_rows(a, "repair", limit=1, extra=[b]) == [{"id": "x"}]
