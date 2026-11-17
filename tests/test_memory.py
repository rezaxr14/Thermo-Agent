"""Retrieval, formatting and the no-leak rule of the hindsight memory."""
from thermo_agent.teacher import memory


def note(i, fault="wrong_fluid", arch="turbine-absorption-dry", lesson="copying the broken label back fails",
         record="r1", task="repair", fix="FLUID @CC.fuel_in = Air"):
    return memory.Note(id=f"n{i}", task=task, fault_type=fault, architecture=arch, symptoms="",
                       lesson=lesson, fix=fix, source="adversarial", accepted=False, record_id=record)


def test_retrieve_prefers_the_matching_fault_class():
    notes = [note(0, fault="wrong_fluid"), note(1, fault="unit_slip", lesson="the power of ten is wrong"),
             note(2, fault="sign_flip", lesson="the sign is flipped")]
    got = memory.retrieve("wrong_fluid CC.fuel_in Water Air", notes, k=1)
    assert got and got[0].fault_type == "wrong_fluid"


def test_retrieve_excludes_the_same_record():
    notes = [note(0, record="r1"), note(1, record="r2")]
    got = memory.retrieve("wrong_fluid", notes, k=5, exclude_record="r1")
    assert [n.record_id for n in got] == ["r2"]


def test_retrieve_filters_by_task():
    notes = [note(0, task="repair"), note(1, task="ledger")]
    got = memory.retrieve("wrong_fluid", notes, k=5, task="ledger")
    assert [n.task for n in got] == ["ledger"]


def test_format_notes_is_actionable_and_includes_the_pattern():
    block = memory.format_notes([note(0)])
    assert "KNOWN FAILURE MODES" in block
    assert "wrong_fluid" in block and "FLUID @CC.fuel_in = Air" in block


def test_augment_with_memory_appends_to_the_system_message_only():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    out = memory.augment_with_memory(msgs, "repair", "wrong_fluid", [note(0)], k=1)
    assert out[1] == msgs[1]
    assert out[0]["content"].startswith("sys")
    assert "KNOWN FAILURE MODES" in out[0]["content"]
    assert msgs[0]["content"] == "sys", "the input must not be modified"


def test_augment_is_a_noop_without_notes():
    msgs = [{"role": "system", "content": "sys"}]
    assert memory.augment_with_memory(msgs, "repair", "x", [], k=3) is msgs


def test_save_load_and_dedupe(tmp_path):
    notes = [note(0), note(1, lesson="a different lesson"), note(0)]   # 0 and 2 are the same lesson
    path = tmp_path / "m.jsonl"
    memory.save(notes, path)
    back = memory.load(path)
    assert len(back) == 3
    assert len(memory.dedupe(back)) == 2
    assert memory.load(tmp_path / "missing.jsonl") == []


def test_failure_reason_summary_and_frequency_notes():
    notes = [note(0, record="r1"), note(1, record="r2"), note(2, lesson="a rare mistake")]
    summary = memory.summarize(notes)
    assert summary["n"] == 3
    assert summary["top_lessons"][0]["count"] == 2
    assert summary["by_task"]["repair"] == 3
    frequent = memory.frequency_notes(notes)
    assert len(frequent) == 1 and "Seen 2 times" in frequent[0].lesson
