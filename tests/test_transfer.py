"""The teacher export must not leak answers, and ingest must keep only what the tool suite accepts."""
import json

from thermo_agent.teacher import memory, transfer

from _rows import make_row


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def read(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def sft_row(row, rid, source="ground_truth", harness="agents+skills"):
    return dict(row, id=rid, task="repair", source=source, harness=harness, memory_notes=[], tokens=1234)


def test_export_keeps_one_prompt_per_plant_and_never_the_answer(tmp_path):
    src, out = tmp_path / "sft", tmp_path / "export"
    a = make_row("wrong_fluid")
    b = dict(make_row("unit_slip"), id="row-b")
    write(src / "repair_train.jsonl", [sft_row(a, "row-a#v0"), sft_row(a, "row-a#v3"),
                                       sft_row(b, "row-b"), sft_row(b, "row-b-t-x", source="frontier")])
    manifest = transfer.export(src, out, tasks=("repair",), per_task=10)
    rows = read(out / "repair_train.jsonl")
    assert len(rows) == 2, "one prompt per ground-truth plant, frontier rows are not re-exported"
    assert all("completion" not in r for r in rows), "the answer must not travel to the teacher"
    assert all(r["prompt"] for r in rows)
    assert manifest["tasks"]["repair"]["n"] == 2 and manifest["rows"] == 2
    assert manifest["tasks"]["repair"]["max_prompt_tokens"] == 1234


def test_export_respects_the_per_task_cap(tmp_path):
    src, out = tmp_path / "sft", tmp_path / "export"
    rows = [sft_row(dict(make_row("wrong_fluid"), id=f"r{i}"), f"r{i}") for i in range(6)]
    write(src / "repair_train.jsonl", rows)
    transfer.export(src, out, tasks=("repair",), per_task=3)
    assert len(read(out / "repair_train.jsonl")) == 3


def test_ingest_keeps_accepted_answers_and_turns_rejections_into_memory(tmp_path):
    src, out, mem = tmp_path / "sft", tmp_path / "frontier", tmp_path / "memory"
    row = make_row("wrong_fluid")
    other = dict(make_row("unit_slip"), id="row-b")
    write(src / "repair_train.jsonl", [sft_row(row, "row-a"), sft_row(other, "row-b")])
    transfer.export(src, tmp_path / "export", tasks=("repair",), per_task=10)
    good = row["completion"][0]["content"]
    bad = "<diagnosis>Candidate repair.</diagnosis>"
    comps = tmp_path / "completions.jsonl"
    write(comps, [{"id": "row-a", "completion": good}, {"id": "row-b", "completion": bad}])

    report = transfer.ingest(tmp_path / "export", comps, out, mem, tasks=("repair",), model="claude-sonnet")
    kept = read(out / "frontier_repair_train.jsonl")
    assert len(kept) == 1 and kept[0]["source"] == "frontier:claude-sonnet" and kept[0]["task"] == "repair"
    assert kept[0]["validation"]["reasons"]
    assert "plant_broken" in kept[0], "label columns must survive for the GRPO reward functions"
    assert report["accepted"] == 1 and report["attempts"] == 2
    assert report["stats"]["rejected:repair"] == 1
    notes = memory.load(mem / "frontier_notes.jsonl")
    assert len(notes) == 2, "both the accepted and the rejected attempt land in memory"
    assert all(n.source.startswith("frontier:") for n in notes)
    assert any(n.accepted for n in notes) and any(not n.accepted for n in notes)
    assert report["stats"]["memory_success_notes"] >= 1 and report["stats"]["memory_notes"] >= 2


def test_ingest_is_resumable(tmp_path):
    src, out = tmp_path / "sft", tmp_path / "frontier"
    row = make_row("wrong_fluid")
    write(src / "repair_train.jsonl", [sft_row(row, "row-a")])
    transfer.export(src, tmp_path / "export", tasks=("repair",), per_task=10)
    comps = tmp_path / "completions.jsonl"
    write(comps, [{"id": "row-a", "completion": row["completion"][0]["content"]}])
    transfer.ingest(tmp_path / "export", comps, out, None, tasks=("repair",))
    report = transfer.ingest(tmp_path / "export", comps, out, None, tasks=("repair",))
    assert len(read(out / "frontier_repair_train.jsonl")) == 1, "a second ingest must not duplicate the row"
    assert report["stats"]["skipped_already_accepted"] == 1


def test_load_exports_resolves_ids_from_extra_dirs_and_record_id(tmp_path):
    """A trace shard is built from several corpora; an id absent from the primary dir is not "unknown"."""
    primary, arch, prov = tmp_path / "processed", tmp_path / "processed_arch", tmp_path / "prov"
    row = make_row("wrong_fluid")
    write(primary / "repair_train.jsonl", [sft_row(row, "rep-1-00001")])
    write(arch / "repair_train.jsonl", [dict(row, id="arch-9-00000", task="repair")])
    # An already-ingested frontier row carries `<record>#v3-t-<model>` and its own record_id.
    write(prov / "frontier_repair_train.jsonl",
          [dict(row, id="rep-129-00072#v3-t-google-gemini", record_id="rep-129-00072#v3", task="repair")])

    exports = transfer.load_exports(primary, tasks=("repair",), extra_dirs=(arch, prov))
    assert set(exports) == {"rep-1-00001", "arch-9-00000", "rep-129-00072#v3"}
    assert exports["rep-1-00001"]["task"] == "repair", "the file name tags the task"
    assert exports["rep-129-00072#v3"]["id"].endswith("-t-google-gemini")


def test_load_exports_primary_dir_wins_over_extra_dirs(tmp_path):
    primary, other = tmp_path / "processed", tmp_path / "other"
    row = make_row("wrong_fluid")
    write(primary / "repair_train.jsonl", [dict(row, id="same", task="repair", topology="primary")])
    write(other / "repair_train.jsonl", [dict(row, id="same", task="repair", topology="other")])
    exports = transfer.load_exports(primary, tasks=("repair",), extra_dirs=(other,))
    assert exports["same"]["topology"] == "primary"
