"""Combine must emit varied, harness-aware, memory-aware rows without leaking across splits, and the output
filenames must be exactly what the SFT and GRPO trainers read."""
import json

from thermo_agent.synth import combine
from thermo_agent.teacher import memory, validate

from _rows import make_row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_rows(path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def note(record="other-row"):
    return memory.Note(id="n1", task="repair", fault_type="wrong_fluid", architecture="dc_chp",
                       symptoms="verifier: second_law", lesson="copying the broken label back fails",
                       fix="FLUID @CC.fuel_in = Air", source="adversarial", accepted=False, record_id=record)


def test_combine_emits_paraphrases_harness_and_memory(tmp_path):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    write_rows(proc / "repair_train.jsonl", [make_row("wrong_fluid")])
    memory.save([note()], mem / "notes.jsonl")
    report = combine.build(proc, front, mem, out, tasks=("repair",), variants=3, harness_fraction=1.0)
    rows = read_rows(out / "repair_train.jsonl")
    assert len(rows) == 3
    assert len({r["id"] for r in rows}) == 3
    assert len({r["completion"][0]["content"] for r in rows}) == 3, "variants must differ"
    assert all(r["harness"] == "agents+skills" for r in rows)
    assert all("KNOWN FAILURE MODES" in r["prompt"][0]["content"] for r in rows)
    assert all(r["source"] == "ground_truth" for r in rows)
    assert report["problems"] == []
    assert (out / "sft_report.json").exists()


def test_memory_is_still_added_without_the_harness(tmp_path):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    write_rows(proc / "repair_train.jsonl", [make_row("unit_slip")])
    memory.save([note()], mem / "notes.jsonl")
    combine.build(proc, front, mem, out, tasks=("repair",), variants=1, harness_fraction=0.0)
    row = read_rows(out / "repair_train.jsonl")[0]
    assert row["harness"] == "none"
    assert "KNOWN FAILURE MODES" in row["prompt"][0]["content"]


def test_frontier_rows_are_merged_and_marked(tmp_path):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    row = make_row("wrong_fluid")
    write_rows(proc / "repair_train.jsonl", [row])
    verdict = validate.validate("repair", row["completion"][0]["content"], row)
    write_rows(front / "frontier_repair_train.jsonl",
               [validate.teacher_row("repair", row["completion"][0]["content"], row, verdict, tag="s0")])
    report = combine.build(proc, front, mem, out, tasks=("repair",), variants=1, harness_fraction=0.0)
    rows = read_rows(out / "repair_train.jsonl")
    assert {r["source"] for r in rows} == {"ground_truth", "frontier"}
    assert report["tasks"]["repair"]["train"]["frontier"] == 1


def test_prompts_over_the_cap_are_dropped_not_truncated(tmp_path):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    write_rows(proc / "repair_train.jsonl", [make_row("wrong_fluid")])
    report = combine.build(proc, front, mem, out, tasks=("repair",), variants=2, max_chars=10)
    assert read_rows(out / "repair_train.jsonl") == []
    assert report["tasks"]["repair"]["train"]["n"] == 0
    assert report["problems"] == []


def test_splits_are_written_separately(tmp_path):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    a, b = make_row("wrong_fluid"), make_row("unit_slip")
    b = dict(b, id="row-other", split="val")
    write_rows(proc / "repair_train.jsonl", [a])
    write_rows(proc / "repair_val.jsonl", [b])
    report = combine.build(proc, front, mem, out, tasks=("repair",), variants=1)
    assert report["splits"]["train"] == 1 and report["splits"]["val"] == 1
    assert read_rows(out / "repair_train.jsonl")[0]["split"] == "train"
    assert read_rows(out / "repair_val.jsonl")[0]["split"] == "val"


def test_design_uses_its_own_ood_split_names(tmp_path):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    row = make_row("wrong_fluid")
    design_row = {"id": "d-1", "split": "test_ood_climate", "task": "design", "prompt": row["prompt"],
                  "completion": [{"role": "assistant", "content": "<design/>"}], "site": row["site"],
                  "ref_fuel_w": 1.0, "topology": "dc_chp"}
    write_rows(proc / "design_test_ood_climate.jsonl", [design_row])
    report = combine.build(proc, front, mem, out, tasks=("design",), variants=1, harness_fraction=0.0)
    assert report["tasks"]["design"]["test_ood_climate"]["n"] == 1
    assert (out / "design_test_ood_climate.jsonl").exists()
