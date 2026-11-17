"""`synth.verify` must notice a corpus that no longer matches the report that produced it.

The failure this guards against is real: one build left `data/sft/repair_train.jsonl` inside a nested `sft.new`
directory, so the report still claimed 129,684 rows while the train split had lost its largest file. The token
audit, the report and `combine` all stayed silent because none of them re-reads the directory after the swap.
"""
import json

from thermo_agent.synth import combine, verify

from _rows import make_row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_rows(path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build(tmp_path, variants=1, harness_fraction=1.0):
    proc, front, mem, out = (tmp_path / x for x in ("processed", "frontier", "memory", "sft"))
    write_rows(proc / "repair_train.jsonl", [make_row("wrong_fluid")])
    report = combine.build(proc, front, mem, out, tasks=("repair",), variants=variants,
                           harness_fraction=harness_fraction)
    return out, report


def test_a_complete_corpus_verifies(tmp_path):
    out, report = build(tmp_path, variants=2)
    r = verify.verify(out)
    assert r["ok"] and r["problems"] == []
    assert r["rows"] == report["splits"]["train"]
    assert r["files"]["repair_train"]["n"] == 2
    assert r["files"]["repair_train"]["harness_share"] == 1.0


def test_a_missing_split_file_is_reported(tmp_path):
    out, _ = build(tmp_path)
    (out / "repair_val.jsonl").unlink()
    r = verify.verify(out)
    assert not r["ok"]
    assert any("missing file: repair_val.jsonl" in p for p in r["problems"])


def test_a_row_count_mismatch_is_reported(tmp_path):
    out, report = build(tmp_path, variants=2)
    rows = read_rows(out / "repair_train.jsonl")
    # Keep the file well formed but drop one paraphrased variant: the count no longer matches the report.
    write_rows(out / "repair_train.jsonl", rows[:1])
    r = verify.verify(out)
    assert not r["ok"]
    assert any("1 rows on disk, report says" in p for p in r["problems"])


def test_a_duplicate_id_across_splits_is_a_leak(tmp_path):
    out, _ = build(tmp_path)
    row = dict(read_rows(out / "repair_train.jsonl")[0], split="val")
    write_rows(out / "repair_val.jsonl", [row])
    r = verify.verify(out)
    assert any(p.startswith("leak:") for p in r["problems"])


def test_a_malformed_prompt_and_a_missing_completion_are_reported(tmp_path):
    out, _ = build(tmp_path)
    rows = read_rows(out / "repair_train.jsonl")
    rows[0]["prompt"] = []
    rows[0]["completion"] = None
    write_rows(out / "repair_train.jsonl", rows)
    r = verify.verify(out)
    problems = " ".join(r["problems"])
    assert "malformed prompt" in problems and "have no completion" in problems
    # GRPO prompts may legitimately be stored without the reference completion.
    ok = verify.verify(out, allow_missing_completion=True)
    assert not any("have no completion" in p for p in ok["problems"])
    assert any("malformed prompt" in p for p in ok["problems"])


def test_a_file_the_report_does_not_know_about_is_reported(tmp_path):
    out, _ = build(tmp_path)
    write_rows(out / "repair_extra.jsonl", read_rows(out / "repair_train.jsonl"))
    r = verify.verify(out)
    assert any("unknown split file: repair_extra.jsonl" in p for p in r["problems"])


def test_a_row_over_the_token_gate_is_reported(tmp_path):
    out, _ = build(tmp_path)
    r = verify.verify(out, max_tokens=10)
    assert not r["ok"]
    assert any("tokens (> 10)" in p for p in r["problems"])
