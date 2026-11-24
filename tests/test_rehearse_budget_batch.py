"""Local-only tests for the zero-cost budget-batch rehearsal.

No network, no Kaggle, no model calls.  Only the end-to-end test is marked
``corpus`` because it reads the real untracked shards/export rows.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rehearsal():
    return _load_module("rehearse_budget_batch_under_test",
                        "scripts/rehearse_budget_batch.py")


@pytest.fixture(scope="module")
def runner(rehearsal):
    return rehearsal.load_runner_module()


# ---------------------------------------------------------------------------
# Journal builder
# ---------------------------------------------------------------------------
def test_build_journal_records_shape_and_unique_ids(rehearsal, runner):
    cfg = runner.Config.from_dict({"mode": "harness", "tiers": ["easy", "medium", "hard"]})
    tools = ["plant_inspect", "verify_candidate"]
    rows = [{"id": f"row-{i}", "record_id": f"row-{i}", "task": "repair",
             "tier": ["easy", "medium", "hard"][i % 3], "topology": "dc_engine_electric",
             "prompt": [{"role": "system", "content": "You repair plants."}]}
            for i in range(9)]
    export_rows = {f"row-{i}": {"completion": [{"role": "assistant", "content": f"<diagnosis>fix {i}</diagnosis>"}]}
                   for i in range(9)}

    records, from_export = rehearsal.build_journal(
        rows, model="gpt-6-luna", run_id="rehearsal-test", shard="fake.jsonl",
        cfg=cfg, tools=tools, export_rows=export_rows,
    )

    assert len(records) == 9
    assert from_export == 9
    assert len({r["id"] for r in records}) == 9
    for record in records:
        answer = record["completion"]
        assert record["error"] is None and record["graded"] is False
        assert record["model"] == "gpt-6-luna" and record["mode"] == "harness"
        assert record["sft_completion"].rstrip().endswith(answer)
        assert "<tool_call" in record["sft_completion"]
        assert "<function=" in record["sft_completion"]
        assert "<tool_response" in record["sft_completion"]
        assert "<tool_call name=" not in record["sft_completion"]
        assert "<tool_result" not in record["sft_completion"]
        assert record["reasoning"].strip()
        traj = record["trajectory"]
        assert len(traj["turns"]) == 3
        assert traj["final"] == answer
        assert traj["tools_available"] == tools
        assert traj["notes_used"] == []
        assert record["provenance"]["model"] == "gpt-6-luna"
        assert record["provenance"]["run_id"] == "rehearsal-test"
        for turn in traj["turns"]:
            assert turn["assistant"].strip()
            assert turn["reasoning"].strip()
            if turn["tool_calls"]:
                assert turn["tool_results"]


def test_synthetic_answer_is_valid_xml_per_task(rehearsal):
    assert "<diagnosis>" in rehearsal.synthetic_answer("repair")
    assert "<edits>" in rehearsal.synthetic_answer("repair")
    assert "<rationale>" in rehearsal.synthetic_answer("design")
    assert "<design>" in rehearsal.synthetic_answer("design")
    audit = rehearsal.synthetic_answer("audit")
    for tag in ("analysis", "ranking", "change", "prediction"):
        assert f"<{tag}>" in audit and f"</{tag}>" in audit
    assert "<diagnosis>" in rehearsal.synthetic_answer("ledger")
    assert "<fix>" in rehearsal.synthetic_answer("ledger")


# ---------------------------------------------------------------------------
# The two process_budget_batch.ps1 guards
# ---------------------------------------------------------------------------
def test_empty_journal_guard(rehearsal, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(rehearsal.JournalGuardError) as excinfo:
        rehearsal.validate_journal(empty)
    assert "journal is empty:" in str(excinfo.value)
    assert "attempted 0 rows" in str(excinfo.value)


def test_error_row_guard(rehearsal, tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps({"id": "__error__", "error": "rows=0 shard=x"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(rehearsal.JournalGuardError) as excinfo:
        rehearsal.validate_journal(journal)
    assert "runner journalled an error instead of traces" in str(excinfo.value)


def test_all_rows_errored_journal_is_refused(rehearsal, tmp_path):
    """A non-empty journal whose rows all failed must not look like a batch."""
    journal = tmp_path / "all_errors.jsonl"
    journal.write_text("".join(
        json.dumps({"id": f"row-{i}", "error": "KeyError: 'ed_json'",
                    "completion": "", "sft_completion": ""}) + "\n"
        for i in range(3)), encoding="utf-8")
    with pytest.raises(rehearsal.JournalGuardError) as excinfo:
        rehearsal.validate_journal(journal)
    assert "0 usable completions" in str(excinfo.value)
    assert "ed_json" in str(excinfo.value)


def test_healthy_journal_passes_guard(rehearsal, tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        json.dumps({"id": "row-1", "completion": "ok"}) + "\n"
        + json.dumps({"id": "row-2", "completion": "ok"}) + "\n",
        encoding="utf-8",
    )
    assert rehearsal.validate_journal(journal) == 2


def test_reject_path_self_checks(rehearsal, tmp_path):
    guards = rehearsal.check_reject_paths(tmp_path)
    assert "journal is empty:" in guards["empty"]
    assert "runner journalled an error instead of traces" in guards["error"]
    assert "0 usable completions" in guards["no_completions"]


# ---------------------------------------------------------------------------
# select_rows interaction (partial shard must not select 0)
# ---------------------------------------------------------------------------
def test_select_rows_two_tier_fixture_is_not_empty(runner):
    cfg = runner.Config.from_dict({"tiers": ["easy", "medium", "hard"],
                                   "equal_tiers": True})
    rows = [{"id": f"e{i}", "tier": "easy"} for i in range(3)] + \
           [{"id": f"m{i}", "cell": "repair/medium"} for i in range(3)]
    selected = cfg.select_rows(rows)
    assert selected, "a 2-tier shard must select rows, never 0"
    assert {cfg.tier_of(r) for r in selected} == {"easy", "medium"}


# ---------------------------------------------------------------------------
# End-to-end on the real shards (no Kaggle / model; gate + ingest are local)
# ---------------------------------------------------------------------------
@pytest.mark.corpus
def test_end_to_end_rehearsal(rehearsal, tmp_path, capsys):
    out = tmp_path / "rehearsal"
    code = rehearsal.main(["--rows", "4", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "REHEARSAL OK" in printed
    assert tmp_path.exists()  # the caller-provided temp dir still exists
