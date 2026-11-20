"""Tests for the read-only frontier CoT datagen progress reporter."""
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "report_datagen_progress", REPO / "scripts/report_datagen_progress.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cohort_file(root, model="gpt-6-luna", task="repair"):
    return root / "data/frontier_tools_cohort" / model / f"frontier_{task}_train.jsonl"


def _row(**overrides):
    row = {
        "task": "repair", "tier": "easy", "record_id": "r1",
        "source": "frontier_tools",
        "provenance": {"model": "gpt-6-luna"},
    }
    row.update(overrides)
    return row


def test_tier_is_derived_from_cell_when_tier_absent(tmp_path):
    row = _row(tier=None, cell="repair/medium", record_id="cell-derived")
    _write_rows(_cohort_file(tmp_path), [row])
    payload, errors = mod.build_report(tmp_path)
    assert errors == []
    assert payload["matrix"]["gpt-6-luna"]["repair"]["medium"] == 1
    assert payload["matrix"]["gpt-6-luna"]["repair"]["easy"] == 0
    assert payload["defects"]["unknown_tier"] == 0
    assert payload["per_tier"]["medium"] == 1


def test_duplicate_record_id_and_id_vn_rows_count_once(tmp_path):
    rows = [
        _row(record_id="dup", id="dup#v1"),          # same question, version 1
        _row(record_id="dup#v2", id="dup#v2"),       # same question, version 2
        _row(record_id=None, id="only-id", tier="hard"),  # id-only fallback
        _row(record_id=None, id="only-id#v3", tier="hard"),  # its version
        _row(record_id="other", id="other", tier="medium"),
    ]
    _write_rows(_cohort_file(tmp_path), rows)
    payload, _ = mod.build_report(tmp_path)
    assert payload["total_accepted_traces"] == 3
    assert payload["matrix"]["gpt-6-luna"]["repair"] == {
        "easy": 1, "medium": 1, "hard": 1,
    }
    # The same record_id asked of a second model is a second accepted trace.
    _write_rows(_cohort_file(tmp_path, model="claude-haiku-5-5"),
                [_row(record_id="dup", id="dup#v1")])
    payload, _ = mod.build_report(tmp_path)
    assert payload["total_accepted_traces"] == 4
    assert payload["per_model"]["claude-haiku-5-5"] == 1


def test_unknown_tier_is_a_defect_and_still_counted(tmp_path):
    _write_rows(_cohort_file(tmp_path), [
        _row(tier=None, cell=None, record_id="no-tier"),
        _row(tier="trivial", record_id="odd-tier"),
    ])
    payload, _ = mod.build_report(tmp_path)
    assert payload["defects"]["unknown_tier"] == 2
    assert payload["per_tier"]["unknown"] == 2
    assert payload["per_model"]["gpt-6-luna"] == 2   # not silently dropped
    assert payload["per_task"]["repair"] == 2
    assert payload["total_accepted_traces"] == 2
    assert payload["matrix"]["gpt-6-luna"]["repair"] == {
        "easy": 0, "medium": 0, "hard": 0,
    }


def test_matrix_lists_all_24_zero_cells_for_empty_cohort(tmp_path):
    payload, errors = mod.build_report(tmp_path)
    assert errors == []
    assert list(payload["matrix"]) == list(mod.MODELS)
    cells = [(m, t, tier) for m in mod.MODELS for t in mod.TASKS for tier in mod.TIERS]
    assert len(cells) == 24
    assert sum(len(payload["matrix"][m][t]) for m in mod.MODELS for t in mod.TASKS) == 24
    assert payload["zero_cells"] == [f"{m}/{t}/{tier}" for m, t, tier in cells]
    assert payload["total_accepted_traces"] == 0


def test_non_frontier_source_and_missing_provenance_are_counted(tmp_path):
    _write_rows(_cohort_file(tmp_path), [
        _row(record_id="clean"),
        _row(record_id="ingest-stamped", source="frontier:gpt-6-luna"),
        _row(record_id="bad-source", source="ground_truth"),
        _row(record_id="no-provenance", provenance={}),
        _row(record_id="no-provenance-key", provenance=None),
    ])
    payload, _ = mod.build_report(tmp_path)
    # `frontier:<model>` is how ingest stamps rows; only non-frontier is a defect.
    assert payload["defects"]["non_frontier_tools_source"] == 1
    assert payload["defects"]["missing_provenance_model"] == 2
    assert payload["defects"]["unknown_tier"] == 0


def test_json_payload_is_deterministic_and_has_no_timestamp(tmp_path):
    _write_rows(_cohort_file(tmp_path), [_row()])
    first, _ = mod.build_report(tmp_path)
    second, _ = mod.build_report(tmp_path)
    blob = json.dumps(first, indent=2, sort_keys=True, ensure_ascii=False)
    assert blob == json.dumps(second, indent=2, sort_keys=True, ensure_ascii=False)
    assert "2026" not in blob
    assert "generated" not in blob
    assert "balance" in mod.format_markdown(first, "2026-11-20T00:00:00+00:00")


def test_unreadable_cohort_file_exits_two(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    bad = _cohort_file(tmp_path)
    bad.mkdir(parents=True, exist_ok=True)  # a directory named *.jsonl cannot be opened
    assert mod.main(["--json", str(tmp_path / "none.json"),
                     "--md", str(tmp_path / "none.md")]) == 2
    assert "cannot read cohort file" in capsys.readouterr().err
