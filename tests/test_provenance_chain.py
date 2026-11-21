"""Unit tests for the eight provenance-chain checks plus one corpus smoke test.

Every check is exercised on tiny synthetic fixtures written into ``tmp_path``:
one passing and one failing case each.  Nothing here touches the network, a
model, or the live corpora (except the explicitly ``@pytest.mark.corpus`` test).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from scripts.audit_provenance_chain import (
    CHECK_NAMES,
    check_harness_prompt_leakage,
    check_memory,
    check_mix_leakage,
    check_mix_propagation,
    check_provenance_fields,
    check_prompt_shards,
    check_source_mapping,
    check_tiers,
    run_audit,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# tiny fixture builders
# --------------------------------------------------------------------------- #
def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _plant(n_components: int = 2) -> str:
    return json.dumps({
        "components": [
            {"id": f"C{i}", "type": "compressor", "params": {}}
            for i in range(n_components)
        ],
    })


def _cohort_row(**overrides) -> dict:
    row = {
        "id": "rep-1-00001-t-gpt-6-luna",
        "record_id": "rep-1-00001",
        "task": "repair",
        "split": "train",
        "source": "frontier:gpt-6-luna",
        "source_detail": "frontier_tools",
        "model": "gpt-6-luna",
        "provider": "openai",
        "run_id": "unit-test-run",
        "prompt": [{"role": "system", "content": "You repair plants. Cite only solver numbers."}],
        "completion": [{"role": "assistant", "content": "<tool_call name=\"inspect_plant\">{}</tool_call>"}],
        "plant_broken": _plant(),
        "provenance": {
            "source_family": "ai_generated",
            "generator": "frontier_teacher",
            "model": "gpt-6-luna",
            "provider": "openai",
            "run_id": "unit-test-run",
        },
    }
    row.update(overrides)
    return row


def _make_cohort(tmp_path: pathlib.Path, rows: list[dict] | None = None) -> pathlib.Path:
    directory = tmp_path / "cohort"
    _write_jsonl(directory / "gpt-6-luna" / "frontier_repair_train.jsonl",
                 rows if rows is not None else [_cohort_row()])
    return directory


def _shard_row(**overrides) -> dict:
    row = {
        "id": "rep-1-00001",
        "task": "repair",
        "tier": "easy",
        "prompt": [{"role": "system", "content": "You repair plants."}],
        "plant": _plant(),
    }
    row.update(overrides)
    return row


def _note(accepted: bool, suffix: str = "a") -> dict:
    return {
        "id": f"repair-{suffix}",
        "task": "repair",
        "fault_type": "unit_slip",
        "architecture": "dc_electric",
        "symptoms": "check V6 failed",
        "lesson": "Use MW, not W.",
        "fix": "SET CH.P_out = 1.0",
        "source": "frontier:gpt-6-luna",
        "accepted": accepted,
        "record_id": f"rep-1-0000{suffix}",
        "extra": {},
    }


def _mix_rows() -> tuple[dict, dict]:
    frontier = {
        "id": "rep-1-00001-t-gpt-6-luna",
        "record_id": "rep-1-00001",
        "task": "repair",
        "source": "frontier_tools",
        "source_model": "gpt-6-luna",
        "prompt": [{"role": "system", "content": "You repair plants."}],
        "provenance": {"source_family": "ai_generated", "model": "gpt-6-luna"},
        "tier": "easy",
    }
    deterministic = {
        "id": "rep-1-00002-dt-direct",
        "record_id": "rep-1-00002",
        "task": "repair",
        "source": "deterministic_tools",
        "prompt": [{"role": "system", "content": "You repair plants."}],
        "provenance": {"source_family": "deterministic", "generator": "deterministic_tools"},
        "tier": "medium",
    }
    return frontier, deterministic


def _make_mix(tmp_path: pathlib.Path, frontier_row: dict | None = None):
    sft = tmp_path / "sft_mix"
    grpo = tmp_path / "grpo_mix"
    frontier, deterministic = _mix_rows()
    rows = [frontier_row or frontier, deterministic]
    _write_jsonl(sft / "repair_train.jsonl", rows)
    _write_jsonl(grpo / "repair_train.jsonl", rows)
    manifest = {
        "generator": "mix_tool_corpora.py",
        "totals": {"sft": len(rows), "grpo": len(rows)},
        "counts": {
            "sft": {"repair": {"deterministic": 1, "frontier": 1, "total": len(rows)}},
            "grpo": {"repair": {"deterministic": 1, "frontier": 1, "total": len(rows)}},
        },
    }
    (sft / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sft, grpo


# --------------------------------------------------------------------------- #
# 1. provenance fields
# --------------------------------------------------------------------------- #
def test_provenance_fields_pass(tmp_path):
    result = check_provenance_fields(_make_cohort(tmp_path))
    assert result.ok, result.problems
    assert result.name == CHECK_NAMES[0]


def test_provenance_fields_missing_model_fails(tmp_path):
    row = _cohort_row()
    del row["model"]
    del row["provenance"]["model"]
    result = check_provenance_fields(_make_cohort(tmp_path, [row]))
    assert not result.ok
    assert any("missing model" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 2. source mapping
# --------------------------------------------------------------------------- #
def test_source_mapping_pass(tmp_path):
    result = check_source_mapping(_make_cohort(tmp_path))
    assert result.ok, result.problems


def test_source_mapping_wrong_model_suffix_fails(tmp_path):
    row = _cohort_row(source="frontier:some-other-model")
    result = check_source_mapping(_make_cohort(tmp_path, [row]))
    assert not result.ok
    assert any("does not name exact model" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 3. tiers are recomputed
# --------------------------------------------------------------------------- #
def test_tiers_recomputed_pass(tmp_path):
    # A single derivable row with no stored tier is fine (missing tier is not a failure).
    result = check_tiers(_make_cohort(tmp_path))
    assert result.ok, result.problems
    # ... and a correct stored tier is fine too.
    result = check_tiers(_make_cohort(tmp_path, [_cohort_row(tier="hard")]))
    assert result.ok, result.problems


def test_tiers_wrong_stored_tier_fails(tmp_path):
    result = check_tiers(_make_cohort(tmp_path, [_cohort_row(tier="easy")]))
    assert not result.ok
    assert any("stored tier" in p and "recomputed" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 4. prompt shards
# --------------------------------------------------------------------------- #
def test_prompt_shards_pass(tmp_path):
    shards = tmp_path / "prompts"
    _write_jsonl(shards / "prompts_repair_easy.jsonl", [_shard_row()])
    result = check_prompt_shards(shards)
    assert result.ok, result.problems


def test_prompt_shards_ref_spec_fails(tmp_path):
    shards = tmp_path / "prompts"
    _write_jsonl(shards / "prompts_repair_easy.jsonl",
                 [_shard_row(ref_spec={"components": []})])
    result = check_prompt_shards(shards)
    assert not result.ok
    assert any("ref_spec" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 5. harness prompt leakage
# --------------------------------------------------------------------------- #
def test_harness_prompt_pass(tmp_path):
    result = check_harness_prompt_leakage(
        _make_cohort(tmp_path), sample=5, rebuild=False)
    assert result.ok, result.problems


def test_harness_prompt_model_id_fails(tmp_path):
    row = _cohort_row(prompt=[
        {"role": "system", "content": "Produced by gpt-6-luna, run unit-test-run."},
    ])
    result = check_harness_prompt_leakage(
        _make_cohort(tmp_path, [row]), sample=5, rebuild=False)
    assert not result.ok
    assert any("harness prompt leaks" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 6. memory outcomes
# --------------------------------------------------------------------------- #
def test_memory_both_outcomes_pass(tmp_path):
    directory = tmp_path / "memory"
    _write_jsonl(directory / "frontier_notes.jsonl",
                 [_note(True, "ok"), _note(False, "bad")])
    result = check_memory(directory)
    assert result.ok, result.problems


def test_memory_only_failures_fails(tmp_path):
    directory = tmp_path / "memory"
    _write_jsonl(directory / "frontier_notes.jsonl",
                 [_note(False, "a"), _note(False, "b")])
    result = check_memory(directory)
    assert not result.ok
    assert any("no accepted memory notes" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 7. mix propagation
# --------------------------------------------------------------------------- #
def test_mix_propagation_pass(tmp_path):
    cohort = _make_cohort(tmp_path)
    sft, grpo = _make_mix(tmp_path)
    result = check_mix_propagation(sft, grpo, cohort)
    assert result.ok, result.problems


def test_mix_row_without_cohort_match_fails(tmp_path):
    cohort = _make_cohort(tmp_path)
    orphan = {
        "id": "rep-999-99999-t-gpt-6-luna",
        "record_id": "rep-999-99999",
        "task": "repair",
        "source": "frontier_tools",
        "source_model": "gpt-6-luna",
        "prompt": [{"role": "system", "content": "You repair plants."}],
        "provenance": {"source_family": "ai_generated", "model": "gpt-6-luna"},
        "tier": "easy",
    }
    sft, grpo = _make_mix(tmp_path, frontier_row=orphan)
    result = check_mix_propagation(sft, grpo, cohort)
    assert not result.ok
    assert any("no cohort match" in p for p in result.problems)


# --- legacy carry-forward (superseded early campaign, e.g. gemini-3.8-flash) --
def _legacy_trace_row(record_id: str, model: str = "google-gemini-3-8-flash") -> dict:
    """Shape of a row in data/frontier_tools_clean: id == the mix record_id."""
    return {
        "id": record_id,
        "record_id": record_id.split("-t-", 1)[0],
        "task": "repair",
        "source": f"frontier:{model}",
        "model": model,
    }


def _make_legacy_store(tmp_path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    directory = tmp_path / "legacy" / "frontier_tools_clean"
    _write_jsonl(directory / "frontier_repair_train.jsonl", rows)
    return directory


def _legacy_mix_row(**overrides) -> dict:
    row = {
        "id": "aud-0-02280-t-google-gemini-3-8-flash-ft-tool_cot-0d20",
        "record_id": "aud-0-02280-t-google-gemini-3-8-flash",
        "task": "repair",
        "source": "frontier_tools",
        "source_model": "google-gemini-3-8-flash",
        "prompt": [{"role": "system", "content": "You repair plants."}],
        "provenance": {
            "source_family": "ai_generated",
            "generator": "frontier_tools",
            "model": "google-gemini-3-8-flash",
        },
        "tier": "easy",
    }
    row.update(overrides)
    return row


def test_legacy_frontier_row_traced_with_explicit_model(tmp_path):
    cohort = _make_cohort(tmp_path)
    legacy = _make_legacy_store(
        tmp_path, [_legacy_trace_row("aud-0-02280-t-google-gemini-3-8-flash")])
    sft, grpo = _make_mix(tmp_path, frontier_row=_legacy_mix_row())
    result = check_mix_propagation(sft, grpo, cohort, legacy_dirs=(legacy,))
    assert result.ok, result.problems
    # The fixture frontier row appears in both SFT and GRPO, so 2 legacy hits.
    assert "2 legacy" in result.detail
    assert "0 cohort" in result.detail


def test_legacy_frontier_row_without_model_label_fails(tmp_path):
    cohort = _make_cohort(tmp_path)
    legacy = _make_legacy_store(
        tmp_path, [_legacy_trace_row("aud-0-02280-t-google-gemini-3-8-flash")])
    row = _legacy_mix_row()
    del row["source_model"]
    del row["provenance"]["model"]
    sft, grpo = _make_mix(tmp_path, frontier_row=row)
    result = check_mix_propagation(sft, grpo, cohort, legacy_dirs=(legacy,))
    assert not result.ok
    assert any("no explicit" in p for p in result.problems)


def test_legacy_frontier_row_model_mismatch_fails(tmp_path):
    cohort = _make_cohort(tmp_path)
    legacy = _make_legacy_store(
        tmp_path, [_legacy_trace_row("aud-0-02280-t-google-gemini-3-8-flash")])
    row = _legacy_mix_row(
        source_model="gpt-6-luna",
        provenance={"source_family": "ai_generated", "generator": "frontier_tools",
                    "model": "gpt-6-luna"},
    )
    sft, grpo = _make_mix(tmp_path, frontier_row=row)
    result = check_mix_propagation(sft, grpo, cohort, legacy_dirs=(legacy,))
    assert not result.ok
    assert any("disagrees with legacy store model" in p for p in result.problems)


def test_untraceable_frontier_row_still_fails(tmp_path):
    cohort = _make_cohort(tmp_path)
    legacy = _make_legacy_store(tmp_path, [_legacy_trace_row(
        "unrelated-t-google-gemini-3-8-flash")])
    orphan = _legacy_mix_row(
        id="rep-999-99999-t-google-gemini-3-8-flash-ft-tool_cot-0d20",
        record_id="rep-999-99999-t-google-gemini-3-8-flash",
    )
    sft, grpo = _make_mix(tmp_path, frontier_row=orphan)
    result = check_mix_propagation(sft, grpo, cohort, legacy_dirs=(legacy,))
    assert not result.ok
    assert any("no legacy store match" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# 8. mix ref_spec leakage
# --------------------------------------------------------------------------- #
def test_mix_leakage_pass(tmp_path):
    sft, grpo = _make_mix(tmp_path)
    result = check_mix_leakage(sft, grpo)
    assert result.ok, result.problems


def test_mix_leakage_ref_spec_fails(tmp_path):
    frontier, deterministic = _mix_rows()
    frontier["ref_spec"] = {"components": []}
    sft = tmp_path / "sft_mix"
    grpo = tmp_path / "grpo_mix"
    _write_jsonl(sft / "repair_train.jsonl", [frontier, deterministic])
    _write_jsonl(grpo / "repair_train.jsonl", [frontier, deterministic])
    result = check_mix_leakage(sft, grpo)
    assert not result.ok
    assert any("ref_spec" in p for p in result.problems)


# --------------------------------------------------------------------------- #
# corpus smoke test (real data, no assertions about completeness)
# --------------------------------------------------------------------------- #
@pytest.mark.corpus
def test_real_audit_executes_all_checks():
    results = run_audit(
        ROOT / "data/frontier_tools_cohort",
        ROOT / "data/sft_mix_v3",
        ROOT / "data/grpo_mix_v3",
        ROOT / "data/memory",
        prompts=ROOT / "data/frontier_tools_cohort_prompts",
        sample=3,
        rebuild=False,
    )
    assert [r.name for r in results] == list(CHECK_NAMES)
    assert all(isinstance(r.detail, str) for r in results)
