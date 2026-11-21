"""Design reference answers are joined at SCORING time, never stored in a training row.

The binding contract keeps ``ref_*`` fields out of the SFT/GRPO mixes; the only
scoring-only source is ``data/frontier_export/design_train.jsonl``. These tests
pin the join, its loud failures, and the fact that a scored row is never mutated.
CPU only, no model / network / Kaggle calls.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thermo_agent.train import grpo  # noqa: E402
from thermo_agent.design import rewards as DR  # noqa: E402

DESIGN_MIX = ROOT / "data" / "grpo_mix_v3" / "design_train.jsonl"
DESIGN_REFS = ROOT / "data" / "frontier_export" / "design_train.jsonl"


def _read_jsonl(path: pathlib.Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ref_keys(row: dict) -> set[str]:
    return {str(k) for k in row if str(k).startswith("ref_")}


def _scoring_pair():
    """A real stripped mix row plus its scoring-only reference, or a clean skip."""
    if not DESIGN_MIX.is_file() or not DESIGN_REFS.is_file():
        pytest.skip("scoring-only design export / v2 mix not present in this clone")
    refs = {r["id"]: r for r in _read_jsonl(DESIGN_REFS)}
    for row in _read_jsonl(DESIGN_MIX):
        for key in DR.identity_keys(row):
            if key in refs:
                return row, refs[key]
    pytest.skip("no design mix row resolves against the scoring-only export")


def _kwargs(row: dict) -> dict:
    """The columns TRL would pass for a stripped design row (no ref_* column)."""
    return {"site": [row["site"]], "id": [row["id"]],
            "record_id": [row.get("record_id", "")]}


# --------------------------------------------------------------------------- #
# 1. A stripped real mix row scores identically via the join (no silent zero).
# --------------------------------------------------------------------------- #
def test_stripped_mix_row_scores_via_scoring_time_join():
    row, ref = _scoring_pair()
    assert not _ref_keys(row), "the mix row must not carry a scoring answer"
    completion = row["completion"][0]["content"]
    kw = _kwargs(row)
    before = copy.deepcopy(kw)

    funcs, weights = grpo.reward_funcs_for_task("design")
    assert [f.__name__ for f in funcs] == ["d_format", "d_valid", "d_quality", "d_grounded"]
    assert len(funcs) == len(weights) == 4

    joined = [f(None, [completion], **kw) for f in funcs]
    # The same reward the row got while its ref_fuel_w column still existed.
    explicit = [f(None, [completion], site=[row["site"]],
                  ref_fuel_w=[ref["ref_fuel_w"]]) for f in funcs]
    assert joined == explicit
    assert joined[2][0] > 0.0, "the test row must actually be a valid design (not a zero default)"
    assert kw == before, "scoring must never add ref_* fields to the caller's row/kwargs"
    assert not _ref_keys(kw)


def test_join_returns_all_three_reference_fields_without_mutating_input():
    row, ref = _scoring_pair()
    kw = _kwargs(row)
    before = copy.deepcopy(kw)
    joined = DR._joined_reference_fields(kw, 1)
    assert set(joined) == set(DR.REF_FIELDS)
    assert joined["ref_fuel_w"] == [ref["ref_fuel_w"]]
    assert kw == before
    assert not _ref_keys(kw)


# --------------------------------------------------------------------------- #
# 2. Missing file / missing id fail loudly (never a silent zero).
# --------------------------------------------------------------------------- #
def test_missing_reference_file_fails_loudly(tmp_path, monkeypatch):
    missing = tmp_path / "no_frontier_export" / "design_train.jsonl"
    monkeypatch.setenv(DR.DESIGN_REFS_ENV, str(missing))
    DR.clear_reference_cache()
    try:
        with pytest.raises(DR.DesignReferenceError, match="reference file is missing"):
            DR.d_quality(None, ["<design></design>"],
                         site=["{}"], id=["des-0-00047"], record_id=["des-0-00047"])
    finally:
        DR.clear_reference_cache()


def test_missing_id_fails_loudly(tmp_path, monkeypatch):
    path = tmp_path / "design_train.jsonl"
    path.write_text(json.dumps({"id": "des-0-00001", "ref_fuel_w": 100.0,
                                "ref_spec": "{}", "ref_evals": 1}) + "\n", encoding="utf-8")
    monkeypatch.setenv(DR.DESIGN_REFS_ENV, str(path))
    DR.clear_reference_cache()
    try:
        with pytest.raises(DR.DesignReferenceError, match="no design reference answer") as exc:
            DR.d_quality(None, ["<design></design>"],
                         site=["{}"], id=["des-0-99999"], record_id=["des-0-99999"])
        assert "des-0-99999" in str(exc.value)
        assert "zero/default" in str(exc.value)
    finally:
        DR.clear_reference_cache()


def test_row_without_an_id_fails_loudly(tmp_path, monkeypatch):
    path = tmp_path / "design_train.jsonl"
    path.write_text(json.dumps({"id": "des-0-00001", "ref_fuel_w": 100.0,
                                "ref_spec": "{}", "ref_evals": 1}) + "\n", encoding="utf-8")
    monkeypatch.setenv(DR.DESIGN_REFS_ENV, str(path))
    DR.clear_reference_cache()
    try:
        with pytest.raises(DR.DesignReferenceError, match="neither 'record_id' nor 'id'"):
            DR.d_quality(None, ["<design></design>"], site=["{}"])
    finally:
        DR.clear_reference_cache()


def test_preflight_reports_incomplete_reference_export(tmp_path, monkeypatch):
    path = tmp_path / "design_train.jsonl"
    path.write_text(json.dumps({"id": "des-0-00001", "ref_fuel_w": 100.0,
                                "ref_spec": "{}", "ref_evals": 1}) + "\n", encoding="utf-8")
    monkeypatch.setenv(DR.DESIGN_REFS_ENV, str(path))
    DR.clear_reference_cache()
    try:
        DR.assert_references_for_rows([{"id": "des-0-00001", "record_id": "des-0-00001"}])
        with pytest.raises(DR.DesignReferenceError, match="1 of 2 design rows") as exc:
            DR.assert_references_for_rows([
                {"id": "des-0-00001", "record_id": "des-0-00001"},
                {"id": "des-0-99999", "record_id": "des-0-99999"},
            ])
        assert "des-0-99999" in str(exc.value)
    finally:
        DR.clear_reference_cache()


# --------------------------------------------------------------------------- #
# 3. Loader never puts ref_* into training rows but keeps the join keys.
# --------------------------------------------------------------------------- #
def test_joint_rows_drop_ref_fields_and_keep_join_keys(tmp_path):
    for task in grpo.JOINT_TASKS:
        row = {"id": f"{task}-0", "record_id": f"{task}-0", "task": task, "tier": "easy",
               "prompt": [{"role": "user", "content": "x"}],
               "plant_broken": "{}", "plant": "{}", "site": "{}", "ledger": "{}"}
        if task == "design":
            row.update(ref_fuel_w=1234.0, ref_spec="{}", ref_evals=1)
        (tmp_path / f"{task}_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = grpo.joint_rows(tmp_path)
    assert all(not _ref_keys(r) for r in rows), "no ref_* may reach a training row"
    design = next(r for r in rows if r["task"] == "design")
    assert design["id"] == "design-0" and design["record_id"] == "design-0"
    assert "ref_fuel_w" not in design and "ref_spec" not in design


# --------------------------------------------------------------------------- #
# 4. Reward dispatch for all four tasks / stage names remain unchanged.
# --------------------------------------------------------------------------- #
def test_reward_dispatch_and_stage_names_are_unchanged():
    _, _, owner = grpo.joint_rewards()
    assert set(owner.values()) == set(grpo.JOINT_TASKS)
    assert {name.split("_")[0]: task for name, task in owner.items()} == {
        "r": "repair", "d": "design", "a": "audit", "l": "ledger"}
    for task in ("repair", "design", "audit", "ledger"):
        funcs, weights = grpo.reward_funcs_for_task(task)
        assert funcs and len(funcs) == len(weights)
    for task in ("repair", "design", "audit", "ledger"):
        for k in (4, 8, 16):
            assert grpo.stage_name(task, k) == f"grpo_k{k}"


def test_non_design_rows_do_not_trigger_the_design_join():
    # dispatch_rewards filters each function's rows before calling it: a batch with
    # no design rows must not consult the reference export at all.
    (d_quality,) = grpo.dispatch_rewards([DR.d_quality])
    batch = dict(prompts=["p0", "p1"], completions=["c0", "c1"],
                 task=["repair", "ledger"])
    assert d_quality(**batch) == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# 5. Identity normalisation for teacher/model suffixes and paraphrase variants.
# --------------------------------------------------------------------------- #
def test_identity_keys_use_record_id_first_then_ground_truth_base():
    keys = DR.identity_keys({"record_id": "des-0-00047-t-gpt-6-luna",
                             "id": "des-0-00047-t-gpt-6-luna#v2"})
    assert keys[:2] == ("des-0-00047-t-gpt-6-luna", "des-0-00047")


def test_multisegment_model_suffix_in_record_id_resolves_to_base(tmp_path, monkeypatch):
    """The 7 mix rows with a `record_id` like `des-0-03009-t-google-gemini-3-8-flash`
    are NOT unresolved: the join must strip the whole `-t-...` suffix and find the
    ground-truth row keyed by `des-0-03009`. Pin it so a stricter (exact-id) join
    cannot silently reintroduce them as zero-scored rows.
    """
    base = "des-0-03009"
    path = tmp_path / "design_train.jsonl"
    path.write_text(json.dumps({"id": base, "ref_fuel_w": 100.0,
                                "ref_spec": "{}", "ref_evals": 1}) + "\n", encoding="utf-8")
    monkeypatch.setenv(DR.DESIGN_REFS_ENV, str(path))
    DR.clear_reference_cache()
    try:
        row = {"record_id": f"{base}-t-google-gemini-3-8-flash",
               "id": f"{base}-t-google-gemini-3-8-flash-t-ft-tool_cot-0d20"}
        assert base in DR.identity_keys(row)
        DR.assert_references_for_rows([row])
        joined = DR._joined_reference_fields(
            {"record_id": [row["record_id"]], "id": [row["id"]]}, 1)
        assert joined["ref_fuel_w"] == [100.0]
    finally:
        DR.clear_reference_cache()
