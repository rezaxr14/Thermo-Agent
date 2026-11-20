"""No scoring-only ``ref_*`` answer fields may be written into a training mix row.

Design reference specs / evals / fuel targets live only in
``data/frontier_export/design_train.jsonl`` and are joined for scoring at scoring
time.  A mix row that stores them leaks the reference answer even when the rendered
prompt does not.  This guards both the written v2 mix (when present) and the
write-time scrubber itself.
"""
import json
import pathlib

import pytest

from scripts import mix_tool_corpora as mix

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIX_DIRS = ("data/sft_mix_v2", "data/grpo_mix_v2")
TIERS = ("easy", "medium", "hard")
SOURCES = {"deterministic_tools", "frontier_tools", "ground_truth"}


def _deep_ref_keys(obj) -> set[str]:
    """Every ``ref_*`` key anywhere in a row (mirrors the provenance audit)."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).startswith("ref_"):
                found.add(str(key))
            found |= _deep_ref_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            found |= _deep_ref_keys(value)
    return found


def _rows(directory: str) -> list[tuple[str, dict]]:
    root = ROOT / directory
    out: list[tuple[str, dict]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*_train.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append((path.name, json.loads(line)))
    return out


# --------------------------------------------------------------------------- #
# unit: the scrubber removes ref_* but keeps every field the training path needs
# --------------------------------------------------------------------------- #
def test_strip_answer_fields_removes_ref_keys_recursively():
    row = {
        "id": "design-1", "task": "design", "tier": "easy",
        "source": "frontier_tools", "prompt": [{"role": "user", "content": "..."}],
        "completion": [{"role": "assistant", "content": "<design>...</design>"}],
        "provenance": {"source_family": "ai_generated", "model": "gpt-6-luna", "run_id": "r1"},
        "source_model": "gpt-6-luna",
        "trajectory": {"sft_completion": "<design>...</design>",
                       "turns": [{"ref_spec": "SECRET", "content": "ok"}]},
        "ref_spec": "SECRET", "ref_fuel_w": 1234.0, "ref_evals": 17,
    }
    clean = mix.strip_answer_fields(row)
    assert not _deep_ref_keys(clean), "all ref_* answer keys must be removed"
    # Everything the training path needs survives untouched.
    assert clean["id"] == "design-1"
    assert clean["task"] == "design" and clean["tier"] == "easy"
    assert clean["source"] == "frontier_tools"
    assert clean["prompt"] == row["prompt"]
    assert clean["completion"] == row["completion"]
    assert clean["trajectory"]["sft_completion"] == row["trajectory"]["sft_completion"]
    assert clean["provenance"] == row["provenance"]
    assert clean["source_model"] == "gpt-6-luna"
    # Original row must not be mutated in place.
    assert _deep_ref_keys(row)


# --------------------------------------------------------------------------- #
# corpus: the committed v2 mix must be clean and still training-ready
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("directory", MIX_DIRS)
def test_mix_rows_carry_no_answer_fields(directory):
    rows = _rows(directory)
    if not rows:
        pytest.skip("v2 mix not built yet")
    bad = []
    for name, row in rows:
        leaked = _deep_ref_keys(row)
        if leaked:
            bad.append((name, row.get("id"), sorted(leaked)))
    assert not bad, f"{len(bad)} mix row(s) still carry ref_* answer fields: {bad[:5]}"


@pytest.mark.parametrize("directory", MIX_DIRS)
def test_mix_rows_keep_training_fields(directory):
    rows = _rows(directory)
    if not rows:
        pytest.skip("v2 mix not built yet")
    for name, row in rows:
        where = f"{directory}/{name}:{row.get('id')}"
        for field in ("id", "task", "tier", "source", "prompt"):
            assert row.get(field), f"{where}: missing {field}"
        assert row["tier"] in TIERS, f"{where}: bad tier {row['tier']!r}"
        assert row["source"] in SOURCES, f"{where}: bad source {row['source']!r}"
        assert row.get("completion") or row.get("trajectory"), (
            f"{where}: missing training target (completion/trajectory)")
        if row["source"] == "frontier_tools":
            prov = row.get("provenance") or {}
            assert prov.get("model") or row.get("model") or row.get("source_model"), (
                f"{where}: frontier row has no model id")


def test_mix_manifest_records_answer_field_scrub():
    manifest_path = ROOT / "data/sft_mix_v2/MANIFEST.json"
    if not manifest_path.is_file():
        pytest.skip("v2 mix manifest not built yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"].get("answer_fields_stripped") == "ref_*"
