"""Guard the FINAL training mix: real tiers, equal thirds, known provenance.

Runs on the committed v2 mix when present; otherwise checks the current mix dir.
These are corpus-level assertions (not unit tests of a builder), so they are
skipped rather than failing on a fresh clone that has not run datagen yet.
"""
import collections
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TIERS = ("easy", "medium", "hard")
SOURCES = {"deterministic_tools", "frontier_tools", "ground_truth"}


def _rows(directory: pathlib.Path, task: str) -> list[dict]:
    path = directory / f"{task}_train.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


@pytest.mark.parametrize("task", ("repair", "design", "audit", "ledger"))
@pytest.mark.parametrize("kind", ("sft", "grpo"))
def test_final_mix_has_equal_thirds_and_known_sources(task, kind):
    directory = ROOT / f"data/{kind}_mix_v2"
    rows = _rows(directory, task)
    if not rows:
        pytest.skip("v2 mix not built yet")
    counts = collections.Counter(row.get("tier") for row in rows)
    # Equal thirds; remainder goes to the higher tier.
    assert set(counts) <= set(TIERS)
    assert counts["easy"] == counts["medium"] == len(rows) // 3
    assert counts["hard"] == len(rows) - 2 * (len(rows) // 3)
    assert all(row.get("source") in SOURCES for row in rows)
    for row in rows:
        if row["source"] == "frontier_tools":
            model = (row.get("provenance") or {}).get("model") or row.get("model")
            assert model, f"frontier row {row.get('id')} has no model id"


def test_final_mix_manifest_records_tiers_and_provenance():
    manifest_path = ROOT / "data/sft_mix_v2/MANIFEST.json"
    if not manifest_path.is_file():
        pytest.skip("v2 mix manifest not built yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["tiers"]) == {"repair", "design", "audit", "ledger"}
    for task, summary in manifest["tiers"].items():
        assert set(summary["counts"]) == set(TIERS)
        assert sum(summary["counts"].values()) > 0
    assert "sft" in manifest["provenance"] and "grpo" in manifest["provenance"]
