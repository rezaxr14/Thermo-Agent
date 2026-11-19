"""The heldout_v2 test set is frozen: manifest hashes, counts and provenance must match disk."""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "data/heldout_v2"
MANIFEST = V2 / "MANIFEST.json"

TASKS = ("repair", "audit", "ledger", "design")
TIERS = ("easy", "medium", "hard")


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.exists(), "data/heldout_v2/MANIFEST.json is missing — run scripts/build_heldout_v2.py"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _rows(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_manifest_identity_and_provenance(manifest):
    assert manifest["eval_set"] == "heldout_v2"
    assert manifest["seed"] == 20261119
    assert manifest["target_rows_per_cell"] == 85
    assert manifest["scoring"]["limit"] == 0
    cmd = " ".join(manifest["generation_command"])
    assert "build_heldout_v2.py" in cmd
    assert "20261119" in cmd and "--target 85" in cmd
    assert manifest["workers"] == 12 and manifest["max_evals"] == 60
    assert manifest["generator_runs"], "generation commands must be recorded"


def test_set_hash_matches_disk(manifest):
    h = hashlib.sha256()
    for p in sorted(V2.glob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    assert h.hexdigest() == manifest["set_hash"]


def test_every_file_hash_matches_manifest(manifest):
    files = manifest["files"]
    on_disk = {p.name for p in V2.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
    assert on_disk == set(files), (on_disk ^ set(files))
    for name, meta in files.items():
        path = V2 / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == meta["sha256"], name
        assert path.stat().st_size == meta["bytes"], name
        if name.endswith(".jsonl"):
            assert manifest["jsonl_sha256"][name] == digest


def test_cells_match_files_and_are_test_only(manifest):
    assert manifest["cells"]
    for key, cell in manifest["cells"].items():
        task, split = key.split("/", 1)
        assert task in TASKS
        assert split.startswith("test_"), f"non-test split in test set: {key}"
        path = V2 / f"{task}_{split}.jsonl"
        assert path.exists(), key
        rows = _rows(path)
        assert len(rows) == cell["rows"], (key, len(rows), cell["rows"])
        assert {r["split"] for r in rows} == {split}, key
        assert sum(cell["tiers"].values()) == cell["rows"], key
    # No train/val file may exist in the frozen test set.
    assert not list(V2.glob("*train*.jsonl"))
    assert not list(V2.glob("*val*.jsonl"))


def test_manifest_records_tiers_and_contamination(manifest):
    for task in TASKS:
        t = manifest["tiers"][task]
        assert set(t["counts"]) == set(TIERS)
        assert sum(t["counts"].values()) == t["n"]
        assert t["measure"]
        assert set(t["ranges"]) >= set(TIERS)
    c = manifest["contamination"]
    assert c["passed"] is True
    assert c["overlap"] == 0
    assert c["self_check_detects_planted_row"] is True
    assert c["files_scanned"] > 0
