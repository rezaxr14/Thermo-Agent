"""Explicit per-row provenance on the additive frontier_prov copies.

The original ``data/frontier/*.jsonl`` rows only carry the producing model in the
``-t-<model>`` id suffix.  ``scripts/backfill_frontier_provenance.py`` writes copies
under ``data/frontier_prov/`` with explicit model/source/provider fields, and records
the originals' SHA-256 in a manifest.  These checks are pure file I/O: no network,
no model, no imports from the training stack.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = ROOT / "data" / "frontier"
PROV_DIR = ROOT / "data" / "frontier_prov"
MANIFEST_PATH = PROV_DIR / "provenance_manifest.json"

VALID_PROVIDERS = {"openai", "anthropic", "google", "other"}
# Refresh the backfill (and this expectation) whenever new frontier rows are ingested.
EXPECTED_TOTAL = 668


def _rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                yield lineno, json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> dict:
    assert MANIFEST_PATH.is_file(), f"missing {MANIFEST_PATH}; run the backfill script"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_every_provenance_row_has_explicit_fields():
    assert PROV_DIR.is_dir(), f"missing {PROV_DIR}; run the backfill script"
    paths = sorted(PROV_DIR.glob("*.jsonl"))
    assert paths, "no provenance JSONL files found"
    violations: list[str] = []
    total = 0
    for path in paths:
        for lineno, row in _rows(path):
            total += 1
            model = row.get("model")
            if not isinstance(model, str) or not model.strip():
                violations.append(f"{path.name}:{lineno} missing/blank model")
                continue
            source = str(row.get("source", ""))
            if not source.startswith("frontier:") or not source[len("frontier:"):].strip():
                violations.append(f"{path.name}:{lineno} bad source {source!r}")
            elif source[len("frontier:"):] != model:
                violations.append(f"{path.name}:{lineno} source {source!r} != model {model!r}")
            provider = row.get("provider")
            if provider not in VALID_PROVIDERS:
                violations.append(f"{path.name}:{lineno} invalid provider {provider!r}")
    assert not violations, "frontier_prov provenance violations:\n" + "\n".join(violations[:20])
    assert total == EXPECTED_TOTAL, (
        f"expected {EXPECTED_TOTAL} provenance rows, found {total}; rerun the backfill script"
    )


def test_provenance_counts_match_originals():
    manifest = _manifest()
    expected_names = set(manifest["files"])
    actual_names = {p.name for p in PROV_DIR.glob("*.jsonl")}
    assert actual_names == expected_names, (
        f"provenance files/manifest disagree: {sorted(actual_names)} vs {sorted(expected_names)}"
    )
    for name in sorted(expected_names):
        original = ORIGINAL_DIR / name
        prov = PROV_DIR / name
        assert original.is_file(), f"missing original {original}"
        n_original = sum(1 for _ in _rows(original))
        n_prov = sum(1 for _ in _rows(prov))
        n_manifest = manifest["files"][name]["rows"]
        assert n_prov == n_original == n_manifest, (
            f"{name}: original={n_original} provenance={n_prov} manifest={n_manifest}"
        )
    assert manifest["total_rows"] == sum(
        manifest["files"][name]["rows"] for name in expected_names
    )


def test_original_hashes_unchanged():
    manifest = _manifest()
    assert manifest.get("algorithm") == "sha256"
    assert manifest["files"], "manifest records no original files"
    for name, info in sorted(manifest["files"].items()):
        expected = info["sha256"]
        assert info["sha256_before"] == info["sha256_after"] == expected, (
            f"{name}: backfill saw a change in the original before/after hashes"
        )
        actual = _sha256(ORIGINAL_DIR / name)
        assert actual == expected, (
            f"data/frontier/{name} changed since backfill: expected {expected}, got {actual}"
        )
