"""heldout_v2 must have ZERO content overlap with any training artifact (F1 / plan 0.6).

Marked ``corpus`` because the training dirs are untracked; on a clone with the corpora present
it performs a full (uncapped) hash scan of every jsonl under data/ except the two held-out sets.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_heldout_v2 as b  # noqa: E402

V2 = b.ROOT / "data/heldout_v2"


@pytest.mark.corpus
def test_scan_covers_every_training_dir():
    dirs = {p.parent.relative_to(b.ROOT).as_posix() for p in b.target_files()}
    for required in ("a_slice", "sft_all", "grpo_all", "sft_tools", "grpo_tools",
                     "sft_tools_h", "grpo_tools_h", "frontier_export", "processed"):
        want = "data/" + required
        assert any(d == want or d.startswith(want + "/") for d in dirs), (want, sorted(dirs))
    # The two held-out sets are not training dirs and must be excluded.
    assert not any(d.startswith("heldout") for d in dirs)


@pytest.mark.corpus
def test_zero_overlap_with_all_training_data():
    index: set[tuple[str, str]] = set()
    rows = 0
    for path in sorted(V2.glob("*_test_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    index |= b.signature_set(json.loads(line))
                    rows += 1
    assert rows > 0 and index
    files = b.target_files()
    leaks, lines = b.scan_files(files, index)
    assert not leaks, (f"{rows} test rows, {len(index)} signatures, {len(files)} files, "
                       f"{lines} lines scanned; leaks: {leaks[:10]}")


@pytest.mark.corpus
def test_scanner_detects_a_planted_v2_row(tmp_path):
    index: set[tuple[str, str]] = set()
    source = sorted(V2.glob("*_test_*.jsonl"))[0]
    with source.open(encoding="utf-8") as fh:
        probe = json.loads(fh.readline())
    index |= b.signature_set(probe)
    planted = tmp_path / "planted.jsonl"
    planted.write_text(json.dumps(probe) + "\n", encoding="utf-8")
    leaks, lines = b.scan_files([planted], index)
    assert lines == 1
    assert leaks, "the overlap scanner failed to detect a planted v2 row"
