"""Shared pytest configuration and small, committed data fixtures.

Fresh-clone contract (IMPLEMENTATION_PLAN.md row P0-14)
-------------------------------------------------------
Only a handful of ``data/`` directories are tracked in git (``heldout``,
``cards``, ``provenance``, ``raw``, ``splits``, ``frontier_eval_export``).
The big generated corpora (``data/processed*``, ``data/sft*``, ``data/grpo*``,
``data/frontier*``, ``data/memory``) are *untracked*.  On a fresh clone those
dirs do not exist, so data-hungry tests used to die with ``FileNotFoundError``
during collection or setup.

This module provides two things so that cannot happen:

1. A ``corpus`` marker plus a session-scoped ``real_corpus`` fixture (and an
   autouse guard) that **cleanly skips** any test marked ``@pytest.mark.corpus``
   when the untracked corpora are absent.  When the corpora are present the
   guard is a no-op, so existing tests keep their exact semantics.
2. Tiny deterministic fixtures under ``tests/fixtures/`` (2 rows per task,
   1 held-out row per task) exposed through pytest fixtures, so prompt /
   parse / tool-shape tests can run on a fresh clone without the real data.

The contamination test (``tests/test_contamination.py``) is deliberately *not*
marked ``corpus``: it must run on a fresh clone.  It tolerates missing corpora
by scanning whichever target files actually exist.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Directories that only exist after the (untracked) data-generation step.
REAL_CORPUS_DIRS = (
    "processed",
    "processed_arch",
    "processed_big",
    "sft",
    "sft_all",
    "grpo",
    "grpo_all",
    "frontier",
)


# ---------------------------------------------------------------------------
# Marker + fresh-clone skip guard
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "corpus: requires the untracked real data corpora "
        "(data/processed*, data/sft*, data/grpo*, data/frontier*); "
        "skipped cleanly on a fresh clone",
    )


@lru_cache(maxsize=1)
def real_corpus_available() -> bool:
    """True only when the generated corpora are actually on disk.

    Cached for the whole session.  We require every core directory to contain
    at least one jsonl file, so a partially generated tree cannot make a test
    fail with ``FileNotFoundError`` part-way through.
    """
    return all(
        (DATA / name).is_dir() and any((DATA / name).glob("*.jsonl"))
        for name in REAL_CORPUS_DIRS
    )


@pytest.fixture(scope="session")
def real_corpus() -> Path:
    """Session fixture granting access to the real corpora.

    A test marked ``@pytest.mark.corpus`` (or one that requests this fixture)
    is skipped instead of erroring when the untracked data is not present.
    """
    if not real_corpus_available():
        missing = [n for n in REAL_CORPUS_DIRS if not any((DATA / n).glob("*.jsonl"))]
        pytest.skip(
            "real data corpora are not present in this clone "
            f"(missing/empty: {', '.join(missing)}); "
            "run the data-generation scripts or use tests/fixtures"
        )
    return DATA


@pytest.fixture(autouse=True)
def _skip_corpus_without_real_data(request: pytest.FixtureRequest) -> None:
    """Attach the session guard to every ``@pytest.mark.corpus`` test.

    Using an autouse guard means existing test modules do not need to request
    ``real_corpus`` explicitly; the ``corpus`` marker is sufficient.  When the
    data is present this is a no-op and nothing is skipped.
    """
    if request.node.get_closest_marker("corpus") is not None:
        request.getfixturevalue("real_corpus")


# ---------------------------------------------------------------------------
# Tiny committed fixtures (available on every clone)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def corpus_fixture_dir() -> Path:
    """Directory with 2 small real rows per task (train split)."""
    return FIXTURES / "corpus"


@pytest.fixture(scope="session")
def heldout_fixture_dir() -> Path:
    """Directory with 1 small real row per task (held-out test split)."""
    return FIXTURES / "heldout"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="session")
def corpus_fixture_rows(corpus_fixture_dir: Path) -> dict[str, list[dict]]:
    """``{task: [row, row]}`` for all four tasks, from committed fixtures."""
    return {
        task: _load_jsonl(corpus_fixture_dir / f"{task}_train.jsonl")
        for task in ("repair", "design", "audit", "ledger")
    }


@pytest.fixture(scope="session")
def heldout_fixture_rows(heldout_fixture_dir: Path) -> dict[str, list[dict]]:
    """``{task: [row]}`` for all four tasks, from committed fixtures."""
    return {
        task: _load_jsonl(heldout_fixture_dir / f"{task}_test_id.jsonl")
        for task in ("repair", "design", "audit", "ledger")
    }
