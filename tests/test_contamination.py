"""Contamination guard: the frozen held-out test set must never appear in any
artifact a model can train on (IMPLEMENTATION_PLAN.md rows P0-13 / P0-14).

What is checked
---------------
Signatures are derived **only** from the held-out ``test_*`` splits (train/val
in ``data/heldout`` are generation helpers, not the frozen eval set).  For each
row we hash high-entropy identity fields:

* plants  -- ``plant`` / ``plant_broken`` / ``plant_original``
* sites   -- ``site``
* specs   -- ``ledger`` and a joint ``site + ref_spec`` digest (a bare
  ``ref_spec`` is a small, heavily repeated template and would false-positive)

plus the row ``id``.  These digests are searched for in ``data/sft*``,
``data/grpo*``, ``data/frontier*``, the teacher prompt exports and
``data/memory`` notes.  JSON stored as a string is parsed before hashing so a
verbatim leak is found regardless of field order / whitespace.

Only the first ``MAX_LINES`` lines of each file are scanned, so the 1.4 GB
``sft_all/repair_train.jsonl`` never blows the runtime budget.  Missing
directories (a fresh clone) are simply skipped -- this test is intentionally
**not** marked ``corpus``.

The self-check writes a held-out row into a temp training file and asserts the
scanner flags it, so a green run proves the scanner can actually fail.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HELDOUT = DATA / "heldout"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

MAX_LINES = 5000

# High-entropy identity fields.  ``ref_spec`` is deliberately absent on its
# own (it is a repeated template); it is folded into the joint spec digest.
PLANT_FIELDS = ("plant", "plant_broken", "plant_original")
SITE_FIELDS = ("site",)
SPEC_FIELDS = ("ledger",)
IDENTITY_FIELDS = PLANT_FIELDS + SITE_FIELDS + SPEC_FIELDS


# ---------------------------------------------------------------------------
# Signature construction
# ---------------------------------------------------------------------------
def _parse(value):
    """Return a dict/list for JSON-or-JSON-string values, else None."""
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, (dict, list)) and value:
        return value
    return None


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def signature_set(row: dict) -> set[tuple[str, str]]:
    """All ``(kind, sha256)`` signatures of a single dataset row."""
    out: set[tuple[str, str]] = set()

    for name in IDENTITY_FIELDS:
        obj = _parse(row.get(name))
        if obj is not None:
            out.add((name, _digest(obj)))

    # Joint design spec: the spec template is shared, but the site is random,
    # so the pair is a unique fingerprint of a design row.
    site = _parse(row.get("site"))
    spec = _parse(row.get("ref_spec"))
    if site is not None and spec is not None:
        out.add(("site+ref_spec", _digest({"site": site, "ref_spec": spec})))

    rid = row.get("id")
    if isinstance(rid, str) and rid:
        out.add(("id", _digest(rid)))

    return out


@dataclass
class LeakIndex:
    signatures: set[tuple[str, str]] = field(default_factory=set)
    ids: set[str] = field(default_factory=set)


def build_heldout_index(heldout_dir: Path = HELDOUT) -> LeakIndex:
    """Hash plants/sites/specs from every held-out *test* split."""
    index = LeakIndex()
    files = sorted(heldout_dir.glob("*_test_*.jsonl"))
    if not files:
        # Fresh-clone fallback: the committed tiny held-out fixtures.
        files = sorted((FIXTURES / "heldout").glob("*_test_*.jsonl"))
    assert files, f"no held-out test splits found under {heldout_dir}"

    for path in files:
        for row in _iter_jsonl(path):
            index.signatures |= signature_set(row)
            rid = row.get("id")
            if isinstance(rid, str) and rid:
                index.ids.add(rid)
    return index


# ---------------------------------------------------------------------------
# File discovery + scanning
# ---------------------------------------------------------------------------
def _iter_jsonl(path: Path, max_lines: int = MAX_LINES):
    """Yield up to ``max_lines`` non-empty, parseable json objects."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in itertools.islice(fh, max_lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                yield row


def target_files() -> list[Path]:
    """Every training / prompt / memory artifact worth scanning.

    Globs are non-fatal: absent directories (fresh clone) simply contribute
    no files.
    """
    patterns = (
        "sft*/*.jsonl",
        "grpo*/*.jsonl",
        "frontier*/*.jsonl",
        "memory/*.jsonl",
        "**/prompts*.jsonl",
        "**/*prompt*.jsonl",
    )
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in DATA.glob(pattern) if p.is_file())
    return sorted(found)


@dataclass
class Leak:
    path: str
    line: int
    kinds: tuple[str, ...]


def scan_paths(
    paths, index: LeakIndex, max_lines: int = MAX_LINES
) -> tuple[list[Leak], int]:
    """Scan files for held-out signatures; return (leaks, lines_scanned)."""
    leaks: list[Leak] = []
    lines = 0
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line_no, raw in enumerate(itertools.islice(fh, max_lines), 1):
                if not raw.strip():
                    continue
                lines += 1
                kinds: set[str] = set()

                # Structured JSON signatures.
                try:
                    row = json.loads(raw)
                except (ValueError, TypeError):
                    row = None
                if isinstance(row, dict):
                    kinds |= {
                        kind for kind, h in signature_set(row) if (kind, h) in index.signatures
                    }

                if kinds:
                    try:
                        label = str(path.relative_to(ROOT))
                    except ValueError:
                        label = str(path)
                    leaks.append(Leak(label, line_no, tuple(sorted(kinds))))
    return leaks, lines


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_training_artifacts_are_free_of_heldout_signatures():
    index = build_heldout_index()
    files = target_files()
    leaks, lines = scan_paths(files, index)
    summary = (
        f"\ncontamination scan: {len(index.ids)} held-out ids, "
        f"{len(index.signatures)} signatures, {len(files)} files, "
        f"{lines} lines sampled"
    )
    assert not leaks, summary + f"\nleaks: {leaks[:10]}"
    print(summary)


def test_scanner_detects_planted_heldout_leak(tmp_path: Path):
    """Self-check: a held-out row copied into a training file MUST be flagged."""
    index = build_heldout_index()

    # A real held-out row (fall back to the committed fixture on any clone).
    source = HELDOUT / "repair_test_id.jsonl"
    if not source.exists():
        source = FIXTURES / "heldout" / "repair_test_id.jsonl"
    rows = list(_iter_jsonl(source, max_lines=1))
    assert rows, f"need one held-out row from {source}"
    leaked_row = rows[0]
    assert leaked_row["id"] in index.ids

    leak_file = tmp_path / "planted_training_file.jsonl"
    leak_file.write_text(json.dumps(leaked_row, ensure_ascii=False) + "\n", encoding="utf-8")

    leaks, lines = scan_paths([leak_file], index)
    assert lines == 1
    assert leaks, "the scanner failed to detect a planted held-out row"
    detected = {kind for leak in leaks for kind in leak.kinds}
    assert {"id", "plant_broken"} <= detected, detected

    # A clean (non-held-out) row must not be flagged.
    clean = FIXTURES / "corpus" / "repair_train.jsonl"
    if clean.exists():
        clean_leaks, _ = scan_paths([clean], index)
        assert not clean_leaks, f"clean corpus row falsely flagged: {clean_leaks[:3]}"
