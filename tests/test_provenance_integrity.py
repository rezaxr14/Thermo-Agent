"""Fast integrity checks for per-row provenance and secret-free data files.

These tests stream / sample the on-disk corpora; they never load a model or call
an API.  Sampling is bounded so the suite stays well under a minute even though
``data/sft_all/repair_train.jsonl`` is over 1 GB.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTIER_DIR = ROOT / "data" / "frontier"
EXPORT_DIR = ROOT / "data" / "frontier_export"
SFT_DIR = ROOT / "data" / "sft_all"
GRPO_DIR = ROOT / "data" / "grpo_all"
DOC = ROOT / "docs" / "PROVENANCE.md"
LEDGER = ROOT / "data" / "provenance" / "trace_ledger.jsonl"

TASKS = ("repair", "ledger", "audit", "design")
SAMPLE = 2000

# (label, compiled pattern).  We only ever report the label, a file name and a
# line number -- never the matched text, so a real leak would not be printed.
SECRET_PATTERNS = {
    "KAGGLE_API_TOKEN": re.compile("KAGGLE_API_TOKEN"),
    "HF_TOKEN=": re.compile(re.escape("HF_TOKEN=")),
    "hf_<20+ alnum>": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "KGAT_": re.compile("KGAT_"),
}


def _iter_lines(path: Path, limit: int | None = None):
    """Yield non-blank lines from *path*, stopping after *limit* physical lines."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield line


def _has_completion(row: dict) -> bool:
    c = row.get("completion")
    if isinstance(c, str):
        return bool(c.strip())
    if isinstance(c, list):
        return any(
            isinstance(part, dict) and str(part.get("content", "")).strip()
            for part in c
        ) or bool(c)
    return False


def _model_provenance(row: dict) -> bool:
    """True when a row identifies the producing model.

    In this repository the model is encoded in the ``-t-<model>`` id suffix
    (documented in docs/PROVENANCE.md) and the source is ``frontier`` /
    ``frontier_tools``.  We also accept the more explicit encodings an exported
    row might carry: a ``model`` key or a ``frontier:...`` source.
    """
    model = row.get("model")
    if isinstance(model, str) and model.strip():
        return True
    source = str(row.get("source", ""))
    if re.search(r"frontier:[^:\s]+", source):  # e.g. source="frontier:google/gemini-..."
        return True
    return bool(re.search(r"-t-[A-Za-z0-9][A-Za-z0-9.\-]*$", str(row.get("id", ""))))


def _task_present(row: dict, path: Path) -> bool:
    if str(row.get("task", "")) in TASKS:
        return True
    name = path.name
    return any(task in name for task in TASKS)


def test_frontier_rows_have_provenance():
    """(a) Every data/frontier/*.jsonl row has id, task, completion and a model."""
    assert FRONTIER_DIR.is_dir(), f"missing {FRONTIER_DIR}"
    counts: dict[str, int] = {}
    mechanisms: dict[str, int] = {"model_key": 0, "source_colon": 0, "id_suffix": 0}
    violations: list[str] = []

    for path in sorted(FRONTIER_DIR.glob("*.jsonl")):
        n = 0
        for line in _iter_lines(path):
            row = json.loads(line)
            n += 1
            missing = []
            if not row.get("id"):
                missing.append("id")
            if not _task_present(row, path):
                missing.append("task")
            if not _has_completion(row):
                missing.append("completion")
            if not _model_provenance(row):
                missing.append("provenance(model)")
            if missing:
                violations.append(f"{path.name}:{n} missing {missing}")
            if isinstance(row.get("model"), str) and row["model"].strip():
                mechanisms["model_key"] += 1
            elif re.search(r"frontier:[^:\s]+", str(row.get("source", ""))):
                mechanisms["source_colon"] += 1
            else:
                mechanisms["id_suffix"] += 1
        counts[path.name] = n

    print("frontier counts:", counts)
    print("provenance mechanisms:", mechanisms)
    assert counts, "no frontier JSONL files found"
    assert not violations, "frontier provenance violations:\n" + "\n".join(violations[:20])


def test_combined_corpus_frontier_rows_have_source():
    """(b) Frontier rows in sft_all/grpo_all carry source starting with 'frontier'."""
    found = 0
    bad: list[str] = []
    for directory in (SFT_DIR, GRPO_DIR):
        dir_found = 0
        for path in sorted(directory.glob("*.jsonl")):
            for line in _iter_lines(path, SAMPLE):
                row = json.loads(line)
                is_frontier = (
                    str(row.get("source", "")).startswith("frontier")
                    or bool(row.get("record_id"))
                    or "-t-" in str(row.get("id", ""))
                )
                if not is_frontier:
                    continue
                dir_found += 1
                source = str(row.get("source", ""))
                if not source.startswith("frontier"):
                    bad.append(f"{directory.name}/{path.name}:{row.get('id')} source={source!r}")
        print(f"{directory.name}: frontier rows in first {SAMPLE} lines/file = {dir_found}")
        found += dir_found
    assert found > 0, "no frontier rows found in the sampled combined corpora"
    assert not bad, "frontier rows with a non-frontier source:\n" + "\n".join(bad[:20])


def test_no_secrets_in_data_files():
    """(c) No secret-looking token is present in frontier*, frontier_export or sampled sft_all."""
    hits: list[str] = []
    checked = 0
    sources: list[tuple[Path, int | None]] = (
        [(p, None) for p in sorted(FRONTIER_DIR.glob("*.jsonl"))]
        + [(p, None) for p in sorted(EXPORT_DIR.glob("*.jsonl"))]
        + [(p, SAMPLE) for p in sorted(SFT_DIR.glob("*.jsonl"))]
    )
    for path, limit in sources:
        for lineno, line in enumerate(_iter_lines(path, limit), 1):
            checked += 1
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    hits.append(f"{path.name}:{lineno} matched {label}")
    print(f"secret scan: {checked} lines checked, {len(hits)} hits")
    assert not hits, "secret-like tokens found:\n" + "\n".join(hits[:20])


def test_provenance_doc_trace_count_matches_ledger():
    """(d) docs/PROVENANCE.md states the same trace total as the ledger it cites."""
    assert DOC.is_file(), f"missing {DOC}"
    text = DOC.read_text(encoding="utf-8")

    doc_numbers: set[int] = set()
    for pattern in (
        r"Generated from\s+(\d+)\s+ledger records",
        r"\|\s*Traces\s*\|\s*(\d+)\s*\|",
    ):
        doc_numbers.update(int(x) for x in re.findall(pattern, text))
    assert doc_numbers, "could not parse a trace count from docs/PROVENANCE.md"
    assert len(doc_numbers) == 1, f"doc has inconsistent trace counts: {sorted(doc_numbers)}"
    doc_n = doc_numbers.pop()

    ledger_n = sum(1 for line in _iter_lines(LEDGER))
    print(f"trace counts: doc={doc_n} ledger={ledger_n}")
    assert doc_n == ledger_n, (
        f"trace-count mismatch: docs/PROVENANCE.md says {doc_n}, "
        f"but {LEDGER.relative_to(ROOT)} contains {ledger_n} records"
    )
