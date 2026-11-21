#!/usr/bin/env python3
"""Reproducible, streaming dataset statistics for the quoted paper/doc tables.

One command regenerates both a deterministic JSON snapshot and a markdown table:

    python scripts/dataset_stats.py \
        --dirs data/sft_tools data/grpo_tools data/sft_all data/grpo_all \
               data/frontier data/heldout \
        --out data/provenance/dataset_stats_2026-11-19.json \
        --markdown reports/dataset_stats.md

Design rules
------------
* Streaming: every input is read one line at a time; no corpus (and no row
  text) is ever held in memory.  Only small numeric aggregates and the input
  file hashes accumulate.
* Deterministic: keys are sorted, directory/task tables are sorted, and no
  timestamp or other wall-clock value goes into either output.  Re-running on
  unchanged inputs produces byte-identical files.  ``generated_from`` carries
  the SHA-256 of every input file, so a data change is visible immediately.
* Safe by construction: only ``*.jsonl`` files directly inside the named
  directories are read.  ``.env`` files, symlinks and hidden files are never
  opened.  The outputs contain counts, lengths and hashes only -- never a
  prompt, completion, answer or any other raw field value.
* Tokens: counted with the *local* Qwen3.5 tokenizer file (the same
  ``tokenizer.json`` that ``synth.combine`` uses).  No model is downloaded; if
  no local tokenizer file exists, token counts are reported as ``not
  measured``.  ``--no-tokens`` forces that path (used by the tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_ENV = "THERMO_TOKENIZER"
DEFAULT_TOKENIZER = Path.home() / ".cache" / "thermo" / "qwen3.5-4b" / "tokenizer.json"
TOKEN_BATCH = 512

KNOWN_TASKS = ("repair", "design", "audit", "ledger")
MISSING = "<missing>"
UNKNOWN = "<unknown>"
TIER_BY_SOURCE_DIR = {
    "data/processed": "easy",
    "data/processed_arch": "medium",
    "data/processed_big": "hard",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _path(root: Path, value: str | Path) -> Path:
    """Resolve a CLI path against ``root`` without depending on the cwd."""
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve()


def _safe_output(path: Path) -> Path:
    """Refuse hidden/``.env`` output names so we never write a secret file."""
    if path.name.startswith(".") or path.name == ".env" or path.suffix == ".env":
        raise ValueError(f"refusing to write to a hidden/env path: {path}")
    return path


def discover_files(root: Path, dirs: list[str]) -> list[tuple[str, Path]]:
    """Whitelist the inputs: top-level ``*.jsonl`` files in the named dirs.

    Returns ``(dir_label, path)`` pairs sorted by label then filename.  Hidden
    files, symlinks and anything whose name mentions ``.env`` are skipped.
    """
    found: set[tuple[str, Path]] = set()
    for raw in dirs:
        directory = _path(root, raw)
        if not directory.is_dir():
            continue
        try:
            label = directory.relative_to(root).as_posix()
        except ValueError:
            label = directory.name
        for path in directory.glob("*.jsonl"):
            name = path.name
            if (not path.is_file() or path.is_symlink() or name.startswith(".")
                    or ".env" in name.lower()):
                continue
            found.add((label, path.resolve()))
    return sorted(found, key=lambda item: (item[0], item[1].name))


def load_local_tokenizer(explicit: str | None) -> tuple[Any | None, str | None]:
    """Load the real Qwen3.5 tokenizer from a *local* file; never download.

    ``transformers.AutoTokenizer.from_pretrained`` is deliberately not used:
    it can reach the network.  ``tokenizers.Tokenizer.from_file`` cannot.
    """
    candidates = [explicit, os.environ.get(TOKENIZER_ENV, ""), str(DEFAULT_TOKENIZER)]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return None, None
        try:
            return Tokenizer.from_file(str(path)), str(path)
        except Exception:
            return None, None
    return None, None


def text_of(value: Any) -> str:
    """Flatten the repo's message-list / string text fields to plain text.

    Message contents are concatenated without a separator, matching the
    convention used by ``thermo_agent.synth.combine.Budget.ntok``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(text_of(item.get("content", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("content", "final", "sft_completion"):
            if key in value:
                return text_of(value[key])
        return ""
    return ""


def trajectory_text(row: dict[str, Any]) -> str:
    """All text inside a stored trajectory (tool turns included)."""
    traj = row.get("trajectory")
    if not isinstance(traj, dict):
        return ""
    parts = [text_of(traj.get("sft_completion")), text_of(traj.get("final"))]
    for turn in traj.get("turns") or []:
        if isinstance(turn, dict):
            parts.append(text_of(turn))
            for call in turn.get("tool_calls") or []:
                parts.append(text_of(call))
    return "".join(parts)


def task_of(row: dict[str, Any], path: Path) -> str:
    """Row ``task`` when present, otherwise infer it from the filename.

    Held-out rows have no ``task`` key; their filename always starts with the
    task (``repair_test_id.jsonl``, ``frontier_audit_train.jsonl`` ...).
    """
    value = row.get("task")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for part in path.stem.split("_"):
        if part in KNOWN_TASKS:
            return part
    return UNKNOWN


def tier_of(row: dict[str, Any]) -> str:
    """Explicit tier field if present, else a tier from ``provenance.source_dir``."""
    for key in ("tier", "difficulty", "level"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    prov = row.get("provenance")
    if isinstance(prov, dict):
        source_dir = prov.get("source_dir")
        if isinstance(source_dir, str) and source_dir.strip():
            normalized = source_dir.strip().replace("\\", "/")
            return TIER_BY_SOURCE_DIR.get(normalized, normalized)
    return MISSING


def _label(value: Any) -> str:
    if value is None:
        return MISSING
    text = str(value)
    return text if text.strip() else MISSING


def quantiles(values: list[int], fraction: float) -> float | None:
    """Linear-interpolation quantile; stable and independent of input order."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": round(quantiles(values, 0.5)),
        "p95": round(quantiles(values, 0.95)),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# aggregation (counters + integer length lists only -- never raw text)
# ---------------------------------------------------------------------------
class Bucket:
    """Numeric aggregates for one directory, or for one task inside it."""

    def __init__(self) -> None:
        self.rows = 0
        self.malformed = 0
        self.per_split: Counter[str] = Counter()
        self.per_tier: Counter[str] = Counter()
        self.per_source: Counter[str] = Counter()
        self.per_harness: Counter[str] = Counter()
        self.flags: Counter[str] = Counter()
        self.prompt_chars: list[int] = []
        self.completion_chars: list[int] = []
        self.prompt_tokens: list[int] = []
        self.completion_tokens: list[int] = []

    def add(self, row: dict[str, Any], prompt: str, completion: str) -> None:
        """Add one row's numeric aggregates (never the row text itself)."""
        self.rows += 1
        self.per_split[_label(row.get("split"))] += 1
        self.per_tier[tier_of(row)] += 1
        self.per_source[_label(row.get("source"))] += 1
        self.per_harness[_label(row.get("harness"))] += 1

        if row.get("completion") is not None:
            self.flags["with_completion"] += 1
        if isinstance(row.get("trajectory"), dict):
            self.flags["with_trajectory"] += 1
        haystack = completion + trajectory_text(row)
        if "<tool_call" in haystack:
            self.flags["with_tool_call"] += 1
        if "<tool_result" in haystack:
            self.flags["with_tool_result"] += 1

        self.prompt_chars.append(len(prompt))
        self.completion_chars.append(len(completion))

    def to_dict(self, tokenizer_status: str, tokenizer_path: str | None,
                tokenizer_reason: str | None) -> dict[str, Any]:
        measured = tokenizer_status == "measured"
        return {
            "rows": self.rows,
            "malformed_lines": self.malformed,
            "per_split": dict(sorted(self.per_split.items())),
            "per_tier": dict(sorted(self.per_tier.items())),
            "source_values": dict(sorted(self.per_source.items())),
            "harness_values": dict(sorted(self.per_harness.items())),
            "flags": {
                "with_stored_completion": self.flags["with_completion"],
                "with_trajectory": self.flags["with_trajectory"],
                "with_tool_call": self.flags["with_tool_call"],
                "with_tool_result": self.flags["with_tool_result"],
            },
            "chars": {
                "prompt": length_stats(self.prompt_chars),
                "completion": length_stats(self.completion_chars),
            },
            "tokens": {
                "status": tokenizer_status,
                "tokenizer": tokenizer_path,
                "reason": tokenizer_reason,
                "prompt": length_stats(self.prompt_tokens) if measured else None,
                "completion": length_stats(self.completion_tokens) if measured else None,
            },
        }


def _short_counts(counter: dict[str, int], limit: int = 3) -> str:
    if not counter:
        return "-"
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{key}={value}" for key, value in items[:limit])
    if len(items) > limit:
        shown += f", +{len(items) - limit}"
    return shown


def _length_cell(stats: dict[str, Any] | None) -> str:
    if not stats or stats.get("n") == 0:
        return "n/a"
    return f"{stats['min']}/{stats['median']}/{stats['p95']}/{stats['max']}"


def build_markdown(stats: dict[str, Any]) -> str:
    """One table, one row per (dir, task), sorted by dir then task."""
    tok = stats["tokenizer"]
    status = tok["status"]
    token_note = (
        f"measured with the local tokenizer `{tok['path']}`"
        if status == "measured" else
        "**not measured** (no local Qwen3.5 tokenizer; no model download attempted)"
    )
    header = [
        "# Dataset statistics",
        "",
        "Generated by `scripts/dataset_stats.py`. This table contains counts and "
        "lengths only -- no prompt, completion or answer text is reproduced.",
        "",
        f"- Token counts: {token_note}.",
        f"- Input files hashed: {len(stats['generated_from'])}.",
        "- Length columns are min / median / p95 / max.",
        "",
        "| Dir | Task | Rows | Splits | Source values | Harness values | Tier values "
        "| tool_call | tool_result | completion stored | trajectory stored "
        "| prompt chars | completion chars | prompt tokens | completion tokens |",
        "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: "
        "| --- | --- | --- | --- |",
    ]
    lines = list(header)
    for dir_name in sorted(stats["dirs"]):
        d = stats["dirs"][dir_name]
        for task_name in sorted(d["per_task"]):
            t = d["per_task"][task_name]
            f = t["flags"]
            lines.append(
                "| {dir} | {task} | {rows} | {split} | {source} | {harness} | {tier} "
                "| {tc} | {tr} | {comp} | {traj} | {pc} | {cc} | {pt} | {ct} |".format(
                    dir=_md(dir_name), task=_md(task_name), rows=t["rows"],
                    split=_md(_short_counts(t["per_split"])),
                    source=_md(_short_counts(t["source_values"])),
                    harness=_md(_short_counts(t["harness_values"])),
                    tier=_md(_short_counts(t["per_tier"])),
                    tc=f["with_tool_call"], tr=f["with_tool_result"],
                    comp=f["with_stored_completion"], traj=f["with_trajectory"],
                    pc=_length_cell(t["chars"]["prompt"]),
                    cc=_length_cell(t["chars"]["completion"]),
                    pt=_length_cell(t["tokens"]["prompt"]),
                    ct=_length_cell(t["tokens"]["completion"]),
                )
            )
    lines += [
        "",
        "Notes: `task` is inferred from the filename when a row has no `task` field "
        "(held-out files). `tier` falls back to `provenance.source_dir` "
        "(processed=easy, processed_arch=medium, processed_big=hard) when no "
        "explicit tier field exists. On rows with a `trajectory` (tool corpora), "
        "`completion` holds only the final answer; tool turns live in "
        "`trajectory.sft_completion` and are used for tool-tag counts but not for "
        "the completion-length columns. Source/harness cells show the three most "
        "common field values.",
        "",
    ]
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# main streaming driver
# ---------------------------------------------------------------------------
def collect(root: Path, dirs: list[str], tokenizer: Any | None,
            tokenizer_path: str | None) -> tuple[dict[str, Any], int]:
    files = discover_files(root, dirs)
    status = "measured" if tokenizer is not None else "not measured"
    reason = None if tokenizer is not None else (
        "no local Qwen3.5 tokenizer file found; pass --tokenizer or set "
        f"{TOKENIZER_ENV}; no model download attempted"
    )

    dir_buckets: dict[str, Bucket] = {}
    task_buckets: dict[str, dict[str, Bucket]] = defaultdict(dict)
    generated_from: dict[str, dict[str, Any]] = {}
    total = Bucket()

    # Tokenisation batch: flatten prompt/completion pairs, encode in one call
    # per batch (the Rust tokenizer parallelises encode_batch internally).
    batch_texts: list[str] = []
    batch_targets: list[tuple[Bucket, Bucket, Bucket]] = []

    def flush_tokens() -> None:
        if not batch_texts:
            return
        encodings = tokenizer.encode_batch(batch_texts, add_special_tokens=False)
        for index, (db, tb, allb) in enumerate(batch_targets):
            n_prompt = len(encodings[2 * index].ids)
            n_completion = len(encodings[2 * index + 1].ids)
            for bucket in (db, tb, allb):
                bucket.prompt_tokens.append(n_prompt)
                bucket.completion_tokens.append(n_completion)
        batch_texts.clear()
        batch_targets.clear()

    for label, path in files:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        db = dir_buckets.setdefault(label, Bucket())
        hasher = hashlib.sha256()
        file_rows = 0
        with path.open("rb") as fh:
            for raw in fh:
                hasher.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeDecodeError):
                    total.malformed += 1
                    db.malformed += 1
                    continue
                if not isinstance(row, dict):
                    total.malformed += 1
                    db.malformed += 1
                    continue
                file_rows += 1
                task = task_of(row, path)
                tb = task_buckets[label].setdefault(task, Bucket())
                prompt = text_of(row.get("prompt"))
                completion = text_of(row.get("completion"))
                for bucket in (db, tb, total):
                    bucket.add(row, prompt, completion)
                if tokenizer is not None:
                    batch_texts.extend((prompt, completion))
                    batch_targets.append((db, tb, total))
                    if len(batch_targets) >= TOKEN_BATCH:
                        flush_tokens()
        flush_tokens()
        generated_from[rel] = {
            "sha256": hasher.hexdigest(),
            "rows": file_rows,
            "bytes": path.stat().st_size,
        }

    dirs_out: dict[str, Any] = {}
    for label in sorted(dir_buckets):
        db = dir_buckets[label]
        dirs_out[label] = db.to_dict(status, tokenizer_path, reason)
        dirs_out[label]["files"] = sum(1 for d, _ in files if d == label)
        dirs_out[label]["per_task"] = {
            task: tb.to_dict(status, tokenizer_path, reason)
            for task, tb in sorted(task_buckets[label].items())
        }

    stats: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "scripts/dataset_stats.py",
        "note": "counts and lengths only; no prompt, completion or answer text",
        "tokenizer": {"status": status, "path": tokenizer_path, "reason": reason},
        "generated_from": dict(sorted(generated_from.items())),
        "totals": total.to_dict(status, tokenizer_path, reason),
        "dirs": dirs_out,
    }
    return stats, len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="dataset directories (relative to --root)")
    parser.add_argument("--out", required=True, help="deterministic JSON output")
    parser.add_argument("--markdown", required=True, help="markdown table output")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument("--no-tokens", action="store_true",
                        help="force token counts to 'not measured' (no tokenizer loaded)")
    parser.add_argument("--tokenizer", default=None,
                        help="explicit local Qwen3.5 tokenizer.json (never downloaded)")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out = _safe_output(_path(root, args.out))
    markdown = _safe_output(_path(root, args.markdown))

    tok, tok_path = (None, None) if args.no_tokens else load_local_tokenizer(args.tokenizer)
    stats, n_files = collect(root, args.dirs, tok, tok_path)

    payload = json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(build_markdown(stats), encoding="utf-8")

    print(f"directories: {len(stats['dirs'])}  files: {n_files}  "
          f"rows: {stats['totals']['rows']}  malformed: {stats['totals']['malformed_lines']}")
    for dir_name in sorted(stats["dirs"]):
        print(f"  {dir_name}: {stats['dirs'][dir_name]['rows']} rows")
    print(f"token counts: {stats['tokenizer']['status']}")
    print(f"wrote {out}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
