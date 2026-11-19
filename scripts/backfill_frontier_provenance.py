#!/usr/bin/env python3
"""Add explicit per-row provenance to the frontier corpus -- additively.

Reads every ``data/frontier/*.jsonl`` and writes a copy with the same file name to
``data/frontier_prov/``.  Every copied row carries::

    source   = "frontier:<model>"
    model    = <slug taken from the row, else from the -t-<model> id suffix>
    provider = openai | anthropic | google | other
    run_id   = <journal/run label, only when the original row has one>

The originals are read-only: this script never writes to ``data/frontier``.  It
hashes every original file before and after the copy and exits loudly if a single
byte changed.  The hashes (and row counts) are recorded in
``data/frontier_prov/provenance_manifest.json`` so the check can be repeated later.

Run with:  python scripts/backfill_frontier_provenance.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.teacher.transfer import (  # noqa: E402
    _slug,
    model_from_id,
    provenance_fields,
)

FRONTIER_DIR = ROOT / "data" / "frontier"
OUT_DIR = ROOT / "data" / "frontier_prov"
MANIFEST_PATH = OUT_DIR / "provenance_manifest.json"
EXPECTED_TOTAL = 668


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_for_row(row: dict, name: str, lineno: int) -> str:
    """Prefer the row's own model label; otherwise decode the ``-t-`` id suffix."""
    explicit = row.get("model")
    if isinstance(explicit, str) and explicit.strip():
        return _slug(explicit)
    suffix = model_from_id(row.get("id"))
    if suffix:
        return suffix
    raise SystemExit(f"cannot determine model for {name}:{lineno} (id={row.get('id')!r})")


def check_previous_manifest(before: dict[str, str]) -> None:
    """An original must still match the hashes recorded by an earlier backfill."""
    if not MANIFEST_PATH.exists():
        return
    previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, info in previous.get("files", {}).items():
        expected = info.get("sha256") or info.get("sha256_after") or info.get("sha256_before")
        actual = before.get(name)
        if actual is None:
            raise SystemExit(f"ORIGINAL MISSING since last backfill: data/frontier/{name}")
        if expected and actual != expected:
            raise SystemExit(
                f"ORIGINAL CHANGED since last backfill: data/frontier/{name}\n"
                f"  manifest sha256 = {expected}\n  current  sha256 = {actual}"
            )


def main() -> int:
    paths = sorted(FRONTIER_DIR.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no frontier JSONL files under {FRONTIER_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    before = {path.name: sha256_file(path) for path in paths}
    check_previous_manifest(before)

    counts: dict[str, int] = {}
    histogram: Counter[str] = Counter()
    for path in paths:
        out_path = OUT_DIR / path.name
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        n = 0
        with path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
            for lineno, line in enumerate(src, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                model = model_for_row(row, path.name, lineno)
                provenance = provenance_fields(model, row.get("run_id"))
                enriched = dict(row)
                original_source = enriched.get("source")
                enriched.update(provenance)
                if original_source and original_source not in (
                    "frontier", f"frontier:{provenance['model']}"
                ):
                    enriched["source_detail"] = original_source
                dst.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                histogram[model] += 1
                n += 1
        tmp_path.replace(out_path)
        counts[path.name] = n
        print(f"{path.name}: {n} rows -> data/frontier_prov/{path.name}")

    after = {path.name: sha256_file(path) for path in paths}
    changed = [name for name, digest in before.items() if after[name] != digest]
    if changed:
        raise SystemExit(
            "ORIGINAL FRONTIER FILES CHANGED DURING BACKFILL (manifest not written):\n"
            + "\n".join(
                f"  data/frontier/{name}: {before[name]} -> {after[name]}" for name in changed
            )
        )

    total = sum(counts.values())
    manifest = {
        "algorithm": "sha256",
        "generated_by": "scripts/backfill_frontier_provenance.py",
        "source_dir": "data/frontier",
        "output_dir": "data/frontier_prov",
        "files": {
            name: {
                "sha256_before": before[name],
                "sha256_after": after[name],
                "sha256": after[name],
                "rows": counts[name],
            }
            for name in counts
        },
        "total_rows": total,
        "model_histogram": dict(sorted(histogram.items())),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print("model histogram:")
    for model, n in sorted(histogram.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {model}: {n}")
    print(f"total rows: {total} (expected {EXPECTED_TOTAL}"
          + ("" if total == EXPECTED_TOTAL else " -- MISMATCH") + ")")
    print(f"original SHA-256: unchanged for all {len(before)} files")
    print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
