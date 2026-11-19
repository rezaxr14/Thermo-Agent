#!/usr/bin/env python3
"""Create gated copies of the harness corpora by removing over-8192 rows only.

Reads ``data/provenance/harness_corpus_audit_2026-11-19.json`` (or ``--audit``)
to learn which row ids exceed the 8192-token gate.  It NEVER modifies the source
corpora: in a real run it writes filtered copies into NEW directories
(``data/sft_tools_h_gated``, ``data/grpo_tools_h_gated``).

This script is designed to be run in ``--dry-run`` mode first; the dry run only
counts how many rows would be kept/dropped and creates nothing.

    .venv/Scripts/python.exe scripts/drop_over_gate.py --dry-run

Measures
--------
The audit records two over-gate lists because the chat-template render drops
inline tool blocks (under-counting) while the ``pipeline`` measure is the repo's
authoritative 8192 gate (``token_audit.py``).  ``--measure`` selects which ids
are dropped:

* ``union`` (default) -- drop any row over 8192 under EITHER measure (safest);
* ``rendered`` -- only the chat-template-rendered over-8192 ids;
* ``pipeline`` -- only the repo-gate over-8192 ids.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = "data/provenance/harness_corpus_audit_2026-11-19.json"
SOURCE_DIRS = ("data/sft_tools_h", "data/grpo_tools_h")
TARGET_DIRS = ("data/sft_tools_h_gated", "data/grpo_tools_h_gated")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", default=DEFAULT_AUDIT)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--dry-run", action="store_true",
                    help="only count kept/dropped rows; create no directories or files")
    ap.add_argument("--measure", choices=("union", "rendered", "pipeline"), default="union",
                    help="which over-8192 ids to exclude (default: union)")
    ap.add_argument("--out-sft", default=TARGET_DIRS[0])
    ap.add_argument("--out-grpo", default=TARGET_DIRS[1])
    args = ap.parse_args(argv)

    root = args.root.resolve()
    audit_path = root / args.audit
    data = json.loads(audit_path.read_text(encoding="utf-8"))

    def ids(key: str) -> set[str]:
        return {str(r["id"]) for r in data.get(key, [])}

    r_ids, p_ids = ids("over_8192_rendered"), ids("over_8192_pipeline")
    if args.measure == "union":
        drop = r_ids | p_ids
    elif args.measure == "rendered":
        drop = r_ids
    else:
        drop = p_ids

    print(f"audit: {audit_path.relative_to(root)}")
    print(f"measure: {args.measure}  (rendered={len(r_ids)}, pipeline={len(p_ids)}, "
          f"selected drop ids={len(drop)})")
    print(f"mode: {'DRY RUN (no files written)' if args.dry_run else 'REAL RUN'}")
    print()

    out_for = dict(zip(SOURCE_DIRS, (args.out_sft, args.out_grpo)))
    summary: dict[str, dict[str, int]] = {}
    for src in SOURCE_DIRS:
        src_dir = root / src
        dst_dir = root / out_for[src]
        src_files = sorted(p for p in src_dir.glob("*.jsonl")
                           if p.is_file() and not p.is_symlink())
        kept = dropped = 0
        print(f"{src} -> {out_for[src]}")
        for src_file in src_files:
            f_kept = f_dropped = 0
            out_lines: list[bytes] = []
            with src_file.open("rb") as fh:
                for raw in fh:
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw.decode("utf-8", errors="replace"))
                    except (ValueError, UnicodeDecodeError):
                        out_lines.append(raw)  # keep malformed lines verbatim
                        f_kept += 1
                        continue
                    if isinstance(row, dict) and str(row.get("id")) in drop:
                        f_dropped += 1
                        continue
                    out_lines.append(raw)
                    f_kept += 1
            kept += f_kept
            dropped += f_dropped
            print(f"  {src_file.name:<22} kept={f_kept:<4} dropped={f_dropped}")
            if not args.dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                (dst_dir / src_file.name).write_bytes(b"".join(out_lines))
        # copy non-jsonl siblings verbatim (e.g. MANIFEST.json) in a real run
        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for extra in src_dir.iterdir():
                if extra.is_file() and extra.suffix.lower() != ".jsonl" and not extra.is_symlink():
                    shutil.copy2(extra, dst_dir / extra.name)
        summary[src] = {"kept": kept, "dropped": dropped,
                        "files": len(src_files)}
        print(f"  TOTAL kept={kept} dropped={dropped} "
              f"(source rows={kept + dropped})\n")

    tot_kept = sum(v["kept"] for v in summary.values())
    tot_drop = sum(v["dropped"] for v in summary.values())
    print(f"GRAND TOTAL: source rows={tot_kept + tot_drop}  "
          f"would keep={tot_kept}  would drop={tot_drop}")
    if args.dry_run:
        print("Dry run complete: no gated directories were created.")
    else:
        print("Real run complete: gated corpora written to NEW directories "
              "(source corpora untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
