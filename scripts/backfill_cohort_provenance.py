"""Add the machine-readable `provenance` dict to accepted cohort rows (idempotent).

`transfer.ingest` already writes `source="frontier:<model>"`, `model`, `provider` and
`run_id`. Earlier ingests ran before the provenance contract existed, so those rows
lack `provenance.model` and the training mix / progress report flag them. This
backfills only what is missing and never rewrites an existing value.

    python scripts/backfill_cohort_provenance.py --dry-run
    python scripts/backfill_cohort_provenance.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
COHORT = ROOT / "data" / "frontier_tools_cohort"
PROVIDERS = {"gpt-6-luna": "openai", "claude-haiku-5-5": "anthropic"}


def provenance_for(row: dict, model: str) -> dict:
    existing = row.get("provenance")
    prov = dict(existing) if isinstance(existing, dict) else {}
    prov.setdefault("source_family", "ai_generated")
    prov.setdefault("generator", "frontier_teacher")
    prov.setdefault("model", str(row.get("model") or model))
    prov.setdefault("provider", str(row.get("provider") or PROVIDERS.get(model, "other")))
    run_id = row.get("run_id")
    if run_id and "run_id" not in prov:
        prov["run_id"] = str(run_id)
    return prov


def backfill(path: pathlib.Path, model: str, dry_run: bool = False) -> tuple[int, int]:
    rows = []
    changed = 0
    total = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        before = json.dumps(row.get("provenance"), sort_keys=True)
        row["provenance"] = provenance_for(row, model)
        if json.dumps(row["provenance"], sort_keys=True) != before:
            changed += 1
        rows.append(row)
    if changed and not dry_run:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
    return total, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dir", default=str(COHORT))
    args = parser.parse_args(argv)

    base = pathlib.Path(args.dir)
    grand_total = grand_changed = 0
    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        model = model_dir.name
        for path in sorted(model_dir.glob("*.jsonl")):
            total, changed = backfill(path, model, args.dry_run)
            grand_total += total
            grand_changed += changed
            print(f"{model}/{path.name}: rows={total} patched={changed}"
                  + (" (dry run)" if args.dry_run else ""))
    print(f"TOTAL rows={grand_total} patched={grand_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
