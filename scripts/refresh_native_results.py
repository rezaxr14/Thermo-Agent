"""Re-serve stored native tool results from the current toolkit.

The teacher's answer, ids, prompts and final XML are never changed. This only
rewrites environment output: skill text (``list_skills``/``load_skill``) and the
compact table shape of ``solve_plant`` / ledger residuals, so a stored trace
teaches exactly what the same tool call returns today. Legacy rows are rejected,
never migrated.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.agent import tool_protocol  # noqa: E402


def refresh_file(path: pathlib.Path, out_dir: pathlib.Path, dry_run: bool) -> dict:
    rows, changed, rejected = [], 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fresh = tool_protocol.normalize_train_row(row)
        if fresh is None:
            rejected += 1
            continue
        if fresh != row:
            changed += 1
        rows.append(fresh)
    if not dry_run and rows:
        target = out_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
    return {"file": str(path.resolve().relative_to(ROOT)), "rows": len(rows),
            "changed": changed, "rejected_legacy": rejected}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default="reports/native_results_refresh.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    report = []
    for directory in map(pathlib.Path, args.dirs):
        for path in sorted(directory.glob("frontier_*_train.jsonl")):
            report.append(refresh_file(path, directory, args.dry_run))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    if any(r["rejected_legacy"] for r in report):
        print("legacy rows present; archive them instead of migrating", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
