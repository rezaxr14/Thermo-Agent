"""Audit per-row provenance and tiers of any training corpus.

Reza's rule: every row must say where it came from -- deterministic ground truth
or an AI-generated trace named by its exact model -- and must carry a real
`tier` (equal thirds per task, see `thermo_agent/synth/tiers.py`).

    python scripts/audit_dataset_provenance.py data/sft_mix_v3 data/grpo_mix_v3

Read-only: never writes to the audited directories. Exit code 1 on any missing
field, an unknown tier, or an AI row without a model id.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.synth.tiers import TIERS  # noqa: E402

SOURCES = {"deterministic_tools", "frontier_tools", "ground_truth"}
AI_SOURCES = {"frontier_tools"}


def audit_dirs(dirs: list[str], root: pathlib.Path | None = None) -> dict:
    """Return a report for all JSONL rows under the given directories.

    ``root`` defaults to the repository root; tests pass a temporary root.
    """
    if root is None:
        root = ROOT
    by_source: collections.Counter = collections.Counter()
    by_model: collections.Counter = collections.Counter()
    by_task_tier: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    problems: list[dict] = []
    rows_seen = 0
    files: list[str] = []
    for d in dirs:
        directory = pathlib.Path(d)
        if not directory.is_absolute():
            directory = root / directory
        for path in sorted(directory.rglob("*.jsonl")):
            try:
                files.append(str(path.relative_to(root)))
            except ValueError:
                files.append(str(path))
            for lineno, line in enumerate(path.open(encoding="utf-8"), 1):
                if not line.strip():
                    continue
                rows_seen += 1
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    problems.append({"file": str(path), "line": lineno, "error": f"bad json: {exc}"})
                    continue
                where = f"{path}:{lineno}"
                source = row.get("source")
                if source not in SOURCES:
                    problems.append({"file": where, "error": f"unknown or missing source {source!r}"})
                else:
                    by_source[source] += 1
                prov = row.get("provenance") or {}
                generator = prov.get("generator") if isinstance(prov, dict) else None
                if isinstance(prov, dict) and not generator and source != "ground_truth":
                    problems.append({"file": where, "error": "provenance.generator missing"})
                if source in AI_SOURCES:
                    model = prov.get("model") or row.get("model")
                    if not model:
                        problems.append({"file": where, "error": "AI row without model id"})
                    else:
                        by_model[str(model)] += 1
                else:
                    by_model["none (deterministic)"] += 1
                tier = row.get("tier")
                task = str(row.get("task") or "<missing>")
                if tier not in TIERS:
                    problems.append({"file": where, "error": f"tier {tier!r} not in {TIERS}"})
                else:
                    by_task_tier[task][tier] += 1
    return {
        "dirs": list(dirs), "files": files, "rows": rows_seen,
        "by_source": dict(by_source),
        "by_model": dict(by_model),
        "by_task_tier": {task: dict(counts) for task, counts in sorted(by_task_tier.items())},
        "problems": problems[:50], "problem_count": len(problems),
        "passed": not problems and rows_seen > 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="directories of JSONL corpora to audit")
    args = ap.parse_args(argv)
    report = audit_dirs(args.dirs)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
