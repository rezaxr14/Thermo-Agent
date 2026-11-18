"""Export exactly the rows the evaluator samples, so saved frontier answers can be replayed by id.

    python scripts/make_eval_export.py --out data/frontier_eval_export --limit 30

`model_eval` reads ground-truth rows in file order and takes the first `--limit` of each split. A teacher export
built from the SFT corpus selects different ids, so replay cannot find them. This script uses the same
`eval.rows.load_split` ordering and keeps only the prompt fields; no ground-truth completion is written.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from thermo_agent.eval.model_eval import DEFAULT_SPLITS
from thermo_agent.eval.rows import load_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/frontier_eval_export")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--dirs", nargs="+", default=["data/processed", "data/processed_arch"])
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    total = 0
    for task, splits in DEFAULT_SPLITS.items():
        for split in splits:
            rows = load_split(a.dirs, task, split)[:a.limit]
            path = out / f"{task}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    clean = {k: v for k, v in r.items() if k != "completion"}
                    clean["task"] = task
                    fh.write(json.dumps(clean) + "\n")
            total += len(rows)
            print(task, split, len(rows))
    (out / "export_manifest.json").write_text(
        json.dumps({"source_dirs": a.dirs, "limit": a.limit, "rows": total}, indent=1) + "\n", encoding="utf-8")
    print("total", total)


if __name__ == "__main__":
    main()
