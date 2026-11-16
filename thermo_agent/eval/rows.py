"""Read one task/split from one or more ground-truth directories.

Evaluation must see the architecture diagrams (`data/processed_arch`) as well as the five-plant mix
(`data/processed`): with ORC, engines and turbines training, the held-out `test_ood_topology` rows exist only in the
architecture corpus. Rows are de-duplicated by `id`, because a plant that appears in both directories must be counted
once.

    from .rows import load_split
    rows = load_split(["data/processed", "data/processed_arch"], "audit", "test_ood_topology")
"""
from __future__ import annotations

import json
import pathlib


def load_split(data_dirs, task: str, split: str) -> list[dict]:
    """Every row of `<task>_<split>.jsonl` under one or more directories, in directory order, ids unique."""
    dirs = [data_dirs] if isinstance(data_dirs, (str, pathlib.Path)) else list(data_dirs)
    rows: list[dict] = []
    seen: set = set()
    for d in dirs:
        path = pathlib.Path(d) / f"{task}_{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in seen:
                continue
            seen.add(row.get("id"))
            rows.append(row)
    return rows
