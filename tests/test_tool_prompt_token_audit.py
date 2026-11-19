"""J2/J3 token budget: p50/max harness-block size before vs after compaction.

Measured with the cached real Qwen3.5 tokenizer on real rows from ``data/processed``.
The compact block is the text every arm is served (tracegen, ``*_h`` training,
eval/GRPO); the verbose block is the historical full AGENTS.md + component skills
+ toolkit docstrings, kept callable only for comparison. The hard gate is that the
compact block stays <= 2,500 tokens for every sampled row.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from thermo_agent.agent.tool_prompt import build_harness_tool_messages
from thermo_agent.core.schema import Plant
from thermo_agent.synth.combine import tokenizer

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("repair", "design", "audit", "ledger")
CEILING = 2500
N_ROWS = 8  # a few real rows per task; enough for a stable p50/max without a long test


def _rows(task: str) -> list[dict]:
    path = ROOT / "data" / "processed" / f"{task}_train.jsonl"
    if not path.is_file():
        pytest.skip(f"no real training rows at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:N_ROWS] if line.strip()]


def _plant(row: dict) -> Plant | None:
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                return Plant.model_validate_json(row[key])
            except Exception:
                pass
    return None


def _tok():
    tok = tokenizer()
    if tok is None:
        pytest.skip("no local Qwen3.5 tokenizer: set THERMO_TOKENIZER or cache "
                    "~/.cache/thermo/qwen3.5-4b/tokenizer.json")
    return tok


def _ntok(tok, text: str) -> int:
    return len(tok.encode(text).ids)


def _block_sizes(tok, task: str, row: dict) -> tuple[int, int]:
    plant = _plant(row)
    base = row["prompt"][0]["content"]
    sizes = []
    for compact in (True, False):
        sysmsg = build_harness_tool_messages(list(row["prompt"]), task, plant,
                                             compact=compact)[0]["content"]
        block = sysmsg[len(base):] if sysmsg.startswith(base) else sysmsg
        sizes.append(_ntok(tok, block))
    return sizes[0], sizes[1]  # compact, verbose


def test_harness_block_p50_max_before_after_and_ceiling():
    tok = _tok()
    compact, verbose, by_task = [], [], {}
    for task in TASKS:
        c_task, v_task = [], []
        for row in _rows(task):
            c, v = _block_sizes(tok, task, row)
            compact.append(c); verbose.append(v)
            c_task.append(c); v_task.append(v)
        by_task[task] = (c_task, v_task)

    assert compact and verbose
    n = len(compact)
    p50c, maxc = int(statistics.median(compact)), max(compact)
    p50v, maxv = int(statistics.median(verbose)), max(verbose)

    print(f"\nHarness block tokens, cached Qwen3.5 tokenizer ({n} real rows):")
    print(f"  {'task':8s} {'n':>3s} {'compact p50/max':>17s} {'verbose p50/max':>17s} {'saved p50':>10s}")
    for task in TASKS:
        c_task, v_task = by_task[task]
        print(f"  {task:8s} {len(c_task):3d} "
              f"{int(statistics.median(c_task)):6d}/{max(c_task):<10d} "
              f"{int(statistics.median(v_task)):6d}/{max(v_task):<10d} "
              f"{int(statistics.median(v_task)) - int(statistics.median(c_task)):10d}")
    print(f"  {'ALL':8s} {n:3d} {p50c:6d}/{maxc:<10d} {p50v:6d}/{maxv:<10d} {p50v - p50c:10d}")

    # The J2/J3 hard gate: every served block is <= 2,500 real tokens.
    assert maxc <= CEILING, f"compact harness max {maxc} exceeds {CEILING} tokens"
    # Compaction must be a real reduction, not a relabel.
    assert p50c < p50v and maxc < maxv, (p50c, p50v, maxc, maxv)
