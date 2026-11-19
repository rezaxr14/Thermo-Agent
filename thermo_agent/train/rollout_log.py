"""Record GRPO rollouts (prompt, completion, per-channel reward) for audit and paper figures.

TRL's `log_completions=True` is useful interactively, but `report_to="none"` keeps the samples out of any saved
file. Wrapping the reward functions preserves their names (and therefore `reward_weights`) while writing one JSON
line per rollout group. Only rank 0 writes, so DDP does not duplicate the stream.

    from .rollout_log import wrap_rewards
    funcs = wrap_rewards(TASKS[task][0], path)
"""
from __future__ import annotations

import functools
import json
import pathlib
import time

from . import ddp
from . import rewards as repair_rewards


def _completion_metrics(completions) -> tuple[float, float]:
    """Mean assistant turns and tool calls over a batch of rollout completions.

    TRL stores a multi-turn rollout as a list of messages (assistant turns interleaved with tool
    results); legacy single-turn completions are plain strings. No reward is touched here.
    """
    turns, calls, n = 0.0, 0.0, 0
    for c in completions or []:
        n += 1
        if not isinstance(c, (list, tuple)):
            # A legacy text completion is one assistant answer with no tool calls.
            turns += 1
            continue
        for message in c:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role in (None, "assistant"):
                turns += 1
                if message.get("tool_calls"):
                    calls += len(message["tool_calls"])
                elif repair_rewards.is_tool_call_content(message.get("content")):
                    calls += 1
            elif role == "tool":
                calls += 1
    return turns / max(n, 1), calls / max(n, 1)


def _zero_variance_share(totals: list[float], k: int) -> tuple[float, int]:
    """Share of consecutive k-sized reward groups whose members all scored the same."""
    groups = [totals[i:i + k] for i in range(0, len(totals), k)]
    complete = [g for g in groups if len(g) == k]
    zero = sum(1 for g in complete if max(g) - min(g) < 1e-9)
    return zero / max(len(complete), 1), len(complete)


class RolloutBatchLogger:
    """Append ONE JSON line per rollout batch to an audit JSONL file (additive, no reward change).

    TRL invokes every reward function on the same batch, in order. Wrapping the list collects each
    channel's values; after the last function returns, one record is written with the weighted
    reward, zero-variance-group share, truncation rate, mean turns and mean tool calls.
    """

    def __init__(self, weights, k: int, path, stage: str = "grpo") -> None:
        self.weights = [float(w) for w in weights]
        self.k = int(k)
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self._batch = 0
        self._values: dict[int, list] = {}

    def wrap(self, funcs):
        n = len(funcs)
        if not n:
            return list(funcs)

        def make(fn, index):
            @functools.wraps(fn)  # reward_weights are mapped by __name__
            def inner(*args, **kwargs):
                values = fn(*args, **kwargs)
                self._values[index] = list(values)
                if len(self._values) == n:
                    self._publish(kwargs)
                    self._values = {}
                return values
            return inner

        return [make(fn, i) for i, fn in enumerate(funcs)]

    def _publish(self, kwargs) -> None:
        if ddp.local_rank_from_env() != 0:
            return
        cols = [self._values[i] for i in range(len(self.weights))]
        n = len(cols[0]) if cols else 0
        totals: list[float] = []
        for j in range(n):
            total = 0.0
            for weight, col in zip(self.weights, cols):
                value = col[j]
                total += weight * (value if value is not None else 0.0)
            totals.append(total)
        share, groups = _zero_variance_share(totals, self.k)
        completions = kwargs.get("completions") or []
        turns, tool_calls = _completion_metrics(completions)
        truncation = sum(1 for c in completions if repair_rewards.truncation_reason(c) is not None)
        self._batch += 1
        record = {
            "stage": self.stage,
            "batch": self._batch,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reward": sum(totals) / max(n, 1),
            "reward_values": totals,
            "zero_variance_share": share,
            "groups": groups,
            "truncation_rate": truncation / max(n, 1),
            "turns": turns,
            "tool_calls": tool_calls,
            "n": n,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def wrap_rewards(funcs, path=None, stage="grpo"):
    """Return the same reward callables, each call appending its scored rollout to `path`."""
    if not path:
        return list(funcs)
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def wrap(fn):
        @functools.wraps(fn)                      # keep __name__: TRL maps reward_weights by function name
        def inner(completions=None, **kwargs):
            values = fn(completions, **kwargs) if completions is not None else fn(**kwargs)
            counter["n"] += 1
            if ddp.local_rank_from_env() == 0:
                prompts = kwargs.get("prompts") or kwargs.get("prompt") or []
                record = {
                    "stage": stage, "group": counter["n"],
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reward_fn": fn.__name__,
                    "prompt": str(prompts[0])[:2000] if prompts else "",
                    "completions": [str(c)[:2000] for c in (completions or [])],
                    "rewards": [float(v) if v is not None else None for v in values],
                }
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            return values
        return inner

    return [wrap(f) for f in funcs]
