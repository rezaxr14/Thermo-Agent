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
