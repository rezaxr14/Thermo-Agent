"""Resume support for the two training stages.

An epoch of the 20-variant corpus is about 37 hours on two T4s, and a Kaggle session ends after 12. The trainer
therefore has to save often enough that the next session can continue, and it has to find that checkpoint on its own
so a restarted session needs no hand-written path.

    from .checkpoint import latest
    trainer.train(resume_from_checkpoint=latest(out_dir))   # None -> fresh run
"""
from __future__ import annotations

import pathlib
import re

# Exactly the names the Transformers Trainer writes: checkpoint-<global step>. Nothing else is a checkpoint.
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def latest(out_dir: str | pathlib.Path) -> str | None:
    """Path of the highest-step trainer checkpoint in `out_dir`, or None when the run has not saved one yet.

    A missing directory is a fresh run, not an error: the first session starts before anything exists. Ties cannot
    happen because the step number is the directory name; a stray file named like a checkpoint is ignored.
    """
    path = pathlib.Path(out_dir)
    if not path.is_dir():
        return None
    best: tuple[int, pathlib.Path] | None = None
    for child in path.iterdir():
        match = _CHECKPOINT.match(child.name)
        if not match or not child.is_dir():
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, child)
    return str(best[1]) if best else None
