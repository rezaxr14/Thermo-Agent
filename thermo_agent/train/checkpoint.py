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


def is_complete(path: pathlib.Path) -> bool:
    """Whether a checkpoint has the state a true resume needs.

    The Trainer writes ``trainer_state.json`` (RNG + step counters), ``optimizer.pt``
    (optimizer + scaler) and adapter/model weights. A session killed mid-save can
    leave a directory with only some of them; resuming there would silently drop
    optimizer and RNG state while still looking like a resume.
    """
    if not (path / "trainer_state.json").is_file():
        return False
    optimizers = ("optimizer.pt", "optimizer.bin")
    if not any((path / name).is_file() for name in optimizers):
        return False
    weights = ("adapter_model.safetensors", "adapter_model.bin",
               "model.safetensors", "pytorch_model.bin")
    return any((path / name).is_file() for name in weights)


def latest(out_dir: str | pathlib.Path, *, complete: bool = False) -> str | None:
    """Path of the highest-step trainer checkpoint in `out_dir`, or None for a fresh run.

    ``complete=True`` (used by both training stages) skips checkpoints that lack
    optimizer/RNG/weight files. The default keeps the historical behaviour for
    callers that just enumerate checkpoint directories. A missing directory is a
    fresh run, not an error; a stray file named like a checkpoint is ignored.
    """
    path = pathlib.Path(out_dir)
    if not path.is_dir():
        return None
    best: tuple[int, pathlib.Path] | None = None
    for child in path.iterdir():
        match = _CHECKPOINT.match(child.name)
        if not match or not child.is_dir():
            continue
        if complete and not is_complete(child):
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, child)
    return str(best[1]) if best else None
