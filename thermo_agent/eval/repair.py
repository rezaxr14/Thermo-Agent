"""Score a proposed repair patch: apply, re-solve, verify. Success = all hard checks pass; exact = restores the original plant."""
from __future__ import annotations
from pydantic import ValidationError
from ..core.schema import Plant
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..synth.faults import apply_patch, _dump


def _close(a, b, rel=1e-9):
    if isinstance(a, dict): return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    return a == b


def score_patch(broken: Plant, original: Plant, patch: list[dict], site=None) -> dict:
    try:
        fixed = Plant.model_validate(apply_patch(_dump(broken), patch))
    except (ValidationError, KeyError, IndexError, ValueError, TypeError) as e:
        return {"applied": False, "success": False, "exact": False, "reason": f"patch/schema: {type(e).__name__}"}
    rep = run_all(fixed, solve(fixed), site)
    return {"applied": True, "success": rep.valid, "exact": _close(_dump(fixed), _dump(original)),
            "reward": rep.reward, "failed": rep.failed()}
