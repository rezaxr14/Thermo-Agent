"""Model-in-the-loop design, optionally followed by an optimizer polish (the LLM + optimizer hybrid)."""
from __future__ import annotations
from typing import Callable
from ..core.schema import Site
from .rewards import evaluate as eval_text, completion_text
from .prompts import design_feedback
from .spec import parse_design
from .search import polish


def design_refine(generate: Callable[[list[dict]], str], row: dict, max_turns: int = 6, patience: int = 2) -> dict:
    """Anytime protocol: after every valid design the model is told its fuel input and may try to improve it.
    `anytime` lists (design evaluations so far, best gap to the reference or None) after each attempt."""
    from .prompts import design_feedback_valid
    messages = list(row["prompt"]); ref = float(row["ref_fuel_w"]); best = None; evals = 0; stale = 0; anytime = []
    for turn in range(1, max_turns + 1):
        text = generate(messages); e = eval_text(text, row["site"])
        if e["built"]: evals += 1
        if e["valid"]:
            f = e["fuel_w"]; improved = best is None or f < best * (1 - 1e-4); best = f if best is None else min(best, f)
            stale = 0 if improved else stale + 1; anytime.append((evals, best / ref - 1))
            if stale >= patience: break
            messages += [{"role": "assistant", "content": text}, design_feedback_valid(f, best)]
        else:
            anytime.append((evals, None if best is None else best / ref - 1))
            messages += [{"role": "assistant", "content": text}, design_feedback(e["reason"] or "invalid")]
    return {"valid": best is not None, "gap": None if best is None else best / ref - 1, "turns": len(anytime), "evals": evals, "anytime": anytime}


def design_loop(generate: Callable[[list[dict]], str], row: dict, max_turns: int = 3, polish_evals: int = 0) -> dict:
    """Returns valid, gap to the reference fuel, turns and design evaluations spent (1 per model attempt that builds)."""
    messages = list(row["prompt"]); evals = 0; ref = float(row["ref_fuel_w"]); site = Site.model_validate_json(row["site"])
    for turn in range(1, max_turns + 1):
        text = generate(messages); e = eval_text(text, row["site"])
        if e["built"]: evals += 1
        if e["valid"]:
            fuel = e["fuel_w"]
            if polish_evals:
                spec, f2, n = polish(site, parse_design(text, needs_heat=site.heat_demand_w > 0), polish_evals); evals += n
                if f2 is not None: fuel = min(fuel, f2)
            return {"valid": True, "gap": fuel / ref - 1, "turns": turn, "evals": evals}
        messages += [{"role": "assistant", "content": text}, design_feedback(e["reason"] or "invalid")]
    return {"valid": False, "gap": None, "turns": max_turns, "evals": evals}
