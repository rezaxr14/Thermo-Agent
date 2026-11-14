"""Model-in-the-loop repair: the model proposes edits, the solver and verifiers answer, the model may try again."""
from __future__ import annotations
from typing import Callable
from ..train.rewards import evaluate, completion_text
from ..train.prompts import feedback_message, describe_failure
from ..agent.dsl import parse_edits, edits_to_patch
from ..core.schema import Plant, Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from ..synth.faults import apply_patch, _dump


def repair_loop(generate: Callable[[list[dict]], str], row: dict, max_turns: int = 3) -> dict:
    """generate(messages) -> completion text. Returns success, turns used, solver calls."""
    messages = list(row["prompt"]); solves = 0; broken = Plant.model_validate_json(row["plant_broken"])
    site = Site.model_validate_json(row["site"]) if row["site"] else None
    current = broken
    for turn in range(1, max_turns + 1):
        text = generate(messages)
        ev = evaluate(text, row["plant_broken"], row["plant_original"], row["site"], bool(row["deducible"]))
        if ev["applied"]: solves += 1
        if ev["physics_ok"] and ev["site_ok"] is not False:
            return {"success": True, "exact": ev["exact"], "turns": turn, "solves": solves}
        messages += [{"role": "assistant", "content": text}]
        if ev["applied"]:
            fixed = Plant.model_validate(apply_patch(_dump(current), edits_to_patch(current, parse_edits(text))))
            res = solve(fixed); messages.append(feedback_message(res, run_all(fixed, res, site)))
        else:
            messages.append({"role": "user", "content": "Your answer could not be parsed or applied. Use exactly <diagnosis>...</diagnosis> then <edits>...</edits> with valid SET lines."})
    return {"success": False, "exact": False, "turns": max_turns, "solves": solves}
