"""Prompt construction for the Repair task."""
from __future__ import annotations
from ..core.schema import Plant, Site, SolveResult
from ..agent.dsl import plant_to_text
from ..verify.verifiers import Report

SYSTEM = (
    "You repair energy-plant models. Units are SI (K, Pa, W, kg/s). Efficiencies, second-law fractions and "
    "pressure-drop fractions are fractions, never percent. A plant is valid when it solves, passes the physics "
    "checks (mass and energy balance, exergy closure, non-negative exergy destruction, no temperature cross, "
    "plausible ranges) and meets the site requirements. Answer with a short <diagnosis> that cites only numbers "
    "shown in the input, then <edits> with at most 6 lines of the form `SET <component>.<param> = <number>`, "
    "`SET @<stream>.<field> = <number>`, `REWIRE @<stream>.<to|from> = <component>.<port>` or `FLUID @<stream> = <fluid>`. "
    "You can only change numbers, endpoints and fluids that already exist. Change as little as possible."
)


def describe_failure(res: SolveResult, rep: Report) -> str:
    if not res.ok:
        return f"SOLVER ERROR {res.error.value}: {res.message}"
    lines = [f"FAILED {c.name}: {c.detail}" for c in rep.checks if not c.ok]
    return "\n".join(lines) if lines else "ALL CHECKS PASS"


def describe_site(site: Site | None) -> str:
    if site is None: return "none"
    parts = [f"it_load_w={site.it_load_w!r}"]
    if site.cooling_load_w: parts.append(f"cooling_load_w={site.cooling_load_w!r}")
    if site.heat_demand_w: parts.append(f"heat_demand_w={site.heat_demand_w!r}")
    parts.append(f"grid_import_allowed={str(site.grid_import_allowed).lower()}")
    return " ".join(parts)


def repair_messages(plant: Plant, site: Site | None, res: SolveResult, rep: Report) -> list[dict]:
    user = f"PLANT\n{plant_to_text(plant)}\n\nSITE\n{describe_site(site)}\n\nRESULT\n{describe_failure(res, rep)}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def feedback_message(res: SolveResult, rep: Report) -> dict:
    return {"role": "user", "content": f"Your edits were applied. RESULT\n{describe_failure(res, rep)}\nReply again in the same format."}
