from __future__ import annotations
from ..core.schema import Plant, Site, SolveResult
from ..agent.dsl import plant_to_text
from ..train.prompts import describe_site
from .truth import summary_text

SYSTEM = (
    "You audit solved energy plants for AI data centres. You are given the plant, the site, and an energy-level summary "
    "(exergy destruction is not shown). Answer with a short <analysis> citing only numbers (MW, K) the plant would show, "
    "then <ranking> listing up to 5 component ids by exergy destruction, largest first, written `A > B > C` (the combustor or gas engine is always the largest and is not scored; the order of the rest is); <change> with "
    "one line `SET <component>.<param> = <number>` that lowers fuel input for the same services (a set-point or loss "
    "parameter within about 20 percent of its current value; component efficiencies cannot be changed; the loop temperatures "
    "of the hot-water and heat-rejection loops cannot be changed); and <prediction> with `fuel_change_mw = <number>`, "
    "the change in fuel input in MW after the prime mover is resized to deliver the same net electricity (negative = saves fuel)."
)


def audit_messages(plant: Plant, site: Site | None, res: SolveResult) -> list[dict]:
    user = f"PLANT\n{plant_to_text(plant)}\n\nSITE\n{describe_site(site)}\n\nSUMMARY\n{summary_text(plant, res)}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
