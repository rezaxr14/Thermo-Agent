from __future__ import annotations
from ..core.schema import Site
from .spec import catalog_text

SYSTEM = (
    "You design on-site energy plants for AI data centres. Given a site, choose the plant configuration and set its "
    "parameters. The plant must solve, pass the physics checks (mass and energy balance, exergy closure, no negative "
    "exergy destruction, no temperature cross, plausible ranges) and meet the site: its IT electricity with no grid import, "
    "its chilled-water load, and any water limit. Among valid plants, the one that burns the least fuel wins. "
    "Answer with a short <rationale> that cites only quantities you expect the solved plant to show (MW, K), then a "
    "<design> block with one `key = value` per line and exactly the keys your choices require. "
    f"Catalog: {catalog_text()}. PR and T4 apply only to a turbine; T_hw and dT_hw only to absorption cooling; "
    "P_orc_bar and orc_frac only when bottoming is an ORC (orc_frac 0.1 = smallest working-fluid flow, 1.0 = largest). "
    "If the site has a heat demand, also set heat_source = exhaust or heat_pump."
)


def describe_site(site: Site) -> str:
    a = site.ambient
    parts = [f"it_load_w={site.it_load_w!r}", f"cooling_load_w={site.cooling_load_w!r}",
             f"ambient_T_K={a.T0!r}", f"ambient_RH={a.RH!r}",
             f"coolant_supply_K={site.t_supply_k!r}", f"coolant_return_K={site.t_return_k!r}",
             f"grid_import_allowed={str(site.grid_import_allowed).lower()}"]
    if site.heat_demand_w > 0: parts.append(f"heat_demand_w={site.heat_demand_w!r}")
    if site.max_water_kg_s is not None: parts.append(f"max_water_kg_s={site.max_water_kg_s!r}")
    return " ".join(parts)


def design_messages(site: Site) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"SITE\n{describe_site(site)}"}]


def design_feedback_valid(fuel_w: float, best_w: float) -> dict:
    return {"role": "user", "content": f"Your design is valid: fuel input {fuel_w/1e6:.3f} MW (best so far {best_w/1e6:.3f} MW). "
            "Propose a design that burns less fuel, or repeat your best design if you cannot improve it. Same format."}


def design_feedback(reason: str) -> dict:
    return {"role": "user", "content": f"Your design was rejected or invalid: {reason}\nReply again in the same format."}
