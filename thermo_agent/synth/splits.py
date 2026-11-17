"""Which diagram combinations are held out whole for `test_ood_topology`.

The corpus trains on diagram *families*: turbine or gas-engine prime mover, absorption or electric cooling, ideal
(`ext`) / dry / wet heat rejection, ORC bottoming on the remaining exhaust (R245fa, cyclopentane, toluene) and
district heat off the exhaust or a heat pump. Holding out a whole family would mean the model never trains on ORC,
co-generation or Rankine at all, which is the opposite of what the corpus is for.

So the held-out unit is a *combination*: the gas turbine with air preheater, HRSG steam export and ORC
(`cgam_steam_orc`, the Rankine plant) and the gas engine with the ideal rejection boundary (`engine-*-ext*`). Both
add a component or a boundary the training diagrams never show together, so `test_ood_topology` still measures
generalisation to an unseen diagram without hiding ORC or engines from training.

    from thermo_agent.synth.splits import is_ood_topology
    is_ood_topology("engine-absorption-ext-orc")   # True
    is_ood_topology("dc_engine_absorption")        # False: an engine, but with absorption cooling
"""
from __future__ import annotations

STEAM_EXPORT = "cgam_steam_orc"
HELD_OUT_PARTS = ("engine", "ext")


def is_ood_topology(name: str | None) -> bool:
    """True for the diagram combinations reserved for `test_ood_topology`.

    Names arrive in both spellings the repo uses: `dc_engine_absorption` from the base generators and
    `engine-absorption-ext` from the architecture catalog. For a MIX plant whose name omits the rejection
    boundary, call `topology_label(plant)` first.
    """
    n = (name or "").replace("_sampled", "").replace("_", "-").lower()
    if n == STEAM_EXPORT.replace("_", "-"):
        return True
    parts = n.split("-")
    return all(part in parts for part in HELD_OUT_PARTS)


def topology_label(plant) -> str:
    """Canonical diagram label of a plant, including the rejection boundary the MIX names leave out.

    `_sample_dc` names every engine plant `dc_engine_{cooling}_sampled` whether it rejects through an ideal
    boundary (`ext`) or a dry/wet cooling tower. Using that name alone put ideal-rejection engines in the
    training split while the catalog's `engine-*-ext*` diagrams were held out -- a leak of the held-out
    topology. Read the graph instead: the ideal boundary sends the chiller rejection stream straight to
    `ext:cw` with no heat-rejector component. The plant's prompt is unchanged; only this metadata label and
    the split derived from it change.
    """
    name = (getattr(plant, "name", "") or "").replace("_sampled", "")
    if is_ood_topology(name):
        return name
    ideal_chiller_rejection = any(str(s.src).startswith("ext:") and str(s.dst).startswith("CH.rej_in")
                                  for s in plant.streams)
    engine = any(c.type == "gas_engine" for c in plant.components)
    return f"{name}-ext" if engine and ideal_chiller_rejection else name
