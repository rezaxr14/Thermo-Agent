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
    `engine-absorption-ext` from the architecture catalog.
    """
    n = (name or "").replace("_sampled", "").replace("_", "-").lower()
    if n == STEAM_EXPORT.replace("_", "-"):
        return True
    parts = n.split("-")
    return all(part in parts for part in HELD_OUT_PARTS)
