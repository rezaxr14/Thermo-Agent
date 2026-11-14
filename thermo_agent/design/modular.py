"""Modular gensets: N identical prime-mover units, each running at rated output or off, so no part-load efficiency curve is needed.

For every load bin the design is sized for that bin's need (as in the ideal part-load evaluation); the installed capacity is the
largest need over the trace times `capacity_margin`, split into `n_units` equal units; the number of units online is the smallest
count that covers the bin's need. The plant is then solved with that many units' output: the surplus electricity is curtailed
and its fuel is wasted. The gap to the ideal part-load fuel is the modularity penalty.

Limits: unit efficiency does not depend on unit size, no minimum run time, start-up or ramp limits, heat-recovery equipment and
chillers are not capacity-limited, standby units are covered only through `capacity_margin`."""
from __future__ import annotations
import math
from ..audit.changes import _scale_get, _scale_set
from ..core.schema import Site
from ..core.solve import solve
from ..verify.verifiers import run_all
from .build import build_design
from .trace_eval import bin_site


def make_plans(template: Site, spec: dict, bins: list[dict], chilled_fraction: float) -> list[dict]:
    """One entry per bin: the plant sized to the bin's need, its prime-mover size and its ideal fuel (None if infeasible)."""
    plans = []
    for b in bins:
        site = bin_site(template, b, chilled_fraction); plan = {"bin": b, "site": site, "plant": None, "x_need": None, "ideal_fuel_w": None}
        try:
            plant = build_design(site, spec); res = solve(plant)
            if res.ok and run_all(plant, res, site).valid:
                plan.update(plant=plant, x_need=_scale_get(plant)[2], ideal_fuel_w=res.kpis["Q_fuel_w"])
        except RuntimeError: pass
        plans.append(plan)
    return plans


def modular_fuel(plans: list[dict], n_units: int, step_hours: float, capacity_margin: float = 1.0) -> dict:
    total = sum(p["bin"]["samples"] for p in plans); ok = [p for p in plans if p["plant"] is not None]
    if not ok: return {"n_units": n_units, "fuel_mwh": None, "infeasible_time_share": 1.0}
    x_unit = capacity_margin * max(p["x_need"] for p in ok) / n_units
    fuel = ideal = it = surplus = online = infeasible = 0.0; per_bin = []
    for p in plans:
        share = p["bin"]["samples"] / total; hours = p["bin"]["samples"] * step_hours
        if p["plant"] is None: infeasible += share; per_bin.append({"valid": False}); continue
        k = max(1, math.ceil(p["x_need"] / x_unit - 1e-9))
        plant = _scale_set(p["plant"], k * x_unit); res = solve(plant); rep = run_all(plant, res, p["site"]) if res.ok else None
        if rep is None or not rep.valid: infeasible += share; per_bin.append({"valid": False, "online": k}); continue
        f = res.kpis["Q_fuel_w"]; load = p["site"].it_load_w
        fuel += f / 1e6 * hours; ideal += p["ideal_fuel_w"] / 1e6 * hours; it += load / 1e6 * hours
        surplus += (res.kpis["W_net_w"] - load) / 1e6 * hours; online += k * share
        per_bin.append({"it_mw": round(load / 1e6, 2), "share": round(share, 3), "online": k, "fuel_mw": round(f / 1e6, 3), "valid": True})
    return {"n_units": n_units, "fuel_mwh": round(fuel, 2), "ideal_fuel_mwh": round(ideal, 2),
            "modularity_penalty": round(fuel / ideal - 1, 4) if ideal else None, "fuel_per_it_energy": round(fuel / it, 4) if it else None,
            "surplus_electricity_mwh": round(surplus, 2), "mean_units_online": round(online, 2), "infeasible_time_share": round(infeasible, 3), "per_bin": per_bin}


def scan_units(template: Site, spec: dict, bins: list[dict], chilled_fraction: float, step_hours: float, units=range(1, 9), capacity_margin: float = 1.0) -> list[dict]:
    plans = make_plans(template, spec, bins, chilled_fraction)
    return [modular_fuel(plans, n, step_hours, capacity_margin) for n in units]
