"""Site + design -> sized plant, and the objective used to compare designs."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from ..core.schema import Site, Plant, SolveResult
from ..core.solve import solve
from ..verify.verifiers import run_all, Report
from ..synth.topologies import dc_chp, size_generation
from .spec import validate, DesignError, ORC_FLUID

T_SUPPLY_DEFAULT, T_RETURN_DEFAULT = 291.15, 301.15


def build_design(site: Site, spec: dict) -> Plant:
    """Raises RuntimeError when the plant cannot be sized (infeasible heat recovery) and DesignError for bad specs."""
    needs_heat = site.heat_demand_w > 0
    validate(spec, needs_heat)
    cf = (site.cooling_load_w or 0.5 * site.it_load_w) / site.it_load_w
    if not (0 < cf <= 1): raise DesignError("cooling load must be in (0, IT load]")
    kw = dict(P_it=site.it_load_w, chilled_fraction=cf, cooling=spec["cooling"], rejection=spec["rejection"],
              prime_mover=spec["prime_mover"], T_amb=site.ambient.T0, RH=site.ambient.RH,
              T_supply=site.t_supply_k or T_SUPPLY_DEFAULT, T_return=site.t_return_k or T_RETURN_DEFAULT, dT_rej=spec["dT_rej"])
    if spec["prime_mover"] == "turbine": kw.update(PR=spec["PR"], T4=spec["T4"])
    if spec["cooling"] == "absorption": kw.update(T_hw=spec["T_hw"], T_gen_out=spec["T_hw"] - spec["dT_hw"])
    if spec["bottoming"] != "none": kw.update(orc=True, orc_fluid=ORC_FLUID[spec["bottoming"]], orc_frac=spec["orc_frac"], P_orc_high=spec["P_orc_bar"] * 1e5)
    if needs_heat:
        kw.update(heat_demand_w=site.heat_demand_w, heat_source=spec["heat_source"], T_dh_in=site.t_dh_return_k or 323.15,
                  T_dh_out=site.t_dh_supply_k or 343.15)
    return size_generation(dc_chp, **kw)


@dataclass
class DesignEval:
    built: bool; valid: bool; reason: str = ""
    fuel_w: float | None = None
    plant: Plant | None = None; result: SolveResult | None = None; report: Report | None = None
    n_solves: int = 0


def evaluate_design(site: Site, spec: dict) -> DesignEval:
    """Build, solve, verify against the site. fuel_w is set only for valid designs."""
    try:
        plant = build_design(site, spec)
    except DesignError as e:
        return DesignEval(False, False, f"design: {e}")
    except RuntimeError as e:
        return DesignEval(False, False, f"sizing: {e}", n_solves=1)
    res = solve(plant); rep = run_all(plant, res, site)
    if not res.ok: return DesignEval(True, False, f"{res.error.value}: {res.message}", plant=plant, result=res, report=rep, n_solves=4)
    ev = DesignEval(True, rep.valid, "" if rep.valid else "failed: " + ",".join(rep.failed()), plant=plant, result=res, report=rep, n_solves=4)
    if rep.valid: ev.fuel_w = res.kpis["Q_fuel_w"]
    return ev
