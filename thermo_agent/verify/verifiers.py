"""Physics verifiers. Each check returns (ok, detail); `hard` checks gate validity, soft ones only lower the score."""
from __future__ import annotations
from dataclasses import dataclass, field
from ..core.schema import Plant, SolveResult

TOL_REL = 1e-6


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hard: bool = True


@dataclass
class Report:
    solved: bool
    error: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def valid(self) -> bool:                       # all hard checks pass
        return self.solved and all(c.ok for c in self.checks if c.hard)

    @property
    def score(self) -> float:                      # fraction of checks passed; 0 if the plant does not solve
        return 0.0 if not self.solved else sum(c.ok for c in self.checks) / max(len(self.checks), 1)

    @property
    def reward(self) -> float:                     # 1.0 only for a fully valid plant; partial credit capped below 1
        return 0.0 if not self.solved else (1.0 if self.valid and self.score == 1.0 else 0.5 * self.score)

    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.ok]

    def as_dict(self) -> dict:
        return {"solved": self.solved, "error": self.error or "", "valid": self.valid, "score": self.score,
                "failed": ",".join(self.failed())}


def v1_mass(plant, res):
    scale = max((s.m_dot for s in res.states), default=1.0)
    bad = [b.component for b in res.balances if abs(b.mass_residual) > TOL_REL * scale]
    return Check("V1_mass_balance", not bad, ",".join(bad))


def v2_energy(plant, res):
    k = res.kpis; scale = abs(k.get("W_net_w", 0)) + abs(k.get("Q_fuel_w", 0)) + 1e6
    bad = [b.component for b in res.balances if abs(b.energy_residual) > TOL_REL * scale]
    return Check("V2_energy_balance", not bad, ",".join(bad))


def v3_exergy(plant, res):
    k = res.kpis; scale = max(abs(k.get("E_fuel_w", 0)), 1e6)
    neg = [f"{b.component}({b.exergy_destruction_w/1e6:.4f} MW)" for b in res.balances if b.exergy_destruction_w < -1e-5 * scale]
    out = [Check("V3a_nonneg_exergy_destruction", not neg, ",".join(neg))]
    out.append(Check("V3b_exergy_closure", abs(k.get("closure_residual_w", 0)) <= 1e-6 * scale,
                     f"residual {k.get('closure_residual_w', 0)/1e6:.6f} MW"))
    return out


def v4_pinch(plant, res, practical_K=1.0):
    cross = [f"{b.component}({b.min_approach_K:.2f} K)" for b in res.balances if b.min_approach_K is not None and b.min_approach_K < 0]
    tight = [f"{b.component}({b.min_approach_K:.2f} K)" for b in res.balances
             if b.min_approach_K is not None and 0 <= b.min_approach_K < practical_K]
    return [Check("V4a_no_temperature_cross", not cross, ",".join(cross)),
            Check("V4b_practical_pinch", not tight, ",".join(tight), hard=False)]


def v5_bounds(plant, res):
    k = res.kpis; out = []
    if "eta_II" in k:
        out.append(Check("V5a_second_law_efficiency", 0 <= k["eta_II"] <= 1, f"eta_II={k['eta_II']:.4f}"))
    if "eta_I_power" in k and k.get("T_max_K", 0) > k["T0_K"]:
        carnot = 1 - k["T0_K"] / k["T_max_K"]
        out.append(Check("V5b_power_below_carnot", k["eta_I_power"] <= carnot, f"{k['eta_I_power']:.4f} <= {carnot:.4f}"))
    if "W_orc_net_w" in k and k.get("Q_evap_orc_w", 0) > 0 and "T_cond_K" in k:
        carnot = 1 - k["T_cond_K"] / k["T_orc_hot_K"]
        eta = k["W_orc_net_w"] / k["Q_evap_orc_w"]
        out.append(Check("V5c_orc_below_carnot", eta <= carnot, f"{eta:.4f} <= {carnot:.4f}"))
    machines = sorted(key[4:] for key in k if key.startswith("COP_") and f"COPrev_{key[4:]}" in k)
    if machines:                                    # every chiller and heat pump: COP at or below the reversible limit of its stream temperatures
        bad = [f"{m} (COP {k['COP_' + m]:.2f} > {k['COPrev_' + m]:.2f})" for m in machines if k["COP_" + m] > k["COPrev_" + m] * (1 + 1e-9)]
        out.append(Check("V5d_cop_below_reversible", not bad, ", ".join(bad)))
    return out


def v6_ranges(plant: Plant, res: SolveResult):
    bad = []
    for c in plant.components:
        for k, v in c.params.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool): continue
            if k.startswith(("eta", "eps")) and not (0 < v <= 1): bad.append(f"{c.id}.{k}={v}")
            elif k == "chilled_fraction" and not (0 < v <= 1): bad.append(f"{c.id}.{k}={v}")
            elif k == "COP_max" and not (0 < v <= 1.2): bad.append(f"{c.id}.{k}={v}")
            elif k == "PR" and v <= 1: bad.append(f"{c.id}.{k}={v}")
            elif k.startswith("dp") and not (0 <= v <= 0.3): bad.append(f"{c.id}.{k}={v}")
            elif k == "flow_frac" and not (0 < v <= 1): bad.append(f"{c.id}.{k}={v}")
            elif k == "f_loss" and not (0 <= v < 0.6): bad.append(f"{c.id}.{k}={v}")
            elif k == "air_excess" and not (1.0 <= v <= 4.0): bad.append(f"{c.id}.{k}={v}")
            elif k == "fan_frac" and not (0 <= v <= 0.1): bad.append(f"{c.id}.{k}={v}")
            elif k.startswith("T_") and not (150 <= v <= 2500): bad.append(f"{c.id}.{k}={v}")
            elif k in ("P_it", "P_el") and not (1e4 <= v <= 1e9): bad.append(f"{c.id}.{k}={v}")
            elif k.startswith("P_") and k not in ("P_it", "P_el") and not (1e3 <= v <= 1e8): bad.append(f"{c.id}.{k}={v}")
            elif k == "approach_K" and not (0 <= v <= 30) and c.type != "heat_rejector": bad.append(f"{c.id}.{k}={v}")
            elif k.startswith("pinch") and not (0 <= v <= 200): bad.append(f"{c.id}.{k}={v}")
            elif k == "approach_K" and c.type == "heat_rejector" and not (0 < v <= 25): bad.append(f"{c.id}.{k}={v}")
            elif k.startswith("dT") and not (0 < v <= 100): bad.append(f"{c.id}.{k}={v}")
            elif k == "q_loss" and not (0 <= v < 0.2): bad.append(f"{c.id}.{k}={v}")
    for s in res.states:
        if not (150 <= s.T <= 2500): bad.append(f"{s.stream}.T={s.T:.1f}")
        if not (1e3 <= s.P <= 1e8): bad.append(f"{s.stream}.P={s.P:.3g}")
        if s.m_dot <= 0: bad.append(f"{s.stream}.m={s.m_dot:.3g}")
    st = {s.stream: s for s in res.states}
    for c in plant.components:                       # direction of pressure change
        if c.type in ("compressor", "pump"):
            i, o = st.get(_sid(plant, c.id, "in")), st.get(_sid(plant, c.id, "out"))
            if i and o and o.P <= i.P: bad.append(f"{c.id}: outlet pressure not above inlet")
        if c.type in ("turbine", "orc_turbine"):
            i, o = st.get(_sid(plant, c.id, "in")), st.get(_sid(plant, c.id, "out"))
            if i and o and o.P >= i.P: bad.append(f"{c.id}: outlet pressure not below inlet")
    return Check("V6_unit_and_range", not bad, "; ".join(bad))


def _sid(plant, comp, port):
    for s in plant.streams:
        if s.src == f"{comp}.{port}" or s.dst == f"{comp}.{port}": return s.id
    return None


def v8_site(plant: Plant, res: SolveResult, site) -> list[Check]:
    """Does the plant actually serve the site it was designed for?"""
    k = res.kpis; out = []
    out.append(Check("V8a_it_load_served", abs(k.get("P_it_w", 0) - site.it_load_w) <= 0.01 * site.it_load_w,
                     f"plant IT load {k.get('P_it_w', 0)/1e6:.2f} MW vs site {site.it_load_w/1e6:.2f} MW"))
    deficit = -(k["W_net_w"] - k.get("P_it_w", 0))
    ok = deficit <= 1e-3 * site.it_load_w or site.grid_import_allowed
    out.append(Check("V8b_electricity_balance", ok, f"net surplus {-deficit/1e6:.2f} MW"))
    if site.cooling_load_w:
        out.append(Check("V8c_cooling_delivered", k.get("Q_cooling_w", 0) >= 0.99 * site.cooling_load_w,
                         f"{k.get('Q_cooling_w', 0)/1e6:.2f} MW vs required {site.cooling_load_w/1e6:.2f} MW"))
    if site.heat_demand_w:
        out.append(Check("V8d_heat_delivered", k.get("Q_heat_w", 0) >= 0.99 * site.heat_demand_w,
                         f"{k.get('Q_heat_w', 0)/1e6:.2f} MW vs required {site.heat_demand_w/1e6:.2f} MW"))
    if site.max_water_kg_s is not None:
        w = k.get("water_use_kg_s", 0.0)
        out.append(Check("V8e_water_limit", w <= site.max_water_kg_s, f"{w:.1f} kg/s vs limit {site.max_water_kg_s:.1f} kg/s"))
    return out


def run_all(plant: Plant, res: SolveResult, site=None) -> Report:
    if not res.ok:
        return Report(False, res.error.value if res.error else "unknown", [])
    cs = [v1_mass(plant, res), v2_energy(plant, res), *v3_exergy(plant, res), *v4_pinch(plant, res),
          *v5_bounds(plant, res), v6_ranges(plant, res)]
    if site is not None: cs += v8_site(plant, res, site)
    return Report(True, None, cs)
