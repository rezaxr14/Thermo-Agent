"""Component models. Each solver reads its inlet states from the Ctx, writes outlet states, and returns
True when finished or False when an inlet is not yet available (the engine retries)."""
from __future__ import annotations
import math
from ._roots import brentq
from . import gasprops as gp, fluids as fl
from .props import (State, SolveError, state_TP, state_hP, state_sP, state_sat, is_gas)

REQUIRED_PORTS = {
    "compressor": ["in", "out"], "turbine": ["in", "out"], "orc_turbine": ["in", "out"], "pump": ["in", "out"],
    "combustor": ["air_in", "fuel_in", "out"],
    "recuperator": ["cold_in", "cold_out", "hot_in", "hot_out"],
    "hrsg": ["hot_in", "hot_out", "water_in", "steam_out"],
    "orc_evaporator": ["hot_in", "hot_out", "cold_in", "cold_out"],
    "condenser": ["hot_in", "hot_out", "cold_in", "cold_out"],
    "dc_load": ["in", "out"],
    "absorption_chiller": ["gen_in", "gen_out", "evap_in", "evap_out", "rej_in", "rej_out"],
    "heat_pump": ["src_in", "src_out", "sink_in", "sink_out"],
    "electric_chiller": ["evap_in", "evap_out", "rej_in", "rej_out"],
    "heat_rejector": ["in", "out"],
    "gas_engine": ["air_in", "fuel_in", "exh_out"],
}

# Parameters per component type: name -> (meaning and unit, default or "required"). Checked against the solver source in tests.
PARAMS: dict[str, dict[str, tuple[str, object]]] = {
    "compressor": {"PR": ("pressure ratio (-)", "required"), "eta_s": ("isentropic efficiency (-)", "required")},
    "turbine": {"P_out": ("outlet pressure (Pa)", "required"), "eta_s": ("isentropic efficiency (-)", "required")},
    "orc_turbine": {"P_out": ("outlet pressure (Pa)", "required"), "eta_s": ("isentropic efficiency (-)", "required")},
    "pump": {"P_out": ("outlet pressure (Pa)", "required"), "eta_s": ("isentropic efficiency (-)", "required")},
    "combustor": {"T_out": ("outlet temperature (K)", "required"), "dp": ("pressure-drop fraction (-)", 0.0), "q_loss": ("heat-loss fraction of fuel LHV (-)", 0.0)},
    "recuperator": {"T_cold_out": ("cold-side outlet temperature (K)", "required"), "dp_cold": ("cold-side pressure-drop fraction (-)", 0.0),
                    "dp_hot": ("hot-side pressure-drop fraction (-)", 0.0), "useful_heat": ("flag: duty counts as delivered heat", False)},
    "hrsg": {"P_steam": ("steam pressure (Pa)", "required"), "dp_gas": ("gas-side pressure-drop fraction (-)", 0.0), "econ_approach_K": ("economiser approach (K)", 15.0)},
    "orc_evaporator": {"pinch_K": ("pinch at the bubble point (K)", 3.0), "dp_hot": ("hot-side pressure-drop fraction (-)", 0.0),
                       "flow_frac": ("working-fluid flow between its minimum (0) and maximum (1); used when the flow is not given", "required if flow not given"),
                       "T_stack_min": ("minimum exhaust outlet temperature (K)", 333.15), "T_max_K": ("maximum working-fluid vapour temperature (K)", 420.0)},
    "condenser": {"dT_cw": ("cooling-water temperature rise (K)", "required")},
    "dc_load": {"P_it": ("IT electrical power (W)", "required"), "chilled_fraction": ("share of the heat sent to the chilled loop (-)", 1.0),
                "T_return": ("coolant return temperature (K)", "required")},
    "absorption_chiller": {"T_chw_out": ("chilled-water outlet temperature (K)", "required"), "T_gen_out": ("generator outlet temperature (K)", "required"),
                           "dT_rej": ("heat-rejection water temperature rise (K)", "required"), "eps": ("fraction of the reversible COP (-)", 0.27),
                           "COP_max": ("first-law COP ceiling (-)", 0.8)},
    "electric_chiller": {"T_chw_out": ("chilled-water outlet temperature (K)", "required"), "dT_rej": ("heat-rejection water temperature rise (K)", "required"),
                         "eps": ("fraction of the reversible COP (-)", 0.5), "approach_K": ("evaporator and condenser approach (K)", 5.0)},
    "heat_pump": {"T_sink_out": ("sink outlet temperature (K)", "required"), "eps": ("fraction of the Lorenz COP (-)", 0.5),
                  "approach_K": ("evaporator and condenser approach (K)", 5.0), "Q_sink": ("heat duty to deliver (W); sizes the source cooling", "one of Q_sink or dT_src"),
                  "dT_src": ("source-side temperature drop (K)", "one of Q_sink or dT_src")},
    "heat_rejector": {"kind": ("'dry' (dry cooler) or 'wet' (evaporative tower)", "dry"), "approach_K": ("approach to dry-bulb or wet-bulb (K)", "required"),
                      "fan_frac": ("fan electricity as a fraction of heat rejected (-)", 0.015)},
    "gas_engine": {"P_el": ("rated electrical output (W)", "required"), "eta_el": ("electrical efficiency on LHV (-)", "required"),
                   "f_loss": ("jacket and radiation heat, fraction of fuel energy, rejected to ambient (-)", "required"),
                   "air_excess": ("air flow over stoichiometric (-)", 1.8), "P_exh": ("exhaust pressure (Pa)", 1.1e5)},
}

# expected fluid class per port: "gas" = ideal-gas mixture stream, "liquid" = CoolProp fluid; ports not listed accept either
_ALL = lambda t, ports, cls: {(t, p): cls for p in ports}
PORT_CLASS: dict[tuple[str, str], str] = {
    **_ALL("compressor", ["in", "out"], "gas"), **_ALL("turbine", ["in", "out"], "gas"),
    **_ALL("combustor", ["air_in", "fuel_in", "out"], "gas"), **_ALL("gas_engine", ["air_in", "fuel_in", "exh_out"], "gas"),
    **_ALL("hrsg", ["hot_in", "hot_out"], "gas"), **_ALL("hrsg", ["water_in", "steam_out"], "liquid"),
    **_ALL("orc_evaporator", ["hot_in", "hot_out"], "gas"), **_ALL("orc_evaporator", ["cold_in", "cold_out"], "liquid"),
    **_ALL("orc_turbine", ["in", "out"], "liquid"), **_ALL("pump", ["in", "out"], "liquid"),
    **_ALL("condenser", ["hot_in", "hot_out", "cold_in", "cold_out"], "liquid"),
    **_ALL("dc_load", ["in", "out"], "liquid"), **_ALL("heat_rejector", ["in", "out"], "liquid"),
    **_ALL("absorption_chiller", REQUIRED_PORTS["absorption_chiller"], "liquid"),
    **_ALL("electric_chiller", REQUIRED_PORTS["electric_chiller"], "liquid"),
    **_ALL("heat_pump", REQUIRED_PORTS["heat_pump"], "liquid"),
}


def _p(c, name, default=None):
    if name in c.params: return c.params[name]
    if default is None: raise SolveError("E_UNDER_SPECIFIED", f"{c.id}: missing parameter '{name}'")
    return default


def compressor(ctx, c):
    i = ctx.inp(c.id, "in")
    if i is None: return False
    PR, eta = _p(c, "PR"), _p(c, "eta_s")
    if eta <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: eta_s must be > 0")
    P2 = i.P * PR
    s2 = state_sP("", i.fluid, i.comp, i.s, P2, i.n)
    h2 = i.h + (s2.h - i.h) / eta
    out = state_hP("", i.fluid, i.comp, h2, P2, i.n)
    ctx.put(c.id, "out", out); ctx.work[c.id] = -i.n * (h2 - i.h)
    return True


def expander(ctx, c):
    i = ctx.inp(c.id, "in")
    if i is None: return False
    Pout, eta = _p(c, "P_out"), _p(c, "eta_s")
    if eta <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: eta_s must be > 0")
    s2 = state_sP("", i.fluid, i.comp, i.s, Pout, i.n)
    h2 = i.h - eta * (i.h - s2.h)
    out = state_hP("", i.fluid, i.comp, h2, Pout, i.n)
    ctx.put(c.id, "out", out); ctx.work[c.id] = i.n * (i.h - h2)
    return True


def pump(ctx, c):
    """Pump. If the working-fluid flow is not yet known (the ORC evaporator sets it), the outlet state is published
    first and the work is computed once the flow is available."""
    i = ctx.inp(c.id, "in")
    if i is None: return False
    a = ctx.aux.setdefault(c.id, {})
    if "out" not in a:
        Pout, eta = _p(c, "P_out"), _p(c, "eta_s")
        if eta <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: eta_s must be > 0")
        s2 = state_sP("", i.fluid, i.comp, i.s, Pout, i.n)
        h2 = i.h + (s2.h - i.h) / eta
        a["out"] = state_hP("", i.fluid, i.comp, h2, Pout, i.n); a["dh"] = h2 - i.h
        ctx.put(c.id, "out", a["out"])
    out = a["out"]; n = i.n if i.n is not None else out.n
    if n is None: return False
    i.n = out.n = n; ctx.work[c.id] = -n * a["dh"]
    return True


def recuperator(ctx, c):
    a = ctx.aux.setdefault(c.id, {})
    ci, hi = ctx.inp(c.id, "cold_in"), ctx.inp(c.id, "hot_in")
    if "cold_out" not in a:                       # cold-side outlet state is fixed by T_cold_out; flow may still be unknown
        if ci is None: return False
        T_out, dpc = _p(c, "T_cold_out"), _p(c, "dp_cold", 0.0)
        a["cold_out"] = state_TP("", ci.fluid, ci.comp, T_out, ci.P * (1 - dpc), ci.n)
        ctx.put(c.id, "cold_out", a["cold_out"])
    out = a["cold_out"]
    if "Q" not in a:                              # duty needs the cold-side flow (set here or by the machine downstream)
        n = ci.n if ci.n is not None else out.n
        if n is None: return False
        out.n = n
        a["Q"] = n * (out.h - ci.h); a["T_cold_in"], a["T_cold_out_T"] = ci.T, out.T
    if hi is None: return False
    dph = _p(c, "dp_hot", 0.0)
    try: hot = state_hP("", hi.fluid, hi.comp, hi.h - a["Q"] / hi.n, hi.P * (1 - dph), hi.n)
    except SolveError as e:
        if e.code != "E_PROP_RANGE": raise
        raise SolveError("E_TEMP_CROSS", f"{c.id}: the hot stream cannot supply the required duty of {a['Q']/1e6:.1f} MW")
    ctx.put(c.id, "hot_out", hot); ctx.Q[c.id] = a["Q"]
    a["min_approach"] = min(hi.T - a["T_cold_out_T"], hot.T - a["T_cold_in"])
    return True


def combustor(ctx, c):
    air, fuel = ctx.inp(c.id, "air_in"), ctx.inp(c.id, "fuel_in")
    if air is None or fuel is None: return False
    if air.comp is None or fuel.comp is None: raise SolveError("E_PROP_RANGE", f"{c.id}: gas inlets required")
    Tout, dp, qloss = _p(c, "T_out"), _p(c, "dp", 0.0), _p(c, "q_loss", 0.0)
    a = air.comp; lhv = gp.lhv_molar()

    def prod(lam):
        y = {"CO2": a.get("CO2", 0) + lam, "H2O": a.get("H2O", 0) + 2 * lam, "O2": a.get("O2", 0) - 2 * lam, "N2": a.get("N2", 0)}
        return {k: v / (1 + lam) for k, v in y.items()}

    def resid(lam):
        y = prod(lam)
        return (1 + lam) * gp.h_mix(y, Tout) - air.h - lam * fuel.h + qloss * lam * lhv
    lam_max = a.get("O2", 0) / 2 * 0.999
    try:
        lam = brentq(resid, 1e-6, lam_max, xtol=1e-14)
    except ValueError:
        raise SolveError("E_NOT_CONVERGED", f"{c.id}: T_out={Tout:.0f} K not reachable with available oxygen")
    y = prod(lam)
    nF = lam * air.n
    out = state_TP("", "FLUE", y, Tout, air.P * (1 - dp), air.n + nF)
    fuel.n = nF
    ctx.put(c.id, "out", out); ctx.qloss[c.id] = qloss * nF * lhv
    ctx.aux[c.id] = {"lam": lam, "Q_fuel": nF * lhv}
    return True


def hrsg(ctx, c):
    hi, wi = ctx.inp(c.id, "hot_in"), ctx.inp(c.id, "water_in")
    if hi is None or wi is None: return False
    Ps, dpg, appr = _p(c, "P_steam"), _p(c, "dp_gas", 0.0), _p(c, "econ_approach_K", 15.0)
    if wi.n is None: raise SolveError("E_UNDER_SPECIFIED", f"{c.id}: feedwater flow not specified")
    sv = state_sat("", wi.fluid, Ps, 1, wi.n)
    Q = wi.n * (sv.h - wi.h)
    out = state_hP("", hi.fluid, hi.comp, hi.h - Q / hi.n, hi.P * (1 - dpg), hi.n)
    h_s = fl.h_TP(wi.fluid, sv.T - appr, Ps)
    T_pp = gp.T_from_h(hi.comp, hi.h - wi.n * (sv.h - h_s) / hi.n)
    ctx.aux[c.id] = {"Q": Q, "min_approach": min(T_pp - sv.T, out.T - wi.T), "T_sat": sv.T}
    ctx.put(c.id, "hot_out", out); ctx.put(c.id, "steam_out", sv); ctx.Q[c.id] = Q
    return True


def orc_evaporator(ctx, c):
    hi, ci = ctx.inp(c.id, "hot_in"), ctx.inp(c.id, "cold_in")
    if hi is None or ci is None: return False
    pinch, dph = _p(c, "pinch_K", 3.0), _p(c, "dp_hot", 0.0)
    Tb = fl.Tsat_P(ci.fluid, ci.P); hb = fl.h_Pq(ci.fluid, ci.P, 0)
    if hi.T <= Tb + pinch:
        raise SolveError("E_TEMP_CROSS", f"{c.id}: hot-side inlet {hi.T:.1f} K cannot supply pinch {pinch} K above bubble point {Tb:.1f} K")
    if ci.n is None:
        # Working-fluid flow between two limits: the smallest flow keeps the vapour at or below T_max_K (fluid thermal limit),
        # the largest keeps the exhaust at or above T_stack_min. flow_frac = 0 -> smallest flow, 1 -> largest.
        ff, T_floor = _p(c, "flow_frac"), _p(c, "T_stack_min", 333.15)
        T_lim = min(_p(c, "T_max_K", 420.0), fl.Tmax(ci.fluid) - 5.0)
        if not (0 < ff <= 1): raise SolveError("E_PROP_RANGE", f"{c.id}: flow_frac must be in (0, 1]")
        h_pp, h_fl = gp.h_mix(hi.comp, Tb + pinch), gp.h_mix(hi.comp, T_floor)
        if h_pp <= h_fl: raise SolveError("E_TEMP_CROSS", f"{c.id}: pinch point {Tb + pinch:.1f} K is below the minimum stack temperature {T_floor:.1f} K")
        if T_lim <= Tb + 1: raise SolveError("E_PROP_RANGE", f"{c.id}: fluid temperature limit {T_lim:.1f} K is not above the bubble point {Tb:.1f} K")
        n_max = hi.n * (h_pp - h_fl) / (hb - ci.h)
        n_min = hi.n * (hi.h - h_pp) / (fl.h_TP(ci.fluid, T_lim, ci.P) - hb)
        if n_min > n_max:
            raise SolveError("E_TEMP_CROSS", f"{c.id}: exhaust at {hi.T:.0f} K is too hot for {ci.fluid} (vapour limit {T_lim:.0f} K) while leaving the stack above {T_floor:.0f} K")
        ci.n = n_min + ff * (n_max - n_min)
    h13 = hb + hi.n * (hi.h - gp.h_mix(hi.comp, Tb + pinch)) / ci.n
    cold = state_hP("", ci.fluid, None, h13, ci.P, ci.n)
    Q = ci.n * (h13 - ci.h)
    hot = state_hP("", hi.fluid, hi.comp, hi.h - Q / hi.n, hi.P * (1 - dph), hi.n)
    ctx.aux[c.id] = {"Q": Q, "min_approach": min(pinch, hot.T - ci.T, hi.T - cold.T)}
    ctx.put(c.id, "cold_out", cold); ctx.put(c.id, "hot_out", hot); ctx.Q[c.id] = Q
    return True


def condenser(ctx, c):
    hi, ci = ctx.inp(c.id, "hot_in"), ctx.inp(c.id, "cold_in")
    if hi is None or ci is None: return False
    dT = _p(c, "dT_cw")
    if dT <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: dT_cw must be > 0")
    hot = state_sat("", hi.fluid, hi.P, 0, hi.n)
    cold = state_TP("", ci.fluid, None, ci.T + dT, ci.P, None)
    n_w = hi.n * (hi.h - hot.h) / (cold.h - ci.h)
    ci.n = n_w; cold.n = n_w
    ctx.aux[c.id] = {"Q": hi.n * (hi.h - hot.h), "min_approach": hot.T - cold.T, "T_sat": hot.T}
    ctx.put(c.id, "hot_out", hot); ctx.put(c.id, "cold_out", cold); ctx.Q[c.id] = ctx.aux[c.id]["Q"]
    return True


def _lm(a, b):
    """Log-mean of two temperatures: integral of dQ/T for a constant-cp stream is Q / T_lm."""
    return a if abs(a - b) < 1e-9 else (a - b) / math.log(a / b)


def dc_load(ctx, c):
    """IT load: consumes P_it of electricity; a fraction of it leaves as heat into the coolant loop, the rest is
    rejected directly to ambient (air side)."""
    i = ctx.inp(c.id, "in")
    if i is None: return False
    P_it, f, T_ret = _p(c, "P_it"), _p(c, "chilled_fraction", 1.0), _p(c, "T_return")
    if not (0 < f <= 1): raise SolveError("E_PROP_RANGE", f"{c.id}: chilled_fraction must be in (0, 1]")
    if T_ret <= i.T: raise SolveError("E_PROP_RANGE", f"{c.id}: T_return {T_ret:.1f} K must exceed supply {i.T:.1f} K")
    if i.n is not None: raise SolveError("E_OVER_SPECIFIED", f"{c.id}: coolant flow is solved from the load; do not specify it")
    out = state_TP("", i.fluid, None, T_ret, i.P, None)
    n = f * P_it / (out.h - i.h); i.n = n; out.n = n
    ctx.put(c.id, "out", out); ctx.work[c.id] = -P_it; ctx.qloss[c.id] = (1 - f) * P_it; ctx.Q[c.id] = f * P_it
    return True


def absorption_chiller(ctx, c):
    """Single-effect, hot-water driven. COP = min(eps * COP_rev, COP_max); COP_rev is the reversible three-temperature
    limit evaluated with log-mean stream temperatures. Measured LiBr/H2O single-effect units run at COP 0.55-0.70
    (Ochoa Villa et al., Acta Sci. Technol. 41, 2019); COP_max is the first-law ceiling of the cycle."""
    gi, ei, ri = ctx.inp(c.id, "gen_in"), ctx.inp(c.id, "evap_in"), ctx.inp(c.id, "rej_in")
    if gi is None or ei is None or ri is None or ei.n is None: return False
    T_chw, T_go, dTr = _p(c, "T_chw_out"), _p(c, "T_gen_out"), _p(c, "dT_rej")
    eps, cop_max = _p(c, "eps", 0.27), _p(c, "COP_max", 0.8)
    if gi.n is not None: raise SolveError("E_OVER_SPECIFIED", f"{c.id}: generator flow is solved; do not specify it")
    if not (T_chw < ei.T): raise SolveError("E_PROP_RANGE", f"{c.id}: chilled-water outlet must be colder than its inlet")
    if not (T_go < gi.T): raise SolveError("E_PROP_RANGE", f"{c.id}: generator outlet must be colder than its inlet")
    if dTr <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: dT_rej must be > 0")
    eo = state_TP("", ei.fluid, None, T_chw, ei.P, ei.n)
    go = state_TP("", gi.fluid, None, T_go, gi.P, None)
    ro = state_TP("", ri.fluid, None, ri.T + dTr, ri.P, None)
    Tg, Te, Tr = _lm(gi.T, T_go), _lm(ei.T, T_chw), _lm(ri.T, ro.T)
    if Tg <= Tr: raise SolveError("E_TEMP_CROSS", f"{c.id}: driving temperature {Tg:.1f} K is not above heat-rejection temperature {Tr:.1f} K")
    if Tr <= Te: raise SolveError("E_TEMP_CROSS", f"{c.id}: heat rejection {Tr:.1f} K is not above evaporator {Te:.1f} K")
    cop_rev = (1 - Tr / Tg) / (Tr / Te - 1)
    COP = min(eps * cop_rev, cop_max)
    Qe = ei.n * (ei.h - eo.h); Qg = Qe / COP; Qr = Qg + Qe
    ng = Qg / (gi.h - go.h); nr = Qr / (ro.h - ri.h)
    gi.n = go.n = ng; ri.n = ro.n = nr
    ctx.put(c.id, "gen_out", go); ctx.put(c.id, "evap_out", eo); ctx.put(c.id, "rej_out", ro)
    ctx.aux[c.id] = {"COP": COP, "COP_rev": cop_rev, "Q_g": Qg}; ctx.Q[c.id] = Qe
    return True


def electric_chiller(ctx, c):
    """Vapour-compression chiller: COP = eps * Te'/(Tr' - Te'), where Te' and Tr' are the refrigerant temperatures
    (log-mean stream temperatures moved outward by approach_K at each heat exchanger); electricity consumed."""
    ei, ri = ctx.inp(c.id, "evap_in"), ctx.inp(c.id, "rej_in")
    if ei is None or ri is None or ei.n is None: return False
    T_chw, dTr, eps, ap = _p(c, "T_chw_out"), _p(c, "dT_rej"), _p(c, "eps", 0.5), _p(c, "approach_K", 5.0)
    if not (T_chw < ei.T): raise SolveError("E_PROP_RANGE", f"{c.id}: chilled-water outlet must be colder than its inlet")
    if dTr <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: dT_rej must be > 0")
    eo = state_TP("", ei.fluid, None, T_chw, ei.P, ei.n)
    ro = state_TP("", ri.fluid, None, ri.T + dTr, ri.P, None)
    Te, Tr = _lm(ei.T, T_chw), _lm(ri.T, ro.T)
    if Tr <= Te: raise SolveError("E_TEMP_CROSS", f"{c.id}: heat rejection {Tr:.1f} K is not above evaporator {Te:.1f} K")
    COP = eps * (Te - ap) / ((Tr + ap) - (Te - ap))
    Qe = ei.n * (ei.h - eo.h); W = Qe / COP; nr = (Qe + W) / (ro.h - ri.h)
    ri.n = ro.n = nr
    ctx.put(c.id, "evap_out", eo); ctx.put(c.id, "rej_out", ro)
    ctx.work[c.id] = -W; ctx.Q[c.id] = Qe; ctx.aux[c.id] = {"COP": COP, "COP_rev": Te / (Tr - Te)}
    return True


def rejector_outlet_T(T0: float, P0: float, RH: float, kind: str, approach_K: float) -> float:
    """Outlet temperature of a heat rejector: dry cooler = dry-bulb + approach; evaporative tower = wet-bulb + approach."""
    import CoolProp.CoolProp as CP
    ref = T0 if kind == "dry" else CP.HAPropsSI("Twb", "T", T0, "P", P0, "R", RH)
    return ref + approach_K


def heat_rejector(ctx, c):
    """Dry cooler or evaporative cooling tower on a water loop; fans consume fan_frac of the heat rejected.
    (Make-up water for evaporation is not modelled.)"""
    i = ctx.inp(c.id, "in")
    if i is None or i.n is None: return False
    kind = c.params.get("kind", "dry")
    if kind not in ("dry", "wet"): raise SolveError("E_PROP_RANGE", f"{c.id}: kind must be 'dry' or 'wet'")
    ap, fan = _p(c, "approach_K"), _p(c, "fan_frac", 0.015)
    if ap <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: approach_K must be > 0")
    amb = ctx.plant.ambient
    T_out = rejector_outlet_T(amb.T0, amb.P0, amb.RH, kind, ap)
    if T_out >= i.T: raise SolveError("E_TEMP_CROSS", f"{c.id}: inlet {i.T:.1f} K is not above the achievable outlet {T_out:.1f} K")
    out = state_TP("", i.fluid, None, T_out, i.P, i.n)
    Q = i.n * (i.h - out.h); W = fan * Q
    ctx.put(c.id, "out", out); ctx.work[c.id] = -W; ctx.qloss[c.id] = Q + W; ctx.Q[c.id] = Q
    # evaporative make-up water: upper bound, all heat rejected as latent heat at the outlet temperature
    import CoolProp.CoolProp as CP
    hfg = CP.PropsSI("H", "T", T_out, "Q", 1, "Water") - CP.PropsSI("H", "T", T_out, "Q", 0, "Water")
    ctx.aux[c.id] = {"water_kg_s": Q / hfg if kind == "wet" else 0.0}
    return True


def gas_engine(ctx, c):
    """Reciprocating gas engine sized by rated electrical output. Fuel = P_el/(eta_el*LHV). Jacket and radiation heat
    (f_loss of the fuel energy) are rejected to ambient, not recovered. The exhaust carries the remaining energy;
    its temperature follows from the energy balance."""
    air, fuel = ctx.inp(c.id, "air_in"), ctx.inp(c.id, "fuel_in")
    if air is None or fuel is None: return False
    if air.n is not None or fuel.n is not None:
        raise SolveError("E_OVER_SPECIFIED", f"{c.id}: air and fuel flows are solved from P_el; do not specify them")
    P_el, eta, fl_, lam_air, P_exh = _p(c, "P_el"), _p(c, "eta_el"), _p(c, "f_loss"), _p(c, "air_excess", 1.8), _p(c, "P_exh", 1.1e5)
    if not (0 < eta < 1 and 0 <= fl_ < 1 and eta + fl_ < 1): raise SolveError("E_PROP_RANGE", f"{c.id}: need eta_el + f_loss < 1")
    if lam_air < 1.0: raise SolveError("E_PROP_RANGE", f"{c.id}: air_excess must be >= 1")
    a = air.comp; lhv = gp.lhv_molar()
    nF = P_el / (eta * lhv); nA = lam_air * nF * 2.0 / a["O2"]
    lam = nF / nA
    y = {"CO2": a.get("CO2", 0) + lam, "H2O": a.get("H2O", 0) + 2 * lam, "O2": a.get("O2", 0) - 2 * lam, "N2": a.get("N2", 0)}
    y = {k: v / (1 + lam) for k, v in y.items()}
    h_ex = (nA * air.h + nF * fuel.h - P_el - fl_ * nF * lhv) / (nA + nF)
    try: T_ex = gp.T_from_h(y, h_ex)
    except ValueError: raise SolveError("E_PROP_RANGE", f"{c.id}: exhaust enthalpy outside the property range; check eta_el, f_loss, air_excess")
    air.n, fuel.n = nA, nF
    ctx.put(c.id, "exh_out", state_TP("", "FLUE", y, T_ex, P_exh, nA + nF))
    ctx.work[c.id] = P_el; ctx.qloss[c.id] = fl_ * nF * lhv; ctx.aux[c.id] = {"Q_fuel": nF * lhv, "T_exh": T_ex}
    return True


def heat_pump(ctx, c):
    """Heat pump lifting heat from a coolant stream to a sink stream. COP = eps * Lorenz COP on refrigerant-side temperatures
    (log-mean stream temperatures moved outward by approach_K); electricity consumed."""
    si, ki = ctx.inp(c.id, "src_in"), ctx.inp(c.id, "sink_in")
    if si is None or ki is None or si.n is None: return False
    T_sink, eps, ap = _p(c, "T_sink_out"), _p(c, "eps", 0.5), _p(c, "approach_K", 5.0)
    if ki.n is not None: raise SolveError("E_OVER_SPECIFIED", f"{c.id}: sink flow is solved; do not specify it")
    if not (T_sink > ki.T): raise SolveError("E_PROP_RANGE", f"{c.id}: sink outlet must be hotter than its inlet")
    ko = state_TP("", ki.fluid, None, T_sink, ki.P, None); Ts = _lm(ki.T, T_sink)

    def cop_for(T_src_out):
        Tc = _lm(si.T, T_src_out)
        if Ts <= Tc: raise SolveError("E_TEMP_CROSS", f"{c.id}: sink {Ts:.1f} K is not hotter than source {Tc:.1f} K")
        return eps * (Ts + ap) / ((Ts + ap) - (Tc - ap))
    if "Q_sink" in c.params:                      # size the source-side cooling to deliver a given heat duty
        Qk = _p(c, "Q_sink"); T_out = si.T - 3.0
        for _ in range(60):
            COP = cop_for(T_out)
            if COP <= 1: raise SolveError("E_PROP_RANGE", f"{c.id}: COP {COP:.2f} <= 1")
            so = state_hP("", si.fluid, None, si.h - Qk * (COP - 1) / COP / si.n, si.P, si.n)
            if abs(so.T - T_out) < 1e-7: break
            T_out = so.T
        else: raise SolveError("E_NOT_CONVERGED", f"{c.id}: heat-pump source temperature did not converge")
        Qs = si.n * (si.h - so.h); W = Qk - Qs
    else:
        dTs = _p(c, "dT_src")
        if dTs <= 0: raise SolveError("E_PROP_RANGE", f"{c.id}: dT_src must be > 0")
        so = state_TP("", si.fluid, None, si.T - dTs, si.P, si.n)
        COP = cop_for(so.T)
        if COP <= 1: raise SolveError("E_PROP_RANGE", f"{c.id}: COP {COP:.2f} <= 1")
        Qs = si.n * (si.h - so.h); Qk = Qs * COP / (COP - 1); W = Qk - Qs
    Tc = _lm(si.T, so.T)
    nk = Qk / (ko.h - ki.h); ki.n = ko.n = nk
    ctx.put(c.id, "src_out", so); ctx.put(c.id, "sink_out", ko)
    ctx.work[c.id] = -W; ctx.Q[c.id] = Qk; ctx.aux[c.id] = {"COP": COP, "COP_rev": Ts / (Ts - Tc)}
    return True


SOLVERS = {"heat_rejector": heat_rejector, "gas_engine": gas_engine, "dc_load": dc_load, "absorption_chiller": absorption_chiller, "electric_chiller": electric_chiller,
           "heat_pump": heat_pump, "compressor": compressor, "turbine": expander, "orc_turbine": expander, "pump": pump,
           "combustor": combustor, "recuperator": recuperator, "hrsg": hrsg,
           "orc_evaporator": orc_evaporator, "condenser": condenser}
