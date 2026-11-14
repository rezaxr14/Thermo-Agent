"""Plant builders (typed graphs for the schema)."""
from __future__ import annotations
from ..core.schema import Plant, Component, Stream, FixedState, Ambient
from ..core import fluids as fl, gasprops as gp

ATM = 101325.0
# maximum working-fluid vapour temperature (K): design constants chosen below each fluid's EOS limit, not measured stability data
ORC_T_MAX = {"R245fa": 420.0, "Cyclopentane": 480.0, "Toluene": 560.0}


def cgam_orc(*, n_air: float = 3174.4, T_amb: float = 298.15, PR: float = 10.0, eta_c: float = 0.86, eta_t: float = 0.86,
             T3: float = 850.0, T4: float = 1520.0, dp_ap_air: float = 0.05, dp_cc: float = 0.05, dp_ap_hot: float = 0.03,
             dp_hrsg: float = 0.05, q_loss: float = 0.02, m_steam: float = 14.0, P_steam: float = 20e5,
             P_orc_high: float = 10e5, T_cond: float = 313.15, m_orc: float | None = None, n_orc: float = 310.2,
             eta_pump: float = 0.75, eta_orct: float = 0.86, pinch_eva: float = 3.0,
             T_cw_in: float | None = None, dT_cw: float = 5.0, P_fuel: float = 12e5, name: str = "cgam_orc") -> Plant:
    """Gas turbine (CGAM) -> air preheater -> HRSG (saturated steam) -> R245fa ORC on the remaining exhaust heat.
    n_air, n_orc in mol/s."""
    T_cw_in = T_amb if T_cw_in is None else T_cw_in
    M_air = gp.mw(gp.AIR_CGAM) / 1000
    P_cond = fl.Psat_T("R245fa", T_cond)
    m_orc = n_orc * fl.M("R245fa") if m_orc is None else m_orc
    P_exh = ATM / ((1 - dp_hrsg) * (1 - dp_ap_hot))
    C = lambda i, t, **p: Component(id=i, type=t, params=p)
    S = lambda i, a, b, fluid, **kw: Stream(id=i, **{"from": a, "to": b}, fluid=fluid, **kw)
    comps = [
        C("AC", "compressor", PR=PR, eta_s=eta_c),
        C("AP", "recuperator", T_cold_out=T3, dp_cold=dp_ap_air, dp_hot=dp_ap_hot),
        C("CC", "combustor", T_out=T4, dp=dp_cc, q_loss=q_loss),
        C("GT", "turbine", P_out=P_exh, eta_s=eta_t),
        C("HRSG", "hrsg", P_steam=P_steam, dp_gas=dp_hrsg, econ_approach_K=15.0),
        C("EVA", "orc_evaporator", pinch_K=pinch_eva),
        C("OT", "orc_turbine", P_out=P_cond, eta_s=eta_orct),
        C("COND", "condenser", dT_cw=dT_cw),
        C("PMP", "pump", P_out=P_orc_high, eta_s=eta_pump),
    ]
    streams = [
        S("air", "ext:air", "AC.in", "CGAM_AIR", m_dot=n_air * M_air, fixed=FixedState(T=T_amb, P=ATM)),
        S("s2", "AC.out", "AP.cold_in", "CGAM_AIR"),
        S("s3", "AP.cold_out", "CC.air_in", "CGAM_AIR"),
        S("fuel", "ext:fuel", "CC.fuel_in", "CH4", fixed=FixedState(T=T_amb, P=P_fuel)),
        S("s4", "CC.out", "GT.in", "FLUE"),
        S("s5", "GT.out", "AP.hot_in", "FLUE"),
        S("s6", "AP.hot_out", "HRSG.hot_in", "FLUE"),
        S("s7", "HRSG.hot_out", "EVA.hot_in", "FLUE"),
        S("stack", "EVA.hot_out", "ext:stack", "FLUE"),
        S("fw", "ext:fw", "HRSG.water_in", "Water", m_dot=m_steam, fixed=FixedState(T=T_amb, P=P_steam)),
        S("steam", "HRSG.steam_out", "ext:steam", "Water"),
        S("o1", "COND.hot_out", "PMP.in", "R245fa", m_dot=m_orc, fixed=FixedState(P=P_cond, x=0.0)),
        S("o2", "PMP.out", "EVA.cold_in", "R245fa"),
        S("o3", "EVA.cold_out", "OT.in", "R245fa"),
        S("o4", "OT.out", "COND.hot_in", "R245fa"),
        S("cw_in", "ext:cw", "COND.cold_in", "Water", fixed=FixedState(T=T_cw_in, P=ATM)),
        S("cw_out", "COND.cold_out", "ext:cw_out", "Water"),
    ]
    return Plant(name=name, ambient=Ambient(T0=T_amb, P0=ATM), components=comps, streams=streams)


def dc_chp(*, P_it: float = 20e6, chilled_fraction: float = 0.5, cooling: str = "absorption", prime_mover: str = "turbine",
           rejection: str = "ext", n_air: float = 1000.0, P_el: float | None = None,
           T_amb: float = 298.15, RH: float = 0.6, PR: float = 18.0, eta_c: float = 0.86, eta_t: float = 0.88, T4: float = 1500.0,
           dp_cc: float = 0.04, dp_hx: float = 0.03, q_loss: float = 0.01,
           eta_el: float = 0.43, f_loss: float = 0.15, air_excess: float = 1.8,
           T_supply: float = 291.15, T_return: float = 301.15, P_loop: float = 3e5,
           T_hw: float = 368.15, T_gen_out: float = 358.15, P_hw: float = 5e5,
           dT_rej: float = 8.0, rej_approach: float | None = None, fan_frac: float = 0.015,
           eps_abs: float = 0.27, COP_max: float = 0.8, eps_ec: float = 0.5,
           orc: bool = False, orc_fluid: str = "R245fa", orc_frac: float = 0.7, P_orc_high: float = 12e5, orc_cond_approach: float = 8.0,
           eta_pump: float = 0.75, eta_orct: float = 0.85, T_stack_min: float = 333.15,
           heat_demand_w: float = 0.0, heat_source: str = "exhaust", T_dh_in: float = 323.15, T_dh_out: float = 343.15,
           P_dh: float = 5e5, eps_hp: float = 0.5,
           P_fuel: float = 20e5, name: str = "dc_chp") -> Plant:
    """On-site genset serving a data-centre campus.
    prime_mover='turbine': simple-cycle gas turbine; 'engine': reciprocating gas engine (jacket/radiation heat rejected).
    cooling='absorption': exhaust -> hot-water loop -> single-effect absorption chiller -> chilled loop -> IT load.
    cooling='electric'  : electric chiller on the same chilled loop (parasitic electricity).
    rejection='ext' : chiller rejects to an ideal cooling-water boundary at ambient;
              'dry' / 'wet': closed loop through a dry cooler / evaporative tower (approach to dry-/wet-bulb, fan power).
    heat_demand_w > 0: district heat (water T_dh_in -> T_dh_out) from heat_source='exhaust' (heat exchanger on the exhaust,
              after the chiller's) or 'heat_pump' (lifts the chiller rejection water; needs rejection 'dry'/'wet').
    orc=True: ORC (orc_fluid) on the remaining exhaust; flow = orc_frac of the largest flow that keeps the stack above T_stack_min;
              condensed through its own heat rejector."""
    from ..core.components import rejector_outlet_T
    M_air = gp.mw(gp.AIR_CGAM) / 1000
    C = lambda i, t, **p: Component(id=i, type=t, params=p)
    S = lambda i, a, b, fluid, **kw: Stream(id=i, **{"from": a, "to": b}, fluid=fluid, **kw)
    if heat_demand_w > 0 and heat_source not in ("exhaust", "heat_pump"): raise ValueError("heat_source must be 'exhaust' or 'heat_pump'")
    if heat_demand_w > 0 and heat_source == "heat_pump" and rejection == "ext": raise ValueError("heat_pump needs a dry/wet rejection loop")
    if rejection not in ("ext", "dry", "wet"): raise ValueError("rejection must be 'ext', 'dry' or 'wet'")
    if cooling not in ("absorption", "electric"): raise ValueError("cooling must be 'absorption' or 'electric'")
    chain = (["HX"] if cooling == "absorption" else []) + (["HXD"] if heat_demand_w > 0 and heat_source == "exhaust" else []) + (["EVA"] if orc else [])
    back = {"HX": dp_hx, "HXD": 0.01, "EVA": 0.0}
    P_exh = ATM
    for n_ in chain: P_exh /= (1 - back[n_])
    if prime_mover == "turbine":
        comps = [C("AC", "compressor", PR=PR, eta_s=eta_c), C("CC", "combustor", T_out=T4, dp=dp_cc, q_loss=q_loss),
                 C("GT", "turbine", P_out=P_exh, eta_s=eta_t)]
        streams = [S("air", "ext:air", "AC.in", "CGAM_AIR", m_dot=n_air * M_air, fixed=FixedState(T=T_amb, P=ATM)),
                   S("s2", "AC.out", "CC.air_in", "CGAM_AIR"),
                   S("fuel", "ext:fuel", "CC.fuel_in", "CH4", fixed=FixedState(T=T_amb, P=P_fuel)),
                   S("s4", "CC.out", "GT.in", "FLUE")]
        exh_src = "GT.out"
    elif prime_mover == "engine":
        comps = [C("ENG", "gas_engine", P_el=P_el if P_el is not None else 1.03 * P_it, eta_el=eta_el, f_loss=f_loss,
                   air_excess=air_excess, P_exh=P_exh)]
        streams = [S("air", "ext:air", "ENG.air_in", "CGAM_AIR", fixed=FixedState(T=T_amb, P=ATM)),
                   S("fuel", "ext:fuel", "ENG.fuel_in", "CH4", fixed=FixedState(T=T_amb, P=P_fuel))]
        exh_src = "ENG.exh_out"
    else:
        raise ValueError("prime_mover must be 'turbine' or 'engine'")
    # exhaust chain: prime mover -> [HX] -> [HXD] -> [EVA] -> stack
    src = exh_src
    for k, node in enumerate(chain):
        streams.append(S("s5" if k == 0 else f"s{5 + k}", src, f"{node}.hot_in", "FLUE")); src = f"{node}.hot_out"
    streams.append(S("s5" if not chain else f"s{5 + len(chain)}", src, "ext:stack", "FLUE"))
    comps.append(C("DC", "dc_load", P_it=P_it, chilled_fraction=chilled_fraction, T_return=T_return))
    streams += [S("chw_ret", "DC.out", "CH.evap_in", "Water"),
                S("chw_sup", "CH.evap_out", "DC.in", "Water", fixed=FixedState(T=T_supply, P=P_loop))]
    ap = rej_approach if rej_approach is not None else (8.0 if rejection == "dry" else 5.0)
    T_rej = None if rejection == "ext" else rejector_outlet_T(T_amb, ATM, RH, rejection, ap)
    use_hp = heat_demand_w > 0 and heat_source == "heat_pump"
    if rejection == "ext":
        streams += [S("cw_in", "ext:cw", "CH.rej_in", "Water", fixed=FixedState(T=T_amb, P=ATM)),
                    S("cw_out", "CH.rej_out", "ext:cw_out", "Water")]
    else:
        comps.append(C("REJ", "heat_rejector", approach_K=ap, fan_frac=fan_frac, kind=rejection))
        if use_hp:
            comps.append(C("HP", "heat_pump", Q_sink=heat_demand_w, T_sink_out=T_dh_out, eps=eps_hp))
            streams += [S("rej_hot", "CH.rej_out", "HP.src_in", "Water"), S("rej_hot2", "HP.src_out", "REJ.in", "Water"),
                        S("dh_in", "ext:dh", "HP.sink_in", "Water", fixed=FixedState(T=T_dh_in, P=P_dh)),
                        S("dh_out", "HP.sink_out", "ext:dh_out", "Water")]
        else:
            streams.append(S("rej_hot", "CH.rej_out", "REJ.in", "Water"))
        streams.append(S("rej_cold", "REJ.out", "CH.rej_in", "Water", fixed=FixedState(T=T_rej, P=ATM)))
    if "HXD" in chain:
        import CoolProp.CoolProp as CP
        dh = (CP.PropsSI("H", "T", T_dh_out, "P", P_dh, "Water") - CP.PropsSI("H", "T", T_dh_in, "P", P_dh, "Water"))
        comps.append(C("HXD", "recuperator", T_cold_out=T_dh_out, dp_hot=0.01, useful_heat=True))
        streams += [S("dh_in", "ext:dh", "HXD.cold_in", "Water", m_dot=heat_demand_w / dh, fixed=FixedState(T=T_dh_in, P=P_dh)),
                    S("dh_out", "HXD.cold_out", "ext:dh_out", "Water")]
    if cooling == "absorption":
        comps += [C("HX", "recuperator", T_cold_out=T_hw, dp_hot=dp_hx),
                  C("CH", "absorption_chiller", T_chw_out=T_supply, T_gen_out=T_gen_out, dT_rej=dT_rej, eps=eps_abs, COP_max=COP_max)]
        streams += [S("hw_hot", "HX.cold_out", "CH.gen_in", "Water"),
                    S("hw_ret", "CH.gen_out", "HX.cold_in", "Water", fixed=FixedState(T=T_gen_out, P=P_hw))]
    else:
        comps.append(C("CH", "electric_chiller", T_chw_out=T_supply, dT_rej=dT_rej, eps=eps_ec))
    if orc:
        base = T_rej if T_rej is not None else T_amb
        if orc_fluid not in ORC_T_MAX: raise ValueError(f"orc_fluid must be one of {list(ORC_T_MAX)}")
        T_cond = base + dT_rej + orc_cond_approach; P_cond = fl.Psat_T(orc_fluid, T_cond)
        comps += [C("EVA", "orc_evaporator", pinch_K=3.0, flow_frac=orc_frac, T_stack_min=T_stack_min, T_max_K=ORC_T_MAX[orc_fluid]),
                  C("OT", "orc_turbine", P_out=P_cond, eta_s=eta_orct), C("COND", "condenser", dT_cw=dT_rej),
                  C("PMP", "pump", P_out=P_orc_high, eta_s=eta_pump)]
        streams += [S("o1", "COND.hot_out", "PMP.in", orc_fluid, fixed=FixedState(P=P_cond, x=0.0)),
                    S("o2", "PMP.out", "EVA.cold_in", orc_fluid), S("o3", "EVA.cold_out", "OT.in", orc_fluid),
                    S("o4", "OT.out", "COND.hot_in", orc_fluid)]
        if rejection == "ext":
            streams += [S("ocw_in", "ext:ocw", "COND.cold_in", "Water", fixed=FixedState(T=T_amb, P=ATM)),
                        S("ocw_out", "COND.cold_out", "ext:ocw_out", "Water")]
        else:
            comps.append(C("REJ2", "heat_rejector", approach_K=ap, fan_frac=fan_frac, kind=rejection))
            streams += [S("ocw_hot", "COND.cold_out", "REJ2.in", "Water"),
                        S("ocw_cold", "REJ2.out", "COND.cold_in", "Water", fixed=FixedState(T=T_rej, P=ATM))]
    return Plant(name=name, ambient=Ambient(T0=T_amb, P0=ATM, RH=RH), components=comps, streams=streams)


def size_generation(build, margin: float = 1.03, **kw) -> Plant:
    """Size the prime mover (air flow for a turbine, rated P_el for an engine) so that net electricity (generation
    minus plant parasitics plus any ORC output) equals margin * IT load. Secant iteration on full plant solves."""
    from ..core.solve import solve
    engine = kw.get("prime_mover") == "engine"; var = "P_el" if engine else "n_air"
    P_it = kw.get("P_it", 20e6); target = margin * P_it
    x = target if engine else P_it / 9000.0              # start at the right order of magnitude
    last = "no feasible sizing point"; r = None
    for _ in range(8):                                    # find a feasible starting point (small plants may lack exhaust heat)
        r = solve(build(**{var: x}, **kw))
        if r.ok: break
        last = r.message; x *= 1.6
    else:
        raise RuntimeError(f"sizing run failed: {last}")
    f = r.kpis["W_net_w"] - target; xp, fp = None, None
    for _ in range(14):
        if abs(f) <= 2e-5 * P_it: return build(**{var: x}, **kw)
        step = (x - xp) / (f - fp) if xp is not None and f != fp else x / target
        xn = x - f * step
        for _ in range(5):                               # a full step can land in an infeasible region: shorten it
            r2 = solve(build(**{var: max(xn, 1e-3 * x)}, **kw))
            if r2.ok: break
            xn = 0.5 * (x + xn)
        else:
            raise RuntimeError(f"sizing run failed: {r2.message}")
        xp, fp, x, f = x, f, max(xn, 1e-3 * x), r2.kpis["W_net_w"] - target
    raise RuntimeError("sizing did not converge")
