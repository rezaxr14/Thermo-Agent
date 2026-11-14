"""Plant solver: sequential-modular, with tear streams for recycle loops.

A stream whose source is a component port but that carries a `fixed` state is a tear stream: its state is
preset so downstream components can start, and the producing component's result must reproduce it.
"""
from __future__ import annotations
from typing import Optional
from .schema import Plant, SolveResult, StatePoint, ComponentBalance, ErrorCode
from . import gasprops as gp
from .props import State, SolveError, check_fluid, comp_of, M_of, state_TP, state_hP, state_sat
from .components import SOLVERS, REQUIRED_PORTS, PORT_CLASS
from .props import is_gas
from .exergy import Env, e_molar


class Ctx:
    def __init__(self, plant: Plant, env: Env):
        self.plant, self.env = plant, env
        self.streams = {s.id: s for s in plant.streams}
        self.port: dict[tuple[str, str], str] = {}
        for s in plant.streams:
            for end in (s.src, s.dst):
                if not end.startswith("ext:"):
                    c, p = end.split(".", 1)
                    if (c, p) in self.port:
                        raise SolveError("E_OVER_SPECIFIED", f"port {c}.{p} connected to more than one stream")
                    self.port[(c, p)] = s.id
        self.states: dict[str, State] = {}
        self.work: dict[str, float] = {}; self.Q: dict[str, float] = {}; self.qloss: dict[str, float] = {}
        self.aux: dict[str, dict] = {}
        self.tear: dict[str, State] = {}

    def sid(self, comp, port):
        if (comp, port) not in self.port:
            raise SolveError("E_DISCONNECTED", f"{comp}.{port} is not connected to any stream")
        return self.port[(comp, port)]

    def inp(self, comp, port) -> Optional[State]:
        return self.states.get(self.sid(comp, port))

    def put(self, comp, port, st: State):
        sid = self.sid(comp, port); st.stream = sid
        if sid in self.tear:
            t = self.tear[sid]
            if abs(st.P - t.P) > 1e-3 * t.P or abs(st.T - t.T) > 0.5:
                raise SolveError("E_NOT_CONVERGED", f"tear stream {sid}: computed T={st.T:.2f} K P={st.P:.4g} Pa differs from specified T={t.T:.2f} K P={t.P:.4g} Pa")
            st.n = t.n if t.n is not None else st.n
        self.states[sid] = st


def _spec_state(s, env: Env) -> State:
    f = s.fixed
    n = None
    if s.m_dot is not None:
        n = s.m_dot / M_of(s.fluid, s.composition)
    if f is None or f.P is None:
        raise SolveError("E_UNDER_SPECIFIED", f"stream {s.id}: fixed pressure required")
    if f.T is not None:
        return state_TP(s.id, s.fluid, s.composition, f.T, f.P, n)
    if f.x is not None:
        return state_sat(s.id, s.fluid, f.P, f.x, n)
    if f.h is not None:
        return state_hP(s.id, s.fluid, s.composition, f.h * M_of(s.fluid, s.composition), f.P, n)
    raise SolveError("E_UNDER_SPECIFIED", f"stream {s.id}: need two of T,P,x,h")


def _fail(code, msg) -> SolveResult:
    return SolveResult(ok=False, error=ErrorCode(code), message=msg)


def solve(plant: Plant, env: Env | None = None) -> SolveResult:
    env = env or Env(plant.ambient.T0, plant.ambient.P0)
    try:
        return _solve(plant, env)
    except SolveError as e:
        return _fail(e.code, e.msg)
    except (ValueError, ZeroDivisionError, OverflowError, KeyError, TypeError, AttributeError) as e:
        return _fail("E_PROP_RANGE", f"{type(e).__name__}: {e}")


def _solve(plant: Plant, env: Env) -> SolveResult:
    for s in plant.streams: check_fluid(s.fluid)
    ctx = Ctx(plant, env)
    for c in plant.components:
        if c.type not in SOLVERS:
            raise SolveError("E_UNKNOWN_COMPONENT", f"component type '{c.type}' has no model")
        for p in REQUIRED_PORTS[c.type]:
            sid = ctx.sid(c.id, p); want = PORT_CLASS.get((c.type, p)); have = "gas" if is_gas(ctx.streams[sid].fluid) else "liquid"
            if want and want != have:
                raise SolveError("E_FLUID_MISMATCH", f"{c.id}.{p} expects a {want} stream but {sid} carries {ctx.streams[sid].fluid}")
    for s in plant.streams:
        if s.src.startswith("ext:"):
            ctx.states[s.id] = _spec_state(s, env)
        elif s.fixed is not None:
            ctx.tear[s.id] = ctx.states[s.id] = _spec_state(s, env)
    pending = list(plant.components)
    while pending:
        nxt = [c for c in pending if not SOLVERS[c.type](ctx, c)]
        if len(nxt) == len(pending):
            raise SolveError("E_UNDER_SPECIFIED", "cannot order components; unsolved: " + ", ".join(c.id for c in nxt))
        pending = nxt
    for sid, t in ctx.tear.items():                       # re-check closure of tear streams
        st = ctx.states[sid]
        if abs(st.P - t.P) > 1e-3 * t.P or abs(st.T - t.T) > 0.5:
            raise SolveError("E_NOT_CONVERGED", f"tear stream {sid} does not close")
    return _report(plant, env, ctx)


def _report(plant, env, ctx) -> SolveResult:
    st = ctx.states; e = {sid: e_molar(s, env) for sid, s in st.items()}
    E = {sid: s.n * (e[sid][0] + e[sid][1]) for sid, s in st.items()}
    points = []
    for sid, s in st.items():
        M = s.M
        points.append(StatePoint(stream=sid, T=s.T, P=s.P, h=s.h / M, s=s.s / M, m_dot=s.n * M, x=s.x,
                                 e_ph=e[sid][0] / M, e_ch=e[sid][1] / M))
    balances = []; ED_tot = 0.0
    for c in plant.components:
        ins = [sid for (cc, p), sid in ctx.port.items() if cc == c.id and ctx.streams[sid].dst.startswith(c.id + ".")]
        outs = [sid for (cc, p), sid in ctx.port.items() if cc == c.id and ctx.streams[sid].src.startswith(c.id + ".")]
        W = ctx.work.get(c.id, 0.0); Ql = ctx.qloss.get(c.id, 0.0)
        mres = sum(st[i].m for i in ins) - sum(st[o].m for o in outs)
        eres = sum(st[i].n * st[i].h for i in ins) - sum(st[o].n * st[o].h for o in outs) - W - Ql
        ED = sum(E[i] for i in ins) - sum(E[o] for o in outs) - W
        ED_tot += ED
        balances.append(ComponentBalance(component=c.id, Q_w=ctx.Q.get(c.id, 0.0), W_w=W, mass_residual=mres,
                                         energy_residual=eres, exergy_destruction_w=ED, Q_loss_w=Ql,
                                         min_approach_K=ctx.aux.get(c.id, {}).get("min_approach")))
    ext_in = [s.id for s in plant.streams if s.src.startswith("ext:")]
    ext_out = [s.id for s in plant.streams if s.dst.startswith("ext:")]
    ctype = {c.id: c.type for c in plant.components}
    W_all = sum(ctx.work.values())
    closure = ED_tot - (sum(E[i] for i in ext_in) - sum(E[o] for o in ext_out) - W_all)
    P_it = -sum(w for k, w in ctx.work.items() if ctype[k] == "dc_load")
    W_net = sum(w for k, w in ctx.work.items() if ctype[k] != "dc_load")       # generation minus plant parasitics
    Q_fuel = sum(a.get("Q_fuel", 0.0) for a in ctx.aux.values())
    hr = [k for k, t in ctype.items() if t == "hrsg"]
    useful_hx = [c.id for c in plant.components if c.type == "recuperator" and c.params.get("useful_heat")]
    hp = [k for k, t in ctype.items() if t == "heat_pump"]
    ch = [k for k, t in ctype.items() if t in ("absorption_chiller", "electric_chiller")]
    Q_heat = sum(ctx.Q[k] for k in hr + hp + useful_hx); Q_cool = sum(ctx.Q[k] for k in ch)
    E_steam = sum(E[ctx.sid(h, "steam_out")] - E[ctx.sid(h, "water_in")] for h in hr)
    E_heat = (sum(E[ctx.sid(h, "sink_out")] - E[ctx.sid(h, "sink_in")] for h in hp)
              + sum(E[ctx.sid(h, "cold_out")] - E[ctx.sid(h, "cold_in")] for h in useful_hx))
    E_cool = sum(E[ctx.sid(h, "evap_out")] - E[ctx.sid(h, "evap_in")] for h in ch)
    E_fuel = sum(E[i] for i in ext_in if st[i].fluid == "CH4")
    k = {"W_net_w": W_net, "Q_fuel_w": Q_fuel, "Q_useful_w": Q_heat + Q_cool, "Q_heat_w": Q_heat, "Q_cooling_w": Q_cool,
         "P_it_w": P_it, "E_fuel_w": E_fuel, "ED_total_w": ED_tot,
         "ED_plant_w": sum(b.exergy_destruction_w for b in balances if ctype[b.component] != "dc_load"), "closure_residual_w": closure,
         "T_max_K": max(s.T for s in st.values()), "T0_K": env.T0}
    k["water_use_kg_s"] = sum(a.get("water_kg_s", 0.0) for a in ctx.aux.values())
    for cid, a in ctx.aux.items():
        if "COP" in a: k[f"COP_{cid}"] = a["COP"]; k[f"COPrev_{cid}"] = a["COP_rev"]
    if Q_fuel > 0:
        k["eta_I"] = (W_net + Q_heat + Q_cool) / Q_fuel
        k["eta_I_power"] = W_net / Q_fuel
    if E_fuel > 0:
        k["eta_II"] = (W_net + E_steam + E_heat + E_cool) / E_fuel
    ev = [c.id for c in plant.components if c.type == "orc_evaporator"]
    ot = [c.id for c in plant.components if c.type == "orc_turbine"]
    if ev and ot:
        Wo = sum(ctx.work[c.id] for c in plant.components if c.type in ("orc_turbine", "pump"))
        k["W_orc_net_w"] = Wo; k["Q_evap_orc_w"] = sum(ctx.Q[c] for c in ev)
        k["T_orc_hot_K"] = max(st[ctx.sid(c, "cold_out")].T for c in ev)
        cd = [c.id for c in plant.components if c.type == "condenser"]
        if cd: k["T_cond_K"] = ctx.aux[cd[0]]["T_sat"]
    return SolveResult(ok=True, states=points, balances=balances, kpis=k)
