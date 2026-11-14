"""Ground truth for Audit: exergy-destruction ranking, the plant summary the model sees, and the effect of each change."""
from __future__ import annotations
from ..core.schema import Plant, Site, SolveResult
from ..core.solve import solve
from ..verify.verifiers import run_all
from . import changes as ch


def ranking(plant: Plant, res: SolveResult) -> list[tuple[str, float]]:
    kind = {c.id: c.type for c in plant.components}
    return sorted(((b.component, b.exergy_destruction_w) for b in res.balances if kind[b.component] != "dc_load"), key=lambda t: -t[1])


def first_law_ranking(plant: Plant, res: SolveResult) -> list[tuple[str, float]]:
    """What a first-law analysis would blame: heat dissipated to ambient by each component."""
    kind = {c.id: c.type for c in plant.components}
    return sorted(((b.component, b.Q_loss_w) for b in res.balances if kind[b.component] != "dc_load"), key=lambda t: -t[1])


def summary_text(plant: Plant, res: SolveResult) -> str:
    """Energy-level view of the solved plant. Exergy destruction is deliberately not shown."""
    k = res.kpis; st = {s.stream: s for s in res.states}; lines = []
    kinds = {c.id: c.type for c in plant.components}
    lines.append(f"KPI fuel_MW={k['Q_fuel_w']/1e6:.2f} net_electricity_MW={k['W_net_w']/1e6:.2f} cooling_MW={k.get('Q_cooling_w', 0)/1e6:.2f} "
                 f"heat_MW={k.get('Q_heat_w', 0)/1e6:.2f} eta_II={k.get('eta_II', 0):.3f}")
    for b in res.balances:
        ts = [st[s.id].T for s in plant.streams if s.id in st and (s.src.startswith(b.component + ".") or s.dst.startswith(b.component + "."))]
        lines.append(f"{b.component} {kinds[b.component]} W_MW={b.W_w/1e6:.2f} Q_MW={b.Q_w/1e6:.2f} loss_MW={b.Q_loss_w/1e6:.2f} "
                     f"T_min_K={min(ts):.0f} T_max_K={max(ts):.0f}")
    stack = next((s for s in plant.streams if s.dst == "ext:stack"), None)
    if stack: lines.append(f"STACK T_K={st[stack.id].T:.0f} m_kg_s={st[stack.id].m_dot:.1f}")
    return "\n".join(lines)


def change_effect(plant: Plant, site: Site | None, cid: str, param: str, value: float, base_net_w: float, base_fuel_w: float) -> dict:
    """Apply the change, resize the prime mover to the same net electricity, solve and verify."""
    if not ch.check_change(plant, cid, param, value): return {"valid": False, "reason": "change not allowed", "delta_w": None}
    out = ch.resize(ch.apply_change(plant, cid, param, value), base_net_w)
    if out is None: return {"valid": False, "reason": "cannot be solved", "delta_w": None}
    p2, r2 = out; rep = run_all(p2, r2, site)
    if not rep.valid: return {"valid": False, "reason": "failed: " + ",".join(rep.failed()), "delta_w": None}
    return {"valid": True, "reason": "", "delta_w": r2.kpis["Q_fuel_w"] - base_fuel_w, "fuel_w": r2.kpis["Q_fuel_w"]}


def truth(plant: Plant, site: Site | None) -> dict | None:
    res = solve(plant)
    if not res.ok or not run_all(plant, res, site).valid: return None
    base_net, base_fuel = res.kpis["W_net_w"], res.kpis["Q_fuel_w"]; table = []
    for cid, p, v in ch.candidates(plant):
        e = change_effect(plant, site, cid, p, v, base_net, base_fuel)
        table.append((ch.edit_text(cid, p, v), e["delta_w"]))
    good = [t for t in table if t[1] is not None]
    best = min(good, key=lambda t: t[1]) if good else None
    return {"result": res, "fuel_w": base_fuel, "net_w": base_net, "ranking": ranking(plant, res), "table": table, "best": best,
            "best_gain_w": -best[1] if best else 0.0}
