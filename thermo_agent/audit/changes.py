"""Allowed single-parameter changes for Audit, and a resizer that holds net electricity constant.

A change is `SET <component>.<param> = <value>` on a whitelisted parameter, within a small window around the current value
(a set-point or loss-parameter change, not a redesign). Tear-coupled values (hot-water and rejector temperatures) are excluded
because editing one side of a loop breaks its closure."""
from __future__ import annotations
from ..core.schema import Plant
from ..core.solve import solve

# (component type, param): kind. 'abs' = within +-0.05 of current (and <= 0.95); 'rel' = within [0.8, 1.25] x current
# Component efficiencies (eta_*, eps) are deliberately excluded: raising the efficiency of the worst component is a
# trivial answer. Changes are operating set-points and heat-exchanger / loss parameters.
RULES = {("compressor", "PR"): "rel", ("combustor", "T_out"): "rel", ("combustor", "dp"): "rel", ("combustor", "q_loss"): "rel",
         ("recuperator", "dp_hot"): "rel", ("hrsg", "dp_gas"): "rel", ("orc_evaporator", "pinch_K"): "rel", ("orc_evaporator", "flow_frac"): "rel",
         ("pump", "P_out"): "rel", ("heat_rejector", "fan_frac"): "rel", ("gas_engine", "air_excess"): "rel", ("gas_engine", "f_loss"): "rel",
         ("absorption_chiller", "dT_rej"): "rel", ("electric_chiller", "dT_rej"): "rel"}
HARD_LIMITS = {"T_out": (1250.0, 1650.0), "flow_frac": (0.1, 1.0), "air_excess": (1.2, 4.0), "eta_s": (0.5, 0.95), "eta_el": (0.3, 0.5), "eps": (0.05, 0.95)}


def allowed(ctype: str, param: str, cur: float, new: float) -> bool:
    kind = RULES.get((ctype, param))
    if kind is None or not (new == new) or new <= 0: return False
    lo, hi = HARD_LIMITS.get(param, (0.0, float("inf")))
    if not (lo <= new <= hi): return False
    return (cur - 0.05 <= new <= cur + 0.05) if kind == "abs" else (0.8 * cur <= new <= 1.25 * cur)


def _round(x: float) -> float:
    return float(f"{x:.6g}")


def candidates(plant: Plant) -> list[tuple[str, str, float]]:
    out = []
    for c in plant.components:
        for p, v in c.params.items():
            kind = RULES.get((c.type, p))
            if kind is None or isinstance(v, bool) or not isinstance(v, (int, float)): continue
            steps = [v + 0.02, v - 0.02] if kind == "abs" else [v * 0.9, v * 1.1]
            for nv in steps:
                nv = _round(nv)
                if nv != v and allowed(c.type, p, v, nv): out.append((c.id, p, nv))
    return out


def edit_text(cid: str, param: str, value: float) -> str:
    return f"SET {cid}.{param} = {value!r}"


def apply_change(plant: Plant, cid: str, param: str, value: float) -> Plant:
    d = plant.model_dump(by_alias=True, mode="json")
    comp = next(c for c in d["components"] if c["id"] == cid); comp["params"][param] = value
    return Plant.model_validate(d)


def check_change(plant: Plant, cid: str, param: str, value: float) -> bool:
    comp = next((c for c in plant.components if c.id == cid), None)
    if comp is None or param not in comp.params or isinstance(comp.params[param], bool): return False
    return allowed(comp.type, param, float(comp.params[param]), float(value))


def _scale_get(plant: Plant):
    for c in plant.components:
        if c.type == "gas_engine": return ("P_el", c.id, c.params["P_el"])
    s = next(s for s in plant.streams if s.id == "air"); return ("air", s.id, s.m_dot)


def _scale_set(plant: Plant, x: float) -> Plant:
    d = plant.model_dump(by_alias=True, mode="json"); kind, ident, _ = _scale_get(plant)
    if kind == "P_el": next(c for c in d["components"] if c["id"] == ident)["params"]["P_el"] = x
    else: next(s for s in d["streams"] if s["id"] == ident)["m_dot"] = x
    return Plant.model_validate(d)


def resize(plant: Plant, target_net_w: float, tol: float = 2e-5):
    """Scale the prime mover so net electricity equals target. Returns (plant, SolveResult) or None if it cannot be solved."""
    x = _scale_get(plant)[2]; p = plant; r = solve(p)
    if not r.ok: return None
    f = r.kpis["W_net_w"] - target_net_w; xp = fp = None
    for _ in range(12):
        if abs(f) <= tol * target_net_w: return p, r
        step = (x - xp) / (f - fp) if xp is not None and f != fp else x / max(target_net_w, 1.0)
        xn = x - f * step
        for _ in range(5):
            p2 = _scale_set(plant, max(xn, 1e-3 * x)); r2 = solve(p2)
            if r2.ok: break
            xn = 0.5 * (x + xn)
        else: return None
        xp, fp, x, f, p, r = x, f, max(xn, 1e-3 * x), r2.kpis["W_net_w"] - target_net_w, p2, r2
    return None
