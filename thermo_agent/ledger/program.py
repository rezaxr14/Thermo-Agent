"""Exergy ledger of a solved plant: stream exergy lines, component destruction lines, and their evaluation.

    T0 = 298.15
    E[s5]  = m[s5]*(h[s5] - T0*s[s5] - g0[s5] + ech[s5])/1e6      stream exergy flow, MW
    ED[HX] = E[s5] + E[hw_ret] - E[s6] - E[hw_hot]                component destruction, MW (- W[c] when the component has work)
g0 = h0 - T0*s0 is the dead-state reference of the stream's fluid, so the physical exergy is h - T0 s - g0."""
from __future__ import annotations
import math
from ..core.schema import Plant, SolveResult
from .expr import evaluate, ExprError

STREAM_TEMPLATE = "m[{x}]*(h[{x}] - T0*s[{x}] - g0[{x}] + ech[{x}])/1e6"
W_MIN_MW = 1e-9


def _render(terms) -> str:
    out = ""
    for i, (sg, fac, kind, ident) in enumerate(terms):
        sym = f"{kind}[{ident}]" if fac == 1 else f"{fac:g}*{kind}[{ident}]"
        out += (("-" if sg < 0 else "") if i == 0 else (" - " if sg < 0 else " + ")) + sym
    return out


def component_terms(topo_c: dict, has_work: bool, cid: str):
    terms = [(1, 1, "E", s) for s in topo_c["in"]] + [(-1, 1, "E", s) for s in topo_c["out"]]
    if has_work: terms.append((-1, 1, "W", cid))
    return terms


def build(plant: Plant, res: SolveResult) -> dict:
    """Correct ledger and everything needed to evaluate and judge edits (JSON-serialisable)."""
    T0 = plant.ambient.T0
    st = {s.stream: s for s in res.states}
    tables = {"m": {k: v.m_dot for k, v in st.items()}, "h": {k: v.h for k, v in st.items()}, "s": {k: v.s for k, v in st.items()},
              "g0": {k: v.h - T0 * v.s - v.e_ph for k, v in st.items()}, "ech": {k: v.e_ch for k, v in st.items()}, "W": {}}
    for b in res.balances:
        if abs(b.W_w) / 1e6 > W_MIN_MW: tables["W"][b.component] = b.W_w / 1e6
    topo = {c.id: {"type": c.type, "in": [s.id for s in plant.streams if s.dst.startswith(c.id + ".")],
                   "out": [s.id for s in plant.streams if s.src.startswith(c.id + ".")]} for c in plant.components}
    lines = {"T0": repr(T0)}
    for s in plant.streams: lines[f"E[{s.id}]"] = STREAM_TEMPLATE.format(x=s.id)
    for c in plant.components: lines[f"ED[{c.id}]"] = _render(component_terms(topo[c.id], c.id in tables["W"], c.id))
    truth_E = {k: v.m_dot * (v.e_ph + v.e_ch) / 1e6 for k, v in st.items()}
    truth_ED = {b.component: b.exergy_destruction_w / 1e6 for b in res.balances}
    return {"consts_true": {"T0": T0}, "tables": tables, "topo": topo, "stream_order": [s.id for s in plant.streams],
            "ext_in": [s.id for s in plant.streams if s.src.startswith("ext:")], "ext_out": [s.id for s in plant.streams if s.dst.startswith("ext:")],
            "lines": lines, "truth_E": truth_E, "truth_ED": truth_ED, "e_fuel_mw": res.kpis.get("E_fuel_w", 0.0) / 1e6}


def run(lines: dict, data: dict) -> dict:
    """Evaluate a (possibly broken) ledger. Returns T0, E, ED, closure residual (MW, None if any line fails) and per-line errors."""
    errors, consts, E, ED = {}, {}, {}, {}
    ctx = {"consts": consts, "tables": {**data["tables"], "E": E}}
    try: consts["T0"] = evaluate(lines["T0"], {"consts": {}, "tables": {}})
    except (ExprError, KeyError) as e: errors["T0"] = str(e)
    for sid in data["stream_order"]:
        key = f"E[{sid}]"
        try:
            if "T0" in errors: raise ExprError("T0 is not available")
            E[sid] = evaluate(lines[key], ctx)
        except (ExprError, KeyError) as e: errors[key] = str(e) if isinstance(e, ExprError) else "missing line"
    for cid in data["topo"]:
        key = f"ED[{cid}]"
        try: ED[cid] = evaluate(lines[key], ctx)
        except (ExprError, KeyError) as e: errors[key] = str(e) if isinstance(e, ExprError) else "missing line"
    closure = None
    if not errors:
        ext = sum(E[i] for i in data["ext_in"]) - sum(E[o] for o in data["ext_out"]) - sum(data["tables"]["W"].values())
        closure = sum(ED.values()) - ext
    return {"T0": consts.get("T0"), "E": E, "ED": ED, "closure": closure, "errors": errors}


def matches_truth(result: dict, data: dict, key: str, tol: float = 1e-6) -> bool:
    """Does line `key` evaluate to the true value? (Equivalent expressions are accepted.)"""
    if key in result["errors"]: return False
    if key == "T0": return result["T0"] is not None and abs(result["T0"] - data["consts_true"]["T0"]) <= tol * data["consts_true"]["T0"]
    kind, ident = key[:-1].split("[")
    got, want = (result["E"].get(ident), data["truth_E"][ident]) if kind == "E" else (result["ED"].get(ident), data["truth_ED"][ident])
    return got is not None and abs(got - want) <= tol * max(1.0, abs(want))
