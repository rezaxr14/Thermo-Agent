"""Trace-weighted fuel of a design.

The trace's IT load is split into quantile bins; the design is evaluated at each bin's median load (and median coolant and,
if present, ambient temperatures) and weighted by the time spent in the bin. IDEALISATION: the plant is re-sized to every bin,
so efficiency does not depend on load (ideal part-load; real units lose efficiency below rated output, and modular gensets
switch units instead). Treat the result as a lower bound on fuel, useful to compare designs, not to predict an annual bill.

    python -m thermo_agent.design.trace_eval TRACE --columns it_power=P t_supply=Ts t_return=Tr [ambient=Ta] --power-unit MW --temp-unit C \\
        --chilled-fraction 0.5 --step-hours 0.1667 [--spec-json '{...}'] [--top 5]"""
from __future__ import annotations
import argparse, json
import numpy as np
from ..core.schema import Site, Ambient
from .build import evaluate_design
from .search import enumerate_designs


def bin_trace(df, n_bins: int = 8, ambient_col: str | None = None) -> list[dict]:
    """df needs it_power_w, t_supply_k, t_return_k (see datasets.frontier.load_frontier); optional ambient column in kelvin."""
    edges = np.quantile(df.it_power_w, np.linspace(0, 1, n_bins + 1)); bins = []
    idx = np.clip(np.searchsorted(edges, df.it_power_w.values, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sub = df[idx == b]
        if len(sub) == 0: continue
        bins.append({"samples": int(len(sub)), "it_load_w": float(sub.it_power_w.median()), "t_supply_k": float(sub.t_supply_k.median()),
                     "t_return_k": float(sub.t_return_k.median()), "ambient_k": float(sub[ambient_col].median()) if ambient_col else None})
    return bins


def bin_site(template: Site, b: dict, chilled_fraction: float) -> Site:
    """The site seen during one load bin."""
    upd = {"it_load_w": b["it_load_w"], "cooling_load_w": b["it_load_w"] * chilled_fraction, "t_supply_k": b["t_supply_k"], "t_return_k": b["t_return_k"]}
    if template.heat_demand_w > 0: upd["heat_demand_w"] = template.heat_demand_w * b["it_load_w"] / template.it_load_w
    if b["ambient_k"] is not None: upd["ambient"] = Ambient(T0=b["ambient_k"], RH=template.ambient.RH)
    return template.model_copy(update=upd)


def trace_fuel(template: Site, spec: dict, bins: list[dict], chilled_fraction: float, step_hours: float) -> dict:
    """Fuel energy over the trace for one design; infeasible bins are counted by time share and excluded from the energy."""
    fuel_mwh = it_mwh = infeasible = 0.0; total = sum(b["samples"] for b in bins); per_bin = []
    for b in bins:
        e = evaluate_design(bin_site(template, b, chilled_fraction), spec); hours = b["samples"] * step_hours
        per_bin.append({"it_mw": round(b["it_load_w"] / 1e6, 2), "share": round(b["samples"] / total, 3), "valid": e.valid, "fuel_mw": round(e.fuel_w / 1e6, 3) if e.fuel_w else None})
        if e.fuel_w is None: infeasible += b["samples"] / total; continue
        fuel_mwh += e.fuel_w / 1e6 * hours; it_mwh += b["it_load_w"] / 1e6 * hours
    return {"fuel_mwh": round(fuel_mwh, 2), "fuel_per_it_energy": round(fuel_mwh / it_mwh, 4) if it_mwh else None,
            "infeasible_time_share": round(infeasible, 3), "per_bin": per_bin,
            "idealisation": "plant re-sized to each load bin: ideal part-load efficiency; a lower bound on fuel"}


def rank_catalog(template: Site, bins: list[dict], chilled_fraction: float, step_hours: float) -> list[tuple[dict, dict]]:
    """All default catalog configurations, best (least fuel per IT energy among fully feasible ones) first."""
    out = [(s, trace_fuel(template, s, bins, chilled_fraction, step_hours)) for s, _ in enumerate_designs(template)]
    key = lambda t: (t[1]["infeasible_time_share"], t[1]["fuel_per_it_energy"] if t[1]["fuel_per_it_energy"] is not None else 1e9)
    return sorted(out, key=key)


def main():
    from ..datasets.frontier import load_frontier
    ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--columns", nargs="+", required=True, help="it_power=<col> t_supply=<col> t_return=<col> [ambient=<col>]")
    ap.add_argument("--power-unit", default="MW"); ap.add_argument("--temp-unit", default="C"); ap.add_argument("--chilled-fraction", type=float, default=0.5)
    ap.add_argument("--step-hours", type=float, required=True); ap.add_argument("--bins", type=int, default=8); ap.add_argument("--spec-json"); ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--scan-units", type=int, help="with --spec-json: modular gensets, fuel for 1..N identical units (see design/modular.py)"); ap.add_argument("--capacity-margin", type=float, default=1.0)
    a = ap.parse_args(); cols = dict(c.split("=", 1) for c in a.columns)
    amb = cols.pop("ambient", None)
    df = load_frontier(a.trace, cols, a.power_unit, a.temp_unit)
    if amb:
        import pandas as pd
        raw = pd.read_csv(a.trace) if a.trace.endswith(".csv") else pd.read_excel(a.trace)
        df["ambient_k"] = raw.loc[df.index, amb] + (273.15 if a.temp_unit == "C" else 0.0)
    bins = bin_trace(df, a.bins, "ambient_k" if amb else None)
    P = float(df.it_power_w.max()); tmpl = Site(it_load_w=P, cooling_load_w=P * a.chilled_fraction, grid_import_allowed=False, t_supply_k=float(df.t_supply_k.median()), t_return_k=float(df.t_return_k.median()))
    if a.scan_units:
        if not a.spec_json: raise SystemExit("--scan-units needs --spec-json")
        from .modular import scan_units
        for r in scan_units(tmpl, json.loads(a.spec_json), bins, a.chilled_fraction, a.step_hours, range(1, a.scan_units + 1), a.capacity_margin):
            print(json.dumps({k: r.get(k) for k in ("n_units", "fuel_mwh", "ideal_fuel_mwh", "modularity_penalty", "mean_units_online", "surplus_electricity_mwh", "infeasible_time_share")}))
    elif a.spec_json: print(json.dumps(trace_fuel(tmpl, json.loads(a.spec_json), bins, a.chilled_fraction, a.step_hours), indent=1))
    else:
        for s, r in rank_catalog(tmpl, bins, a.chilled_fraction, a.step_hours)[: a.top]:
            print(json.dumps({"spec": s, **{k: r[k] for k in ("fuel_mwh", "fuel_per_it_energy", "infeasible_time_share")}}))


if __name__ == "__main__":
    main()
