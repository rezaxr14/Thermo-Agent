import numpy as np, pandas as pd, pytest
from thermo_agent.core.schema import Site
from thermo_agent.design.spec import default_spec
from thermo_agent.design.trace_eval import bin_trace, trace_fuel
from thermo_agent.design.modular import make_plans, modular_fuel, scan_units

rng = np.random.default_rng(1); N = 400
DF = pd.DataFrame({"it_power_w": rng.uniform(8e6, 30e6, N), "t_supply_k": 291.0 + rng.normal(0, 0.2, N), "t_return_k": 301.0 + rng.normal(0, 0.2, N)})
TMPL = Site(it_load_w=30e6, cooling_load_w=15e6, grid_import_allowed=False, t_supply_k=291.0, t_return_k=301.0)
BINS = bin_trace(DF, 6)
SPEC = default_spec("engine", "absorption", "wet")
PLANS = make_plans(TMPL, SPEC, BINS, 0.5)

def test_one_unit_runs_at_rated_load_in_every_bin_so_fuel_is_constant_and_surplus_large():
    r = modular_fuel(PLANS, 1, 1 / 6)
    fuels = {b["fuel_mw"] for b in r["per_bin"]}; assert len(fuels) == 1 and r["mean_units_online"] == 1.0
    assert r["modularity_penalty"] > 0.2 and r["surplus_electricity_mwh"] > 0

def test_units_online_follow_the_load_by_ceiling():
    r = modular_fuel(PLANS, 4, 1 / 6); online = [b["online"] for b in r["per_bin"]]
    assert online == sorted(online) and max(online) == 4 and min(online) < max(online)
    top = max(PLANS, key=lambda p: p["x_need"]); x_unit = top["x_need"] / 4
    for p, b in zip(PLANS, r["per_bin"]): assert b["online"] == max(1, int(np.ceil(p["x_need"] / x_unit - 1e-9)))

def test_more_units_cut_the_modularity_penalty_toward_the_ideal_bound():
    t = scan_units(TMPL, SPEC, BINS, 0.5, 1 / 6, units=[1, 2, 4, 8, 16])
    pen = [r["modularity_penalty"] for r in t]; assert pen[-1] < pen[0] and pen[-1] < pen[1] and all(p >= -1e-9 for p in pen)
    ideal = trace_fuel(TMPL, SPEC, BINS, 0.5, 1 / 6)["fuel_mwh"]; assert all(r["fuel_mwh"] >= ideal * 0.999 for r in t) and t[-1]["fuel_mwh"] < t[0]["fuel_mwh"]

def test_every_bin_plant_passes_the_verifiers_and_curtailed_power_is_never_negative():
    r = modular_fuel(PLANS, 3, 1 / 6); assert r["infeasible_time_share"] == 0 and all(b["valid"] for b in r["per_bin"]) and r["surplus_electricity_mwh"] >= 0

def test_turbine_designs_work_too_and_infeasible_bins_are_reported():
    plans = make_plans(TMPL, default_spec("turbine", "absorption", "wet"), BINS, 0.5); r = modular_fuel(plans, 3, 1 / 6)
    assert r["infeasible_time_share"] == 0 and r["fuel_mwh"] > 0
    hot = DF.copy(); hot["ambient_k"] = 312.0; hb = bin_trace(hot, 3, "ambient_k")
    rd = modular_fuel(make_plans(TMPL, default_spec("engine", "absorption", "dry"), hb, 0.5), 2, 1 / 6); assert rd["infeasible_time_share"] > 0


def test_command_line_scan_runs_on_a_spreadsheet(tmp_path, monkeypatch, capsys):
    import json, sys
    from thermo_agent.design import trace_eval
    raw = pd.DataFrame({"P": DF.it_power_w / 1e6, "Ts": DF.t_supply_k - 273.15, "Tr": DF.t_return_k - 273.15}); p = tmp_path / "t.xlsx"; raw.to_excel(p, index=False)
    monkeypatch.setattr(sys, "argv", ["x", str(p), "--columns", "it_power=P", "t_supply=Ts", "t_return=Tr", "--step-hours", "0.1667", "--bins", "4",
                                      "--spec-json", json.dumps(SPEC), "--scan-units", "3"])
    trace_eval.main(); lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert [l["n_units"] for l in lines] == [1, 2, 3] and lines[0]["modularity_penalty"] > lines[2]["modularity_penalty"] >= 0
