import numpy as np, pandas as pd, pytest
from thermo_agent.core.schema import Site, Ambient
from thermo_agent.design.spec import default_spec
from thermo_agent.design.trace_eval import bin_trace, trace_fuel, rank_catalog

rng = np.random.default_rng(0); N = 600
DF = pd.DataFrame({"it_power_w": rng.uniform(10e6, 30e6, N), "t_supply_k": 291.0 + rng.normal(0, 0.3, N), "t_return_k": 301.0 + rng.normal(0, 0.3, N)})
TMPL = Site(it_load_w=30e6, cooling_load_w=15e6, grid_import_allowed=False, t_supply_k=291.0, t_return_k=301.0)

def test_bins_cover_the_trace_and_are_ordered():
    b = bin_trace(DF, 6); assert sum(x["samples"] for x in b) == N and [x["it_load_w"] for x in b] == sorted(x["it_load_w"] for x in b)

def test_fuel_energy_is_weighted_by_time_and_intensity_is_nearly_load_independent():
    b = bin_trace(DF, 5); r = trace_fuel(TMPL, default_spec("engine", "absorption", "wet"), b, 0.5, 1 / 6)
    assert r["infeasible_time_share"] == 0 and r["fuel_mwh"] > 0 and "lower bound" in r["idealisation"]
    fuel_per_it = [x["fuel_mw"] / x["it_mw"] for x in r["per_bin"]]; assert max(fuel_per_it) / min(fuel_per_it) < 1.05      # ideal part-load
    assert r["fuel_mwh"] == pytest.approx(sum(x["fuel_mw"] * x["share"] for x in r["per_bin"]) * N / 6, rel=2e-3)

def test_a_more_efficient_design_uses_less_fuel_over_the_trace():
    b = bin_trace(DF, 4); a = trace_fuel(TMPL, default_spec("engine", "absorption", "wet"), b, 0.5, 1 / 6)
    e = trace_fuel(TMPL, default_spec("engine", "electric", "wet"), b, 0.5, 1 / 6); assert a["fuel_mwh"] < e["fuel_mwh"]

def test_hot_ambient_bins_hurt_dry_cooling_more_than_wet_and_infeasible_bins_are_counted():
    hot = DF.copy(); hot["ambient_k"] = 312.0; b = bin_trace(hot, 3, "ambient_k")
    dry = trace_fuel(TMPL, default_spec("engine", "absorption", "dry"), b, 0.5, 1 / 6); wet = trace_fuel(TMPL, default_spec("engine", "absorption", "wet"), b, 0.5, 1 / 6)
    assert wet["infeasible_time_share"] == 0 and dry["infeasible_time_share"] > 0                # dry cooler cannot drive the chiller at 39 C

def test_catalog_ranking_puts_fully_feasible_designs_first():
    b = bin_trace(DF, 3); ranked = rank_catalog(TMPL, b, 0.5, 1 / 6)
    assert len(ranked) == 32 and ranked[0][1]["infeasible_time_share"] == 0
    shares = [r["infeasible_time_share"] for _, r in ranked]; assert shares == sorted(shares)
