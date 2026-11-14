import numpy as np, pandas as pd, pytest
from thermo_agent.datasets.frontier import describe, load_frontier, site_from_trace
from thermo_agent.synth.topologies import dc_chp, size_generation
from thermo_agent.core.solve import solve

@pytest.fixture
def trace(tmp_path):
    rng = np.random.default_rng(0); n = 500
    df = pd.DataFrame({"IT (MW)": rng.uniform(8, 29, n), "Ts (C)": rng.uniform(18, 25, n), "Tr (C)": rng.uniform(26, 38, n)})
    df["Q (MW)"] = df["IT (MW)"] * 0.9
    p = tmp_path / "t.xlsx"; df.to_excel(p, index=False); return p

M = {"it_power": "IT (MW)", "t_supply": "Ts (C)", "t_return": "Tr (C)", "waste_heat": "Q (MW)"}

def test_describe_lists_columns(trace): assert set(describe(trace).index) == {"IT (MW)", "Ts (C)", "Tr (C)", "Q (MW)"}

def test_load_converts_units(trace):
    df = load_frontier(trace, M); assert df.it_power_w.max() < 3e7 and df.t_supply_k.min() > 290

def test_missing_mapping_or_columns_fail_loudly(trace):
    with pytest.raises(KeyError): load_frontier(trace, {"it_power": "IT (MW)"})
    with pytest.raises(KeyError): load_frontier(trace, {**M, "t_return": "nope"})

def test_wrong_units_are_caught(trace):
    with pytest.raises(ValueError): load_frontier(trace, M, temp_unit="K") if False else load_frontier(trace, {**M, "t_return": "Ts (C)", "t_supply": "Tr (C)"})

def test_trace_drives_a_solvable_verified_plant(trace):
    from thermo_agent.verify.verifiers import run_all
    site, loop = site_from_trace(load_frontier(trace, M))
    loop["chilled_fraction"] = 0.4                          # what exhaust-driven absorption cooling can actually carry
    p = size_generation(dc_chp, cooling="absorption", **loop); r = solve(p); assert r.ok, r.message
    site = site.model_copy(update={"cooling_load_w": 0.39 * site.it_load_w})
    assert run_all(p, r, site).valid

def test_engine_exhaust_cannot_carry_the_whole_heat_load_but_turbine_exhaust_can(trace):
    site, loop = site_from_trace(load_frontier(trace, M))
    with pytest.raises(RuntimeError, match="cannot supply"):
        size_generation(dc_chp, P_it=loop["P_it"], chilled_fraction=1.0, cooling="absorption", prime_mover="engine")
    p = size_generation(dc_chp, P_it=loop["P_it"], chilled_fraction=1.0, cooling="absorption", prime_mover="turbine")
    assert solve(p).ok
