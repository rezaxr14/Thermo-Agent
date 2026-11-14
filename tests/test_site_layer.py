import pytest
from thermo_agent.core.schema import Plant, Site
from thermo_agent.core.solve import solve
from thermo_agent.synth.topologies import dc_chp, size_generation
from thermo_agent.synth.faults import _dump
from thermo_agent.verify.verifiers import run_all

def _sited(**kw):
    p = size_generation(dc_chp, **kw); return p, solve(p)

@pytest.mark.parametrize("cooling", ["absorption", "electric"])
def test_dc_plant_solves_and_verifies(cooling):
    p, r = _sited(cooling=cooling); assert r.ok, r.message
    site = Site(it_load_w=20e6, cooling_load_w=10e6, grid_import_allowed=False)
    rep = run_all(p, r, site); assert rep.valid and rep.score == 1.0, (rep.failed(), [c.detail for c in rep.checks if not c.ok])
    assert abs(r.kpis["closure_residual_w"]) < 1.0
    assert abs(r.kpis["W_net_w"] - 1.03 * 20e6) < 1e3 * 20          # sized to 103 % of IT load
    assert abs(r.kpis["Q_cooling_w"] - 10e6) < 1e3                   # half of the IT heat goes to the chilled loop

def test_absorption_chiller_cop_in_measured_range_and_energy_closes():
    p, r = _sited(cooling="absorption"); k = r.kpis
    assert 0.55 <= k["COP_CH"] <= 0.8 and k["COP_CH"] <= k["COPrev_CH"]
    assert all(abs(b.energy_residual) < 1e-3 and abs(b.mass_residual) < 1e-9 for b in r.balances)

def test_electric_chiller_uses_parasitic_power():
    p, r = _sited(cooling="electric"); w = {b.component: b.W_w for b in r.balances}
    assert w["CH"] < 0 and abs(w["CH"] - (-r.kpis["Q_cooling_w"] / r.kpis["COP_CH"])) < 1.0
    assert 3 < r.kpis["COP_CH"] < 12

def test_absorption_beats_electric_on_electricity_but_not_necessarily_on_fuel():
    pa, ra = _sited(cooling="absorption"); pe, re_ = _sited(cooling="electric")
    assert ra.kpis["Q_fuel_w"] < re_.kpis["Q_fuel_w"]                # no parasitic electricity -> smaller gas turbine

def test_machine_beyond_reversible_limit_is_flagged_by_exergy():
    p = size_generation(dc_chp, cooling="absorption", eps_abs=5.0, COP_max=1.2)
    r = solve(p); rep = run_all(p, r)
    assert not r.ok or "V3a_nonneg_exergy_destruction" in rep.failed() or "V6_unit_and_range" in rep.failed()

def test_driving_temperature_below_rejection_is_rejected():
    p = dc_chp(T_hw=300.0, T_gen_out=299.0); r = solve(p); assert r.error.value == "E_TEMP_CROSS"

def test_underserved_site_flagged():
    p, r = _sited(cooling="absorption")
    rep = run_all(p, r, Site(it_load_w=20e6, cooling_load_w=15e6, grid_import_allowed=False))
    assert "V8c_cooling_delivered" in rep.failed()
    rep2 = run_all(p, r, Site(it_load_w=30e6, grid_import_allowed=False)); assert "V8a_it_load_served" in rep2.failed()

def test_heat_pump_second_law_and_energy():
    from thermo_agent.core.schema import Component, Stream, FixedState
    S = lambda i, a, b, **kw: Stream(id=i, **{"from": a, "to": b}, fluid="Water", **kw)
    p = Plant(name="hp", components=[Component(id="HP", type="heat_pump", params={"dT_src": 8.0, "T_sink_out": 343.15, "eps": 0.5})],
              streams=[S("a", "ext:a", "HP.src_in", m_dot=100.0, fixed=FixedState(T=318.15, P=3e5)), S("b", "HP.src_out", "ext:b"),
                       S("c", "ext:c", "HP.sink_in", fixed=FixedState(T=323.15, P=5e5)), S("d", "HP.sink_out", "ext:d")])
    r = solve(p); assert r.ok, r.message
    k = r.kpis; cop = k["COP_HP"]; assert 2 < cop <= k["COPrev_HP"]
    assert abs(k["Q_heat_w"] - (-r.balances[0].W_w) * cop) < 1.0 and abs(k["closure_residual_w"]) < 1.0
    assert r.balances[0].exergy_destruction_w > 0

def test_plant_destruction_excludes_it_load():
    p, r = _sited(cooling="absorption"); k = r.kpis; b = {x.component: x.exergy_destruction_w for x in r.balances}
    assert abs(k["ED_plant_w"] - (k["ED_total_w"] - b["DC"])) < 1.0

def test_repair_generator_covers_both_topologies():
    import collections
    from thermo_agent.synth.repair_dataset import make_records
    c = collections.Counter(r.plant_original.name for r in make_records(40, seed=11))
    assert any(n.startswith("dc_") for n in c) and "cgam_orc_sampled" in c

def test_site_spec_exposes_plausible_unit_slip():
    from thermo_agent.synth.faults import apply_patch, INJECTORS
    p, r = _sited(cooling="absorption"); site = Site(it_load_w=20e6, cooling_load_w=9.9e6, grid_import_allowed=False)
    d = _dump(p)
    for i, c in enumerate(d["components"]):
        if c["id"] == "CC": c["params"]["T_out"] -= 273.15        # degC entered where K expected: still a valid plant
    q = Plant.model_validate(d); res = solve(q)
    assert res.ok and run_all(q, res).valid                         # physics alone cannot tell
    assert "V8b_electricity_balance" in run_all(q, res, site).failed()   # the site requirement can


def test_cop_above_the_reversible_limit_is_named_by_v5d():
    p = size_generation(dc_chp, cooling="electric", eps_ec=3.0); r = solve(p); assert r.ok
    rep = run_all(p, r, Site(it_load_w=20e6, cooling_load_w=9.9e6, grid_import_allowed=False))
    names = rep.failed(); assert "V5d_cop_below_reversible" in names and "V6_unit_and_range" in names
    assert "CH" in next(c.detail for c in rep.checks if c.name == "V5d_cop_below_reversible")

def test_normal_machines_pass_v5d():
    for cooling in ("absorption", "electric"):
        p = size_generation(dc_chp, cooling=cooling); rep = run_all(p, solve(p))
        assert "V5d_cop_below_reversible" in [c.name for c in rep.checks] and rep.valid
