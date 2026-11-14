import pytest
from thermo_agent.core.schema import Site, Plant, ErrorCode
from thermo_agent.core.solve import solve
from thermo_agent.synth.topologies import dc_chp, size_generation
from thermo_agent.synth.faults import _dump
from thermo_agent.verify.verifiers import run_all

SITE = Site(it_load_w=20e6, cooling_load_w=9.9e6, grid_import_allowed=False)

def _ok(**kw):
    p = size_generation(dc_chp, **kw); r = solve(p); assert r.ok, r.message
    rep = run_all(p, r, SITE); assert rep.valid, (rep.failed(), [c.detail for c in rep.checks if not c.ok])
    assert abs(r.kpis["closure_residual_w"]) < 1.0
    assert all(abs(b.energy_residual) < 1e-2 and abs(b.mass_residual) < 1e-9 for b in r.balances)
    assert abs(r.kpis["W_net_w"] - 1.03 * 20e6) < 2e4
    return p, r

@pytest.mark.parametrize("prime", ["turbine", "engine"])
@pytest.mark.parametrize("cooling", ["absorption", "electric"])
@pytest.mark.parametrize("rejection", ["ext", "dry", "wet"])
def test_every_prime_mover_cooling_rejection_combination(prime, cooling, rejection):
    _ok(prime_mover=prime, cooling=cooling, rejection=rejection)

def test_wet_tower_beats_dry_cooler_on_electric_chiller_power():
    _, rd = _ok(cooling="electric", rejection="dry"); _, rw = _ok(cooling="electric", rejection="wet")
    assert rw.kpis["COP_CH"] > rd.kpis["COP_CH"]                 # colder rejection temperature

def test_rejector_fan_power_counts_as_parasitic_load():
    p, r = _ok(cooling="absorption", rejection="dry"); b = next(b for b in r.balances if b.component == "REJ")
    assert b.W_w < 0 and abs(abs(b.W_w) - 0.015 * b.Q_w) < 1e-6 * b.Q_w

def test_engine_energy_split_and_exhaust_temperature():
    p, r = _ok(prime_mover="engine", cooling="absorption")
    k = r.kpis; eng = p.components[0].params
    assert abs(k["eta_I_power"] - eng["eta_el"]) < 1e-9                              # no parasitics with absorption cooling
    assert 600 < max(s.T for s in r.states if s.stream == "s5") < 1000                # exhaust temperature plausible (K)

def test_hot_ambient_hurts_dry_cooling_more_than_wet():
    base = dict(cooling="electric")
    cd = size_generation(dc_chp, rejection="dry", T_amb=313.15, **base); cw = size_generation(dc_chp, rejection="wet", T_amb=313.15, **base)
    rd, rw = solve(cd), solve(cw); assert rd.ok and rw.ok and rd.kpis["COP_CH"] < rw.kpis["COP_CH"]

def test_fluid_class_mismatch_is_reported():
    d = _dump(dc_chp()); next(s for s in d["streams"] if s["id"] == "chw_ret")["fluid"] = "CGAM_AIR"
    r = solve(Plant.model_validate(d)); assert r.error == ErrorCode.FLUID_MISMATCH and "chw_ret" in r.message


def test_engine_heat_recovery_limit_is_caught_by_the_temperature_cross_check():
    p = size_generation(dc_chp, P_it=20e6, chilled_fraction=0.8, cooling="absorption", prime_mover="engine")
    r = solve(p); assert r.ok
    rep = run_all(p, r, Site(it_load_w=20e6, cooling_load_w=15.8e6, grid_import_allowed=False))
    assert "V4a_no_temperature_cross" in rep.failed()

def test_wet_tower_water_use_is_physical_and_dry_cooler_uses_none():
    pw, rw = _ok(cooling="electric", rejection="wet"); pd, rd = _ok(cooling="electric", rejection="dry")
    assert rd.kpis["water_use_kg_s"] == 0.0
    b = next(b for b in rw.balances if b.component == "REJ")
    assert 0.35 < rw.kpis["water_use_kg_s"] / (b.Q_w / 2.45e6) < 1.05          # latent heat of water near 2.4 MJ/kg

def test_site_water_limit_is_a_verified_requirement():
    p, r = _ok(cooling="electric", rejection="wet")
    tight = Site(it_load_w=20e6, cooling_load_w=9.9e6, grid_import_allowed=False, max_water_kg_s=0.5 * r.kpis["water_use_kg_s"])
    assert "V8e_water_limit" in run_all(p, r, tight).failed()
