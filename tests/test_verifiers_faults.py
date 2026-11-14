import random, pytest
from thermo_agent.core.solve import solve
from thermo_agent.core.schema import Plant
from thermo_agent.synth.topologies import cgam_orc
from thermo_agent.synth.faults import INJECTORS, apply_patch, _dump
from thermo_agent.synth.repair_dataset import sample_plant, make_records
from thermo_agent.verify.verifiers import run_all
from thermo_agent.verify.grounding import check_numbers
from thermo_agent.eval.repair import score_patch

def test_feasible_plant_passes_all():
    p = cgam_orc(); r = solve(p); rep = run_all(p, r)
    assert rep.valid and rep.score == 1.0 and rep.reward == 1.0, rep.failed()

def test_temperature_cross_flagged():
    p = cgam_orc(T_cond=298.0074, T_cw_in=298.0074, dT_cw=10.0); rep = run_all(p, solve(p))
    assert not rep.valid and {"V3a_nonneg_exergy_destruction", "V4a_no_temperature_cross"} <= set(rep.failed())

def _with(comp_id, key, val):
    d = _dump(cgam_orc())
    for c in d["components"]:
        if c["id"] == comp_id: c["params"][key] = val
    return Plant.model_validate(d)

def test_pump_efficiency_above_one_flagged_by_second_law_and_range():
    p = _with("PMP", "eta_s", 1.2); rep = run_all(p, solve(p))
    assert rep.solved and not rep.valid and {"V6_unit_and_range", "V3a_nonneg_exergy_destruction"} <= set(rep.failed())

def test_compressor_efficiency_above_one_rejected():
    p = _with("AC", "eta_s", 1.2); res = solve(p); rep = run_all(p, res)
    assert (not res.ok) or not rep.valid

def test_closure_flags_missing_energy_path():
    p = cgam_orc(); r = solve(p); r.kpis["closure_residual_w"] = 5e5
    assert "V3b_exergy_closure" in run_all(p, r).failed()

@pytest.mark.parametrize("kind", list(INJECTORS))
def test_inverse_patch_restores_original(kind):
    rng = random.Random(3)
    for _ in range(5):
        orig, _, _ = sample_plant(rng, topology="dc_electric" if kind == "swapped_ports" else None); broken, f = INJECTORS[kind](orig, rng)
        restored = Plant.model_validate(apply_patch(_dump(broken), f.inverse_patch))
        assert _dump(restored) == _dump(orig)
        assert score_patch(broken, orig, f.inverse_patch)["exact"]

def test_noop_patch_does_not_fix_detected_fault():
    rng = random.Random(5); n = 0
    for r in make_records(9, seed=5):
        if r.verifier_report_broken["detected"]:
            assert not score_patch(r.plant_broken, r.plant_original, [])["success"]; n += 1
    assert n >= 5

def test_grounding():
    p = cgam_orc(); r = solve(p); k = r.kpis
    good = f"Net power is {k['W_net_w']/1e6:.2f} MW with eta_I of {k['eta_I']*100:.1f} % and turbine inlet {p.components[2].params['T_out']:.0f} K."
    bad = good + " The HRSG delivers 55.0 MW."
    assert check_numbers(good, p, r)["fraction"] == 1.0
    g = check_numbers(bad, p, r); assert g["ungrounded"] == ["55.0 MW"]
