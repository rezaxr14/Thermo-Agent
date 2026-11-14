import pytest
from thermo_agent.core.cgam_plant import solve as ref_solve
from thermo_agent.core.solve import solve
from thermo_agent.synth.topologies import cgam_orc
from thermo_agent.core.schema import ErrorCode

R = ref_solve(); T11 = R["T"][11]
P = cgam_orc(n_air=R["n_a"] * 1e3, T_cond=T11, T_cw_in=T11, dT_cw=10.0)
RES = solve(P)
rel = lambda a, b: abs(a - b) / abs(b)

def test_solves(): assert RES.ok, RES.message
def test_net_power(): assert rel(RES.kpis["W_net_w"] / 1e6, R["W_net"] + R["W_orc_net"]) < 1e-3
def test_energy_efficiency(): assert rel(RES.kpis["eta_I"], R["eta_I"]) < 5e-3
def test_exergy_efficiency(): assert rel(RES.kpis["eta_II"], R["eta_II"]) < 0.02
def test_hrsg_pinch():
    b = {x.component: x for x in RES.balances}; assert abs(b["HRSG"].min_approach_K - R["dT_pinch_hrsg"]) < 0.05
def test_component_destruction():
    b = {x.component: x.exergy_destruction_w / 1e6 for x in RES.balances}
    for c, r in {"AC": "AC", "AP": "AP", "GT": "GT", "HRSG": "HRSG", "EVA": "EVA"}.items(): assert rel(b[c], R["ED"][r]) < 0.01
    assert rel(b["CC"], R["ED"]["CC"]) < 0.02
def test_closure(): assert abs(RES.kpis["closure_residual_w"]) < 1.0
def test_balances_close():
    assert all(abs(b.mass_residual) < 1e-9 and abs(b.energy_residual) < 1e-3 for b in RES.balances)

def test_unknown_fluid():
    d = P.model_dump(by_alias=True); d["streams"][0]["fluid"] = "Unobtainium"
    from thermo_agent.core.schema import Plant
    r = solve(Plant.model_validate(d)); assert r.error == ErrorCode.UNKNOWN_FLUID

def test_disconnected():
    from thermo_agent.core.schema import Plant
    d = P.model_dump(by_alias=True); d["streams"] = [s for s in d["streams"] if s["id"] != "s2"]
    r = solve(Plant.model_validate(d)); assert r.error == ErrorCode.DISCONNECTED
