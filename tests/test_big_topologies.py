"""The expert-scale plant is verifier-clean and additive to the small catalog."""
from thermo_agent.core.solve import solve
from thermo_agent.verify.verifiers import run_all


def test_big_steam_reheat_plant_is_verifier_clean():
    from thermo_agent.synth.big_topologies import big_cchp_steam2p_reheat_hp

    plant = big_cchp_steam2p_reheat_hp()
    assert len(plant.components) == 19
    assert len(plant.streams) == 34
    result = solve(plant)
    assert result.ok, result.message
    report = run_all(plant, result)
    assert report.valid, report
    assert result.kpis["P_it_w"] >= 60e6
    assert result.kpis["W_net_w"] > 0
    assert result.kpis["Q_fuel_w"] > result.kpis["W_net_w"]


def test_big_catalog_is_additive():
    from thermo_agent.synth import architectures as a

    assert len(a.ARCHITECTURES) == 65          # old small diagrams unchanged
    assert len(a.all_architectures()) == 66     # one new expert template
    assert a.by_name("big_cchp_steam2p_reheat_hp")["kind"] == "big"
