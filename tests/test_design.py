import json, random, pytest
from thermo_agent.core.schema import Site, Ambient
from thermo_agent.design import spec as S, rewards as R
from thermo_agent.design.build import evaluate_design
from thermo_agent.design.search import enumerate_designs, polish, reference_design
from thermo_agent.design.data import sample_site, make_row, assign_split, build
from thermo_agent.design.loop import design_loop

SITE = Site(it_load_w=20e6, cooling_load_w=10e6, grid_import_allowed=False, ambient=Ambient(T0=298.15, RH=0.6), t_supply_k=291.15, t_return_k=301.15)

def test_spec_roundtrip_and_validation():
    for pm in S.DISCRETE["prime_mover"]:
        for co in S.DISCRETE["cooling"]:
            sp = S.default_spec(pm, co, "wet"); assert S.parse_design("a " + S.format_design(sp)) == sp
    for bad in ["no block", "<design>\nprime_mover = turbine\n</design>",
                "<design>\n" + S.format_design(S.default_spec("engine", "electric", "dry")).replace("engine", "rocket").split("\n", 1)[1],
                S.format_design({**S.default_spec("engine", "electric", "dry"), "PR": 18.0}).replace("</design>", "PR = 18.0\n</design>")]:
        with pytest.raises(S.DesignError): S.parse_design(bad)
    with pytest.raises(S.DesignError): S.parse_design(S.format_design({**S.default_spec("turbine", "electric", "dry"), "T4": 2500.0}))

def test_all_default_designs_for_a_normal_site_are_valid_and_cost_fuel():
    tab = enumerate_designs(SITE); assert len(tab) == 32
    assert all(e.valid and e.fuel_w > 0 for s, e in tab if s["bottoming"] == "none")
    assert any(e.valid for s, e in tab if s["bottoming"] != "none")            # some ORC options are feasible, not all
    assert min(e.fuel_w for _, e in tab if e.fuel_w) < max(e.fuel_w for _, e in tab if e.fuel_w)

def test_engine_cannot_carry_full_chilled_load():
    site = SITE.model_copy(update={"cooling_load_w": 20e6}); e = evaluate_design(site, S.default_spec("engine", "absorption", "wet"))
    assert not e.valid and e.fuel_w is None

def test_water_limit_removes_wet_towers():
    site = SITE.model_copy(update={"max_water_kg_s": 0.0}); tab = enumerate_designs(site)
    assert all(e.fuel_w is None for s, e in tab if s["rejection"] == "wet")
    assert all(e.valid for s, e in tab if s["rejection"] == "dry" and s["bottoming"] == "none")

def test_polish_never_worse_than_start_and_reference_is_best_known():
    sp = S.default_spec("turbine", "absorption", "wet"); f0 = evaluate_design(SITE, sp).fuel_w
    sp2, f1, n = polish(SITE, sp, max_evals=25); assert f1 <= f0 and n <= 40
    ref = reference_design(SITE, max_evals=25); assert ref["fuel_w"] <= min(f for _, f, _ in ref["table"] if f) + 1e-6

def test_expert_completion_gets_full_reward_and_bad_designs_do_not():
    row = make_row("t-1", SITE, max_evals=25); kw = {"site": [row["site"]], "ref_fuel_w": [row["ref_fuel_w"]]}
    good = row["completion"][0]["content"]
    for f in R.REWARD_FUNCS: assert f(None, [good], **kw)[0] >= 0.999, f.__name__
    worst = S.format_design(S.default_spec("turbine", "electric", "dry"))
    assert R.d_quality(None, ["<rationale>x</rationale>\n" + worst], **kw)[0] < 0.9
    assert R.d_valid(None, ["nonsense"], **kw)[0] == 0.0 and R.d_format(None, ["nonsense"], **kw)[0] == 0.0

def test_fabricated_prediction_is_penalised():
    row = make_row("t-2", SITE, max_evals=15); kw = {"site": [row["site"]], "ref_fuel_w": [row["ref_fuel_w"]]}
    good = row["completion"][0]["content"]; bad = good.replace("</rationale>", " Fuel input will be 12.0 MW.</rationale>")
    assert R.d_grounded(None, [good], **kw)[0] > R.d_grounded(None, [bad], **kw)[0]

def test_conversational_completion_accepted():
    row = make_row("t-3", SITE, max_evals=15); kw = {"site": [row["site"]], "ref_fuel_w": [row["ref_fuel_w"]]}
    assert R.d_valid(None, [[{"role": "assistant", "content": row["completion"][0]["content"]}]], **kw)[0] == 1.0

def test_splits_are_structural():
    mk = lambda **k: Site(it_load_w=k.get("P", 20e6), cooling_load_w=k.get("P", 20e6) * k.get("cf", 0.5), ambient=Ambient(T0=k.get("T", 298.0)))
    assert assign_split(mk(P=50e6), "a") == "test_ood_scale" and assign_split(mk(T=307.0), "a") == "test_ood_climate"
    assert assign_split(mk(cf=0.9), "a") == "test_ood_load" and assign_split(mk(), "a") in ("train", "val", "test_id")

def test_loop_recovers_after_an_invalid_design_and_hybrid_polish_counts_evals():
    row = make_row("t-4", SITE, max_evals=15); calls = []
    bad = "<rationale>x</rationale>\n<design>\nprime_mover = rocket\n</design>"
    def model(m): calls.append(len(m)); return bad if len(calls) == 1 else row["completion"][0]["content"]
    out = design_loop(model, row, max_turns=3); assert out["valid"] and out["turns"] == 2 and out["evals"] == 1 and abs(out["gap"]) < 1e-6
    out2 = design_loop(lambda m: row["completion"][0]["content"], row, polish_evals=20); assert out2["evals"] > 1 and out2["gap"] <= 1e-9

def test_dataset_builder_small(tmp_path):
    counts = build(tmp_path, 8, seed=3, max_evals=12); assert sum(counts.values()) >= 5
    seen = {}
    for f in tmp_path.glob("design_*.jsonl"):
        rows = [json.loads(l) for l in open(f)]; seen[f.stem.replace("design_", "")] = len(rows)
        for r in rows: assert r["ref_fuel_w"] > 0 and "<design>" in r["completion"][0]["content"]
    assert seen == counts                                   # every written row survives in its file


def test_orc_bottoming_on_engine_exhaust_saves_fuel_but_turbine_exhaust_is_too_hot():
    base = evaluate_design(SITE, S.default_spec("engine", "absorption", "wet")); orc = evaluate_design(SITE, S.default_spec("engine", "absorption", "wet", "orc_cyclopentane"))
    assert base.valid and orc.valid and orc.fuel_w < base.fuel_w and orc.result.kpis["W_orc_net_w"] > 0
    hot = evaluate_design(SITE, S.default_spec("turbine", "absorption", "wet", "orc_r245fa"))
    assert not hot.valid and "too hot" in hot.reason

def test_heat_demand_adds_a_required_choice_and_both_sources_deliver():
    site = SITE.model_copy(update={"heat_demand_w": 4e6})
    with pytest.raises(S.DesignError): S.parse_design(S.format_design(S.default_spec("engine", "absorption", "wet")), needs_heat=True)
    ex = evaluate_design(site, S.default_spec("engine", "absorption", "wet", heat_source="exhaust"))
    hp = evaluate_design(site, S.default_spec("engine", "absorption", "wet", heat_source="heat_pump"))
    assert ex.valid and hp.valid and abs(ex.result.kpis["Q_heat_w"] - 4e6) < 1 and abs(hp.result.kpis["Q_heat_w"] - 4e6) < 1
    assert hp.fuel_w > ex.fuel_w                           # the heat pump spends electricity; exhaust heat is free here
    assert len(enumerate_designs(site)) == 64

def test_heat_pump_requires_a_rejection_loop_and_serves_demand_under_water_limit():
    site = SITE.model_copy(update={"heat_demand_w": 3e6, "max_water_kg_s": 0.0})
    e = evaluate_design(site, S.default_spec("turbine", "electric", "dry", heat_source="heat_pump"))
    assert e.valid and e.result.kpis["COP_HP"] > 2.5
