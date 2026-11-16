import json, random, pytest
from thermo_agent.core.schema import Site, Plant
from thermo_agent.core.solve import solve
from thermo_agent.synth.topologies import dc_chp, size_generation
from thermo_agent.synth.repair_dataset import sample_plant
from thermo_agent.synth.splits import is_ood_topology
from thermo_agent.audit import changes as ch, rewards as R
from thermo_agent.audit.spec import parse_audit, AuditError
from thermo_agent.audit.truth import truth, ranking, first_law_ranking, summary_text
from thermo_agent.audit.data import make_row, build, expert_completion

P = 20e6
PLANT = size_generation(dc_chp, P_it=P, prime_mover="engine", cooling="absorption", rejection="wet")
SITE = Site(it_load_w=P, cooling_load_w=0.99 * 0.5 * P, grid_import_allowed=False)
ROW = make_row("t", PLANT)
KW = lambda rows: {k: [r[k] for r in rows] for k in ("plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w")}

def test_efficiencies_and_loop_temperatures_cannot_be_changed():
    assert not ch.check_change(PLANT, "ENG", "eta_el", 0.45) and not ch.check_change(PLANT, "CH", "eps", 0.3)
    assert not ch.check_change(PLANT, "CH", "T_gen_out", 350.0) and not ch.check_change(PLANT, "HX", "T_cold_out", 360.0)
    assert not ch.check_change(PLANT, "NOPE", "x", 1.0) and not ch.check_change(PLANT, "REJ", "fan_frac", 0.5)   # outside +-20 %
    assert ch.check_change(PLANT, "REJ", "fan_frac", 0.012)

def test_every_candidate_is_allowed_and_resizing_holds_net_electricity():
    T = truth(PLANT, SITE); assert T and len(T["table"]) >= 5
    for cid, p, v in ch.candidates(PLANT)[:6]:
        assert ch.check_change(PLANT, cid, p, v)
        out = ch.resize(ch.apply_change(PLANT, cid, p, v), T["net_w"]); assert out is not None
        assert abs(out[1].kpis["W_net_w"] - T["net_w"]) <= 2e-5 * T["net_w"] * 1.01

def test_exergy_ranking_excludes_the_it_load_and_differs_from_first_law_ranking():
    res = solve(PLANT); ex = ranking(PLANT, res); fl = first_law_ranking(PLANT, res)
    assert "DC" not in [c for c, _ in ex] and ex[0][0] == "ENG" and fl[0][0] == "REJ"
    assert abs(sum(v for _, v in ex) + [b.exergy_destruction_w for b in res.balances if b.component == "DC"][0] - res.kpis["ED_total_w"]) < 1.0

def test_summary_hides_exergy_destruction():
    s = summary_text(PLANT, solve(PLANT)); assert "exergy" not in s.lower() and "KPI fuel_MW=" in s

def test_parser():
    ids = {"ENG", "HX", "CH"}
    ok = "<ranking>ENG > HX</ranking><change>SET REJ.fan_frac = 0.01</change><prediction>fuel_change_mw = -0.4</prediction>"
    assert parse_audit(ok, ids)["ranking"] == ["ENG", "HX"]
    for bad in ["x", ok.replace("ENG > HX", "ENG > ZZZ"), ok.replace("ENG > HX", "ENG > ENG"), ok.replace("SET REJ.fan_frac = 0.01", "set it lower"), ok.replace("fuel_change_mw = -0.4", "-0.4")]:
        with pytest.raises(AuditError): parse_audit(bad, ids)

def test_expert_completion_earns_full_reward_on_every_channel():
    comp = [ROW["completion"][0]["content"]]
    for f in R.REWARD_FUNCS: assert f(None, comp, **KW([ROW]))[0] >= 0.999, f.__name__

def test_wrong_answers_are_scored_down():
    comp = ROW["completion"][0]["content"]; kw = KW([ROW])
    wrong_rank = comp.replace(f"<ranking>{comp.split('<ranking>')[1].split('</ranking>')[0]}</ranking>", "<ranking>REJ > CH</ranking>")
    assert R.a_rank(None, [wrong_rank], **kw)[0] < 0.5
    forbidden = comp.replace(comp.split("<change>")[1].split("</change>")[0], "SET ENG.eta_el = 0.49")
    assert R.a_valid(None, [forbidden], **kw)[0] <= 0.2 and R.a_gain(None, [forbidden], **kw)[0] == 0.0
    bad_pred = comp.replace("fuel_change_mw = ", "fuel_change_mw = 5").replace("= 5-", "= -5")
    assert R.a_pred(None, [bad_pred], **kw)[0] == 0.0
    fabricated = comp.replace("</analysis>", " The stack carries 77.0 MW.</analysis>")
    assert R.a_grounded(None, [fabricated], **kw)[0] < R.a_grounded(None, [comp], **kw)[0]

def test_a_change_that_costs_fuel_gets_no_gain_reward():
    T = truth(PLANT, SITE); worse = max((t for t in T["table"] if t[1] is not None), key=lambda t: t[1])
    if worse[1] <= 0: pytest.skip("every allowed change saves fuel for this plant")
    comp = ROW["completion"][0]["content"]; swapped = comp.replace(comp.split("<change>")[1].split("</change>")[0], worse[0])
    assert R.a_gain(None, [swapped], **KW([ROW]))[0] == 0.0

def test_ndcg_properties():
    ed = {"A": 10.0, "B": 5.0, "C": 1.0, "D": 0.0}
    assert R.ndcg(["A", "B", "C"], ed) == pytest.approx(1.0) and R.ndcg(["D", "C", "B"], ed) < R.ndcg(["B", "A", "C"], ed) < 1.0

def test_data_builder_rows_and_files_agree(tmp_path):
    counts = build(tmp_path, 8, seed=2); assert sum(counts.values()) >= 4
    seen = {f.stem.replace("audit_", ""): sum(1 for _ in open(f)) for f in tmp_path.glob("audit_*.jsonl")}
    assert seen == counts
    for f in tmp_path.glob("audit_*.jsonl"):
        for l in open(f):
            r = json.loads(l); assert r["best_gain_w"] > 0 and r["split"] in f.stem
            assert (r["split"] == "test_ood_topology") == is_ood_topology(r["topology"])
