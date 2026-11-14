import json, math, random, pytest
from thermo_agent.synth.repair_dataset import sample_plant, TOPOLOGIES
from thermo_agent.ledger import program as P, faults as F, rewards as R
from thermo_agent.ledger.expr import evaluate, parse, ExprError
from thermo_agent.ledger.data import make_row, build, split_of
from thermo_agent.ledger.run_eval import run as run_baselines

CTX = {"consts": {"T0": 300.0}, "tables": {"m": {"a": 2.0}, "E": {"a": 5.0}, "W": {}, "h": {}, "s": {}, "g0": {}, "ech": {}}}

@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "E[a].real", "E[a]**2", "(lambda: 1)()", "'text'", "True", "[1, 2]", "m[1]", "m[a][b]",
                                 "foo", "E['a']", "1 if 1 else 2", "x" * 400, "(" * 200 + "1" + ")" * 200, "E[a] +"])
def test_evaluator_refuses_anything_but_arithmetic(bad):
    with pytest.raises(ExprError): evaluate(bad, CTX)

def test_evaluator_arithmetic_and_errors():
    assert evaluate("m[a]*(E[a] - 1)/4 + T0*0 - -1", CTX) == pytest.approx(3.0)
    for e in ("E[zz]", "1/0", "1e308*1e308", "W[c]"):
        with pytest.raises(ExprError): evaluate(e, CTX)

PLANTS = [sample_plant(random.Random(10 + i), topology=t, rich=True) for i, t in enumerate(TOPOLOGIES)]

@pytest.mark.parametrize("i", range(len(PLANTS)))
def test_correct_ledger_closes_and_every_line_matches_truth(i):
    p, res, _ = PLANTS[i]; d = P.build(p, res); r = P.run(d["lines"], d)
    assert not r["errors"] and abs(r["closure"]) < 1e-9 * max(d["e_fuel_mw"], 1)
    assert all(P.matches_truth(r, d, k) for k in d["lines"])
    assert all(v > -1e-6 for v in r["ED"].values())

def test_equivalent_expressions_are_accepted_by_value():
    p, res, _ = PLANTS[1]; d = P.build(p, res); key = next(k for k in d["lines"] if k.startswith("ED[")); terms = d["lines"][key].split(" + ")
    lines = {**d["lines"], key: " + ".join(reversed(terms)) if len(terms) > 1 else d["lines"][key]}
    assert P.matches_truth(P.run(lines, d), d, key)

def test_every_injection_changes_a_value_and_still_evaluates():
    d = P.build(PLANTS[1][0], PLANTS[1][1]); rng = random.Random(0); seen = set()
    for i in range(120):
        out = F.inject(rng, d, 1 + (i % 4 == 0))
        assert out is not None
        lines, recs = out; r = P.run(lines, d); assert not r["errors"]
        for rec in recs:
            seen.add(rec["type"]); assert not P.matches_truth(r, d, rec["key"]) and rec["buggy"] != rec["correct"]
    assert seen >= set(F.STREAM_FAULTS) | set(F.CONST_FAULTS) | (set(F.COMPONENT_FAULTS) - {"dropped_work"})
    assert F.HELD_OUT & seen                                                       # held-out types are generated (dropped_work needs a plant with work terms)

def test_internal_stream_errors_cancel_in_the_closure_but_show_up_elsewhere():
    d = P.build(PLANTS[1][0], PLANTS[1][1]); internal = next(s for s in d["stream_order"] if s not in d["ext_in"] and s not in d["ext_out"])
    lines = {**d["lines"], f"E[{internal}]": "2*" + d["lines"][f"E[{internal}]"]}; r = P.run(lines, d)
    assert abs(r["closure"]) < 1e-9 * max(d["e_fuel_mw"], 1)                    # telescopes
    assert not P.matches_truth(r, d, f"E[{internal}]")

ROW = None
def _row():
    global ROW
    if ROW is None:
        rng = random.Random(77); p, _, _ = sample_plant(rng, topology="dc_absorption", rich=True); ROW = make_row("t", p, rng)
    return ROW

def _kw(row): return {"ledger": [row["ledger"]]}

def test_expert_earns_full_reward_on_every_channel():
    row = _row(); c = [row["completion"][0]["content"]]
    for f in R.REWARD_FUNCS: assert f(None, c, **_kw(row))[0] >= 0.999, f.__name__

def test_noop_and_wrong_fixes_are_scored_down():
    row = _row(); L = json.loads(row["ledger"]); key = L["faulty"][0]
    noop = "<diagnosis>all fine</diagnosis><fix>\nFIX T0 = " + L["lines"]["T0"] + "\n</fix>"
    assert R.l_exact(None, [noop], **_kw(row))[0] == 0.0 and R.l_closure(None, [noop], **_kw(row))[0] < 1.0 or key == "T0"
    victim = next(k for k in L["lines"] if k.startswith("ED[") and k not in L["faulty"])
    wrecked = f"<diagnosis>x</diagnosis><fix>\nFIX {victim} = 0\n</fix>"
    assert R.l_exact(None, [wrecked], **_kw(row))[0] == 0.0

def test_extra_edits_and_fabricated_numbers_are_penalised():
    row = _row(); good = row["completion"][0]["content"]; L = json.loads(row["ledger"])
    victim = next(k for k in L["lines"] if k.startswith("ED[") and k not in L["faulty"])
    padded = good.replace("</fix>", f"FIX {victim} = {L['data']['lines'][victim]}\n</fix>")
    assert R.l_minimal(None, [padded], **_kw(row))[0] < R.l_minimal(None, [good], **_kw(row))[0] == 1.0
    fab = good.replace("</diagnosis>", " The loss is 123456 MW.</diagnosis>")
    assert R.l_grounded(None, [fab], **_kw(row))[0] < R.l_grounded(None, [good], **_kw(row))[0]

def test_fix_syntax_is_enforced():
    row = _row(); L = json.loads(row["ledger"])
    for bad in ["<fix>\nFIX E[nope] = 1\n</fix>", "<fix>\nFIX T0 = __import__('os')\n</fix>", "<fix>\n" + "FIX T0 = 1\n" * 5 + "</fix>", "no block", "<fix>\n</fix>"]:
        with pytest.raises((ValueError, ExprError)): R.parse_fixes(bad, set(L["lines"]))

def test_splits_are_structural():
    assert split_of("cgam_orc", {"sign_flip"}, "a") == "test_ood_topology" and split_of("dc_engine_electric", set(), "a") == "test_ood_topology"
    assert split_of("dc_absorption", {"wrong_stream"}, "a") == "test_ood_fault" and split_of("dc_absorption", {"sign_flip"}, "a") in ("train", "val", "test_id")

def test_builder_files_meta_and_baselines(tmp_path):
    counts = build(tmp_path, 24, seed=5); assert sum(counts.values()) >= 18
    assert {f.stem.replace("ledger_", ""): sum(1 for _ in open(f)) for f in tmp_path.glob("ledger_*.jsonl")} == counts
    assert json.loads((tmp_path / "meta_ledger.json").read_text())["params"]["n"] == 24
    f = next(tmp_path.glob("ledger_train.jsonl")); res = run_baselines(f, 20)
    assert res["regenerate_solved"] == 1.0 and res["no_op_solved"] == 0.0 and res["symptom_rate_by_fault"]
