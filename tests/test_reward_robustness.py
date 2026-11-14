"""Reward functions must never raise and must stay bounded, whatever the model emits: an exception would end a GRPO run."""
import json, math, random
import pytest
from thermo_agent.synth.repair_dataset import make_records
from thermo_agent.synth.sft_data import to_row
from thermo_agent.design.data import make_row as design_row, sample_site
from thermo_agent.audit.data import make_row as audit_row
from thermo_agent.synth.repair_dataset import sample_plant
from thermo_agent.train import rewards as RR
from thermo_agent.design import rewards as DR
from thermo_agent.audit import rewards as AR

REP = [r for r in (to_row(x) for x in make_records(10, seed=31)) if r["detected"]][:3]
DES = [r for r in (design_row(f"d{i}", sample_site(random.Random(i)), max_evals=6) for i in range(3)) if r][:2]
AUD = [r for r in (audit_row(f"a{i}", sample_plant(random.Random(40 + i), topology="dc_absorption", rich=True)[0]) for i in range(3)) if r][:2]

TOKENS = ["<edits>", "</edits>", "<diagnosis>", "</diagnosis>", "<design>", "</design>", "<rationale>", "</rationale>", "<analysis>", "</analysis>",
          "<ranking>", "</ranking>", "<change>", "</change>", "<prediction>", "</prediction>", "SET ", "REWIRE ", "FLUID ", "@", ".", " = ", " > ", "\n",
          "AC", "CC", "GT", "HX", "CH", "REJ", "ENG", "DC", "eta_s", "PR", "T_out", "eps", "P_it", "fixed.P", "to", "from", "Water", "FLUE", "nan", "inf",
          "-inf", "1e999", "-1e999", "0", "-0.0", "1e-320", "99999999999999999999", "0.5", "-3", "fuel_change_mw", "prime_mover", "turbine", "engine",
          "cooling", "absorption", "electric", "rejection", "dry", "wet", "bottoming", "orc_toluene", "none", "heat_source", "exhaust", "dT_rej", "T4", "ünï", "💥", "\x00"]

def junk(rng, n):
    return "".join(rng.choice(TOKENS) for _ in range(n))

def _ok(vals):
    for v in vals:
        assert v is None or (isinstance(v, (int, float)) and math.isfinite(v) and -0.01 <= v <= 1.06), v

def _cols(rows, cols): return {c: [r[c] for r in rows] for c in cols}

@pytest.mark.parametrize("seed", range(6))
def test_repair_rewards_survive_junk(seed):
    rng = random.Random(seed); comps = [junk(rng, rng.randint(0, 40)) for _ in range(12)]
    comps += ["<diagnosis>x</diagnosis><edits>\nSET AC.eta_s = 1e999\n</edits>", "<edits>" + "SET AC.PR = 1\n" * 500 + "</edits>", "x" * 200000]
    rows = (REP * 5)[:len(comps)]; kw = _cols(rows, ("plant_broken", "plant_original", "site", "deducible", "min_edits"))
    for f in RR.REWARD_FUNCS: _ok(f(None, comps, **kw))

@pytest.mark.parametrize("seed", range(6))
def test_design_rewards_survive_junk(seed):
    rng = random.Random(seed); comps = [junk(rng, rng.randint(0, 40)) for _ in range(12)]
    comps += ["<rationale>x</rationale><design>\nprime_mover = turbine\nPR = 1e999\n</design>", "<design>\n" + "dT_rej = 8\n" * 300 + "</design>", "y" * 200000]
    rows = (DES * 8)[:len(comps)]; kw = _cols(rows, ("site", "ref_fuel_w"))
    for f in DR.REWARD_FUNCS: _ok(f(None, comps, **kw))

@pytest.mark.parametrize("seed", range(6))
def test_audit_rewards_survive_junk(seed):
    rng = random.Random(seed); comps = [junk(rng, rng.randint(0, 40)) for _ in range(12)]
    comps += ["<ranking>CC</ranking><change>SET CC.T_out = 1e999</change><prediction>fuel_change_mw = 1e999</prediction>",
              "<ranking>CC</ranking><change>SET CC.T_out = -5</change><prediction>fuel_change_mw = 0</prediction>", "z" * 200000]
    rows = (AUD * 8)[:len(comps)]; kw = _cols(rows, ("plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w"))
    for f in AR.REWARD_FUNCS: _ok(f(None, comps, **kw))

def test_a_valid_answer_hidden_among_junk_is_still_read():
    r = REP[0]; good = r["completion"][0]["content"]
    kw = _cols([r], ("plant_broken", "plant_original", "site", "deducible", "min_edits"))
    assert RR.r_valid(None, ["noise " + good + " noise"], **kw)[0] >= 0.999


from thermo_agent.train.perturb import mutate, EXTREMES

def test_mutated_valid_answers_never_crash_any_reward():
    rng = random.Random(7)
    for task, rows, cols, mod in (("repair", REP, ("plant_broken", "plant_original", "site", "deducible", "min_edits"), RR),
                                  ("design", DES, ("site", "ref_fuel_w"), DR), ("audit", AUD, ("plant", "site", "ed_json", "fuel_w", "net_w", "best_gain_w"), AR)):
        for r in rows:
            comps = [mutate(r["completion"][0]["content"], rng) for _ in range(40)]
            kw = _cols([r] * len(comps), cols)
            for f in mod.REWARD_FUNCS: _ok(f(None, comps, **kw))


from thermo_agent.ledger.data import make_row as ledger_row
from thermo_agent.ledger import rewards as LR
_LP = sample_plant(random.Random(88), topology="dc_electric", rich=True)[0]
LED = [ledger_row(f"l{i}", _LP, random.Random(i)) for i in range(2)]
LED = [r for r in LED if r]

def test_ledger_rewards_survive_junk_and_mutations():
    rng = random.Random(9); r = LED[0]
    comps = [junk(rng, rng.randint(0, 40)) for _ in range(10)] + [mutate(r["completion"][0]["content"], rng) for _ in range(40)]
    comps += ["<fix>\nFIX T0 = 1e999\n</fix>", "<fix>\nFIX E[air] = " + "1+" * 100 + "1\n</fix>", "<fix>\nFIX T0 = " + "(" * 500 + "1" + ")" * 500 + "\n</fix>", "q" * 200000]
    kw = {"ledger": [r["ledger"]] * len(comps)}
    for f in LR.REWARD_FUNCS: _ok(f(None, comps, **kw))
