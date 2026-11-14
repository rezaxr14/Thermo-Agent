import json, random, pytest
from thermo_agent.synth.repair_dataset import make_records
from thermo_agent.synth.sft_data import to_row, expert_completion, assign_split
from thermo_agent.train import rewards as R
from thermo_agent.train.prompts import SYSTEM

ROWS = [to_row(r) for r in make_records(50, seed=21)]
KW = lambda rows: {k: [r[k] for r in rows] for k in ("plant_broken", "plant_original", "site", "deducible", "min_edits")}

def test_prompt_is_compact():
    for r in ROWS: assert len(SYSTEM) + len(r["prompt"][1]["content"]) < 6500

def test_expert_completion_earns_full_reward_on_detected_faults():
    rows = [r for r in ROWS if r["detected"]]; assert len(rows) >= 30
    comps = [r["completion"][0]["content"] for r in rows]
    for f in R.REWARD_FUNCS:
        out = f(None, comps, **KW(rows))
        for r, v in zip(rows, out):
            if v is None: continue
            assert v >= 0.999, (f.__name__, r["fault_type"], r["topology"], v)

def test_exact_reward_only_for_deducible():
    rows = ROWS; comps = [r["completion"][0]["content"] for r in rows]
    out = R.r_exact(None, comps, **KW(rows))
    for r, v in zip(rows, out): assert (v is None) == (not r["deducible"])

def test_garbage_and_noop_score_low():
    rows = [r for r in ROWS if r["detected"]][:6]
    for bad in ["I think the plant is fine.", "<diagnosis>fine</diagnosis>\n<edits>\n</edits>",
                "<diagnosis>x</diagnosis>\n<edits>\nSET NOPE.eta_s = 0.9\n</edits>"]:
        comps = [bad] * len(rows)
        assert max(R.r_valid(None, comps, **KW(rows))) < 1.0
        assert all((v or 0.0) == 0.0 for v in R.r_exact(None, comps, **KW(rows)))

def test_fabricated_numbers_in_diagnosis_are_penalised():
    r = next(x for x in ROWS if x["detected"]); good = r["completion"][0]["content"]
    bad = good.replace("</diagnosis>", " The loop pressure should be 123456789 Pa.</diagnosis>")
    kw = KW([r]); assert R.r_grounded(None, [good], **kw)[0] > R.r_grounded(None, [bad], **kw)[0]

def test_extra_edits_reduce_minimality_reward():
    r = next(x for x in ROWS if x["detected"]); good = r["completion"][0]["content"]
    comp = json.loads(r["plant_original"])["components"][0]; key = next(iter(comp["params"]))
    padded = good.replace("</edits>", f"SET {comp['id']}.{key} = {comp['params'][key]!r}\n</edits>")
    kw = KW([r]); assert R.r_minimal(None, [padded], **kw)[0] < R.r_minimal(None, [good], **kw)[0]

def test_splits_are_structural():
    names = {r["split"] for r in ROWS}
    for r in ROWS:
        if r["topology"] == "cgam_orc" or r["topology"].startswith("dc_engine"): assert r["split"] == "test_ood_topology"
        elif r["fault_type"] in ("sign_flip", "swapped_ports"): assert r["split"] == "test_ood_fault"
        else: assert r["split"] in ("train", "val", "test_id")

def test_search_baseline_repairs_deducible_faults_and_counts_solves():
    from thermo_agent.eval.baselines import search_repair
    from thermo_agent.core.schema import Plant, Site
    from thermo_agent.eval.repair import score_patch
    rows = [r for r in ROWS if r["detected"] and r["deducible"]][:5]; assert rows
    ok = 0
    for r in rows:
        b = Plant.model_validate_json(r["plant_broken"]); o = Plant.model_validate_json(r["plant_original"])
        site = Site.model_validate_json(r["site"]) if r["site"] else None
        patch, n = search_repair(b, site, hint=r["prompt"][1]["content"].split("RESULT")[-1])
        if patch is not None and score_patch(b, o, patch, site)["success"]: ok += 1; assert n >= 1
    assert ok >= 3

def test_repair_loop_with_scripted_model():
    from thermo_agent.agent.loop import repair_loop
    r = next(x for x in ROWS if x["detected"]); calls = []
    def model(messages):
        calls.append(len(messages))
        return "garbage" if len(calls) == 1 else r["completion"][0]["content"]
    out = repair_loop(model, r, max_turns=3)
    assert out["success"] and out["turns"] == 2 and out["solves"] == 1 and calls[1] > calls[0]

def test_training_scripts_compile_and_reward_contract():
    import py_compile, pathlib, inspect
    for f in ("sft.py", "grpo.py", "config.py"):
        py_compile.compile(str(pathlib.Path(R.__file__).parent / f), doraise=True)
    assert len(R.REWARD_FUNCS) == len(R.REWARD_WEIGHTS)
    for f in R.REWARD_FUNCS:
        ps = inspect.signature(f).parameters
        assert list(ps)[:2] == ["prompts", "completions"] and any(p.kind is p.VAR_KEYWORD for p in ps.values())

def test_grpo_batch_is_divisible_by_group_size():
    from thermo_agent.train.config import GrpoSettings
    g = GrpoSettings(); assert (g.per_device_batch * g.grad_accum) % g.num_generations == 0

def test_conversational_completions_are_accepted():
    r = next(x for x in ROWS if x["detected"]); kw = KW([r])
    as_msgs = [[{"role": "assistant", "content": r["completion"][0]["content"]}]]
    assert R.r_valid(None, as_msgs, **kw)[0] >= 0.999
