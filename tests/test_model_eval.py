import json, pathlib, pytest
from thermo_agent.eval import model_eval as ME, summarize
from thermo_agent.synth.build_repair_data import build as build_repair
from thermo_agent.design.data import build as build_design
from thermo_agent.audit.data import build as build_audit
from thermo_agent.ledger.data import build as build_ledger

@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    build_repair(d, 40, seed=4); build_design(d, 8, seed=4, max_evals=8); build_audit(d, 10, seed=4); build_ledger(d, 30, seed=4)
    # make sure each split used below has rows: reuse train rows as test_id when the hash left it empty
    for task in ("repair", "design", "audit", "ledger"):
        t = d / f"{task}_test_id.jsonl"
        if not t.exists() or t.stat().st_size == 0:
            rows = [l for l in open(d / f"{task}_train.jsonl")][:3] if (d / f"{task}_train.jsonl").exists() else []
            t.write_text("".join(rows))
    return d

def _expert(data):
    table = {}
    for f in data.glob("*_test_id.jsonl"):
        for l in open(f):
            r = json.loads(l); table[r["prompt"][1]["content"]] = r["completion"][0]["content"]
    return lambda messages: table.get(messages[1]["content"], "garbage")

def test_expert_generator_scores_perfectly_on_every_task(data):
    rep, log = ME.run(_expert(data), data, ["repair", "design", "audit", "ledger"], splits=["test_id"], limit=3)
    l = rep["ledger"]["test_id"]; assert l["all_correct"] == 1.0 and l["exact"] == 1.0 and l["closure"] == 1.0 and l["all_correct_ci95"][1] == 1.0
    r = rep["repair"]["test_id"]; d = rep["design"]["test_id"]; a = rep["audit"]["test_id"]
    assert r["success"] == 1.0 and r["first_turn_success"] == 1.0 and r["mean_solves"] == 1.0
    assert d["valid"] == 1.0 and abs(d["mean_gap_pct"]) < 1e-6 and d["mean_evals"] == 1.0
    assert a["format"] == 1.0 and a["rank"] >= 0.999 and a["gain"] >= 0.999 and a["pred"] >= 0.999
    assert len(log) >= 3

def test_refine_protocol_records_anytime_curves_per_example(data):
    rep, _ = ME.run(_expert(data), data, ["design"], splits=["test_id"], limit=2, refine_turns=3)
    d = rep["design"]["test_id"]; assert d["refine_turns"] == 3 and d["valid"] == 1.0
    for ex in d["examples"]:
        assert ex["anytime"][0][0] == 1 and abs(ex["anytime"][0][1]) < 1e-6 and len(ex["anytime"]) == 3        # best design first, two stale confirmations


def test_garbage_generator_scores_zero_without_crashing(data):
    rep, _ = ME.run(lambda m: "no idea", data, ["repair", "design", "audit", "ledger"], splits=["test_id"], limit=3, max_turns=2)
    assert rep["ledger"]["test_id"]["all_correct"] == 0.0 and rep["ledger"]["test_id"]["format"] == 0.0
    assert rep["repair"]["test_id"]["success"] == 0.0 and rep["design"]["test_id"]["valid"] == 0.0
    assert rep["audit"]["test_id"]["format"] == 0.0

def test_missing_split_is_reported_as_none(data):
    rep, _ = ME.run(lambda m: "x", data, ["audit"], splits=["test_ood_nothing"], limit=2); assert rep["audit"]["test_ood_nothing"] is None

def test_harness_text_reaches_the_generator(data):
    seen = []
    def gen(messages): seen.append(messages[0]["content"]); return "x"
    ME.run(gen, data, ["audit"], splits=["test_id"], limit=1, mode="agents+skills")
    assert seen and "AGENTS.md" in seen[0] and "SKILL exergy-audit" in seen[0]

def test_think_blocks_are_stripped_and_summary_table_renders(data, tmp_path):
    assert ME._THINK.sub("", "<think>secret</think>answer").strip() == "answer"
    rep, _ = ME.run(_expert(data), data, ["repair", "audit"], splits=["test_id"], limit=2)
    p = tmp_path / "r.json"; p.write_text(json.dumps({"meta": {"adapter": None, "harness": "none"}, **rep}))
    t = summarize.table([p]); assert "### repair" in t and "test_id" in t


def test_wilson_interval_known_values():
    lo, hi = ME.wilson(8, 10); assert abs(lo - 0.49) < 0.005 and abs(hi - 0.943) < 0.005
    assert ME.wilson(0, 10)[0] == 0.0 and ME.wilson(10, 10)[1] == 1.0 and ME.wilson(0, 0) is None

def test_summary_shows_intervals(data, tmp_path):
    rep, _ = ME.run(_expert(data), data, ["repair", "design"], splits=["test_id"], limit=3)
    p = tmp_path / "r.json"; p.write_text(json.dumps({"meta": {"adapter": None, "harness": "none"}, **rep}))
    t = summarize.table([p]); assert "[" in t.split("### repair")[1].split("### design")[0].splitlines()[3]

def test_load_rows_merges_directories_and_drops_duplicate_ids(tmp_path):
    """Evaluation reads the ground-truth dirs, so the architecture corpus has to be merged in; ids never collide."""
    a, b = tmp_path / "processed", tmp_path / "arch"
    a.mkdir(); b.mkdir()
    (a / "repair_test_id.jsonl").write_text(json.dumps({"id": "rep-1"}) + "\n" + json.dumps({"id": "rep-2"}) + "\n", encoding="utf-8")
    (b / "repair_test_id.jsonl").write_text(json.dumps({"id": "arch-1"}) + "\n" + json.dumps({"id": "rep-1"}) + "\n", encoding="utf-8")
    assert [r["id"] for r in ME.load_rows([a, b], "repair", "test_id", None)] == ["rep-1", "rep-2", "arch-1"]
    assert ME.load_rows(a, "repair", "test_id", 1) == [{"id": "rep-1"}], "one directory still works"
    assert ME.load_rows(tmp_path / "nope", "repair", "test_id", None) is None