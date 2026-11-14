import json, random
import pytest
from thermo_agent.eval import compare as CMP, model_eval as ME

def _ex(vals, key="success"): return [{"id": f"e{i}", key: v} for i, v in enumerate(vals)]

def test_mcnemar_detects_a_consistent_difference_and_ignores_none():
    a = _ex([1] * 20 + [0] * 4); b = _ex([0] * 12 + [1] * 8 + [0] * 4)
    r = CMP.paired(a, b, "success", True); assert r["only_a"] == 12 and r["only_b"] == 0 and r["p"] < 0.01 and r["diff"] > 0.4 and r["ci95"][0] > 0
    same = CMP.paired(a, a, "success", True); assert same["p"] == 1.0 and same["diff"] == 0
    c = _ex([1, None, 1]); assert CMP.paired(c, c, "success", True)["n"] == 2 and CMP.paired(c, _ex([1]), "success", True)["n"] == 1

def test_permutation_test_on_continuous_differences():
    rng = random.Random(1); a = _ex([0.5 + 0.1 * rng.random() for _ in range(40)], "rank"); b = _ex([0.3 + 0.1 * rng.random() for _ in range(40)], "rank")
    r = CMP.paired(a, b, "rank", False); assert r["p"] < 0.01 and 0.15 < r["diff"] < 0.25 and r["ci95"][0] > 0
    assert CMP.paired(a, a, "rank", False)["p"] == 1.0

def test_holm_is_monotone_capped_and_never_smaller_than_raw():
    ps = [0.01, 0.04, 0.03, 0.5]; adj = CMP.holm(ps)
    assert all(a >= p for a, p in zip(adj, ps)) and max(adj) <= 1.0 and adj[0] == pytest.approx(0.04) and adj[3] == pytest.approx(0.5)

def test_compare_reports_end_to_end_on_a_real_evaluation(tmp_path):
    from thermo_agent.synth.build_repair_data import build
    build(tmp_path, 45, seed=12)
    f = tmp_path / "repair_train.jsonl"; import shutil; shutil.copy(f, tmp_path / "repair_test_id.jsonl")
    table = {}
    for l in open(tmp_path / "repair_test_id.jsonl"):
        r = json.loads(l); table[r["prompt"][1]["content"]] = r["completion"][0]["content"]
    expert = lambda m: table.get(m[1]["content"], "x")
    ra, _ = ME.run(expert, tmp_path, ["repair"], splits=["test_id"], limit=12); rb, _ = ME.run(lambda m: "nothing", tmp_path, ["repair"], splits=["test_id"], limit=12, max_turns=1)
    ex = ra["repair"]["test_id"]["examples"]; assert ex and {"id", "success", "first_turn_success", "solves"} <= set(ex[0])
    rows = CMP.compare(ra, rb, ["repair"]); s = next(r for r in rows if r["metric"] == "success")
    assert s["diff"] == 1.0 and s["only_a"] == s["n"] and s["p_holm"] >= s["p"]
