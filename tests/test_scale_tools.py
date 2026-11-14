import json, pytest
from thermo_agent.synth.build_repair_data import build as build_repair
from thermo_agent.design.data import build as build_design
from thermo_agent.audit.data import build as build_audit
from thermo_agent.ledger.data import build as build_ledger
from thermo_agent.eval import data_report, thermoqa

def test_parallel_repair_builder_matches_serial_counts_and_has_unique_ids(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    ca = build_repair(a, 135, seed=3, workers=1); cb = build_repair(b, 135, seed=3, workers=2)
    assert ca == cb
    ids = [json.loads(l)["id"] for f in b.glob("repair_*.jsonl") for l in open(f)]
    assert len(ids) == len(set(ids)) == sum(cb.values())

def test_data_report_flags_missing_splits_and_accepts_expert_rewards(tmp_path):
    build_repair(tmp_path, 90, seed=6); build_design(tmp_path, 4, seed=6, max_evals=6); build_audit(tmp_path, 6, seed=6); build_ledger(tmp_path, 30, seed=6)
    rep = data_report.report(tmp_path, check_rows=5)
    for task in ("repair", "design", "audit", "ledger"):
        assert task in rep and rep[task]["expert_reward_means"]
        assert not [p for p in rep["problems"] if "expert completions score" in p and p.startswith(task)]
    assert rep["repair"]["train_fault_types"] and "unit_slip" in rep["repair"]["train_fault_types"]

def test_thermoqa_answer_matching():
    assert thermoqa.correct("The enthalpy is about 2,676.2 kJ/kg", "2676") and not thermoqa.correct("It is 3000", "2676")
    assert thermoqa.correct("Answer: superheated vapor", "Superheated vapor") and thermoqa.last_number("no numbers") is None
    rows = [{"q": "x", "a": "10"}, {"q": "y", "a": "5"}]
    res = thermoqa.evaluate(lambda m: "10.1" if m[1]["content"] == "x" else "7", rows, "q", "a"); assert res["accuracy"] == 0.5

def test_builders_write_reproducibility_metadata_and_report_shows_it(tmp_path):
    build_repair(tmp_path, 45, seed=8); build_design(tmp_path, 3, seed=8, max_evals=5); build_audit(tmp_path, 4, seed=8); build_ledger(tmp_path, 10, seed=8)
    for t, key in (("repair", "n"), ("design", "max_evals"), ("audit", "seed"), ("ledger", "n")):
        m = json.loads((tmp_path / f"meta_{t}.json").read_text()); assert key in m["params"] and m["coolprop"] and m["schema_version"]
    rep = data_report.report(tmp_path, check_rows=2)
    assert set(rep["meta"]) == {"repair", "design", "audit", "ledger"} and not [p for p in rep["problems"] if "meta" in p or "CoolProp" in p]
    (tmp_path / "meta_audit.json").unlink()
    assert any("no meta_audit.json" in p for p in data_report.report(tmp_path, check_rows=2)["problems"])
