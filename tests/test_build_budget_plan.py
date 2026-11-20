"""Plans must cover every task/tier cell with equal thirds, for thousands of traces.

History: the plan files held only the handful of cells that were missing at the
time, so the overnight loop ran out of rows after one more batch.
"""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_budget_plan", ROOT / "scripts" / "build_budget_plan.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _plant(n: int) -> dict:
    return {"components": [{"id": f"c{i}"} for i in range(n)]}


def _ledger(n: int) -> str:
    return json.dumps({"data": {"topo": {f"c{i}": {} for i in range(n)}}})


def _prompts():
    rows = []
    for i in range(6):
        rows.append({"id": f"rep-{i}", "task": "repair", "plant_broken": _plant(i + 1)})
        rows.append({"id": f"aud-{i}", "task": "audit", "plant": _plant(i + 1)})
        rows.append({"id": f"led-{i}", "task": "ledger", "ledger": _ledger(i + 1)})
    return rows


def test_plan_covers_all_12_task_tier_cells():
    planned, unscored = mod.plan_model(_prompts(), {}, "gpt-6-luna")
    assert unscored == []
    cells = {r["cell"] for r in planned}
    assert cells == {f"{t}/{tier}" for t in ("repair", "audit", "ledger")
                     for tier in ("easy", "medium", "hard")}
    assert len(planned) == 18


def test_plan_is_equal_thirds_per_task():
    planned, _ = mod.plan_model(_prompts(), {}, "gpt-6-luna")
    for task in ("repair", "audit", "ledger"):
        tiers = [r["tier"] for r in planned if r["task"] == task]
        assert tiers.count("easy") == tiers.count("medium") == 2
        assert tiers.count("hard") == 2


def test_design_without_ref_spec_is_reported_not_invented():
    rows = [{"id": "des-1", "task": "design", "site": "{}"}]
    planned, unscored = mod.plan_model(rows, {}, "gpt-6-luna")
    assert planned == []
    assert unscored == ["des-1"]


def test_design_specs_join_is_read_from_the_reference_file(tmp_path):
    path = tmp_path / "design_train.jsonl"
    path.write_text(json.dumps({"id": "des-1", "ref_spec": {"x": 1}}) + "\n", encoding="utf-8")
    assert mod.design_specs(path) == {"des-1": '{"x": 1}'}


def test_scored_row_adds_ref_spec_only_for_design():
    row = {"id": "des-1", "task": "design", "site": "{}"}
    out = mod.scored_row("design", row, {"des-1": '{"x": 1}'})
    assert out["ref_spec"] == '{"x": 1}'
    assert "ref_spec" not in row, "the compact prompt must not be mutated in place"
    # Non-design rows are returned untouched.
    repair = {"id": "rep-1", "task": "repair"}
    assert mod.scored_row("repair", repair, {}) is repair


def test_plan_output_is_deterministic():
    first, _ = mod.plan_model(_prompts(), {}, "gpt-6-luna")
    second, _ = mod.plan_model(list(reversed(_prompts())), {}, "gpt-6-luna")
    assert [r["id"] for r in first] == [r["id"] for r in second]
