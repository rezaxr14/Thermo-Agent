"""The CoT data-gen harness is a LangGraph state machine using REAL numeric physics tools."""
import json

from thermo_agent.teacher.agent_graph import run
from thermo_agent.teacher.toolkit import make_tools, tool_names


def _row():
    return json.loads(open("data/processed/repair_train.jsonl", encoding="utf-8").readline())


def test_tools_are_real_solver_results():
    assert tool_names() == ["check_answer", "first_law_check", "second_law_check", "validity_check", "solve_plant"]
    row = _row()
    gt = row["completion"][0]["content"]
    t = {x.__name__: x for x in make_tools("repair", row)}
    first = json.loads(t["first_law_check"](gt))
    assert first["ok"] is True
    assert first["max_mass_residual_kg_s"] == 0.0
    assert first["max_energy_residual_W"] == 0.0
    second = json.loads(t["second_law_check"](gt))
    assert second["ok"] is True and "total_exergy_destruction_MW" in second
    solve = json.loads(t["solve_plant"](gt))
    assert solve["ok"] and solve["n_streams"] > 0 and "T_K" in solve["streams"][0]


def test_graph_self_corrects_and_records_trajectory():
    row = _row()
    gt = row["completion"][0]["content"]

    def respond(messages, tools):
        if any("Tool results:" in m["content"] for m in messages):
            return gt, [], "balances close"
        return gt, [{"name": "first_law_check", "arguments": {"candidate": gt}}], "checking"

    state = run(row, "repair", respond, max_rounds=3)
    assert state["accepted"] is True
    assert state["verdict"]["accepted"] is True
    sft = state["sft_completion"]
    assert "<tool_call" in sft and "<tool_result" in sft
    assert "mass_residual_kg_s" in sft  # a real numeric tool result is in the SFT text
    assert state["final"] == gt
