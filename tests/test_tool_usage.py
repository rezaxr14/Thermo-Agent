"""Tool-use measurement for the harness-effectiveness arm of the paper."""
from thermo_agent.eval import tool_usage


def test_tool_usage_summarizes_calls_cost_and_failures():
    records = [
        {"id": "a", "task": "repair", "completion": "ok", "tool_calls": [{"name": "check_answer"}],
         "reasoning": "because ...", "usage": {"total_cost_nanodollars": 100}},
        {"id": "b", "task": "repair", "completion": "ok", "tool_calls": [], "usage": {}},
        {"id": "c", "task": "audit", "error": "quota", "tool_calls": []},
    ]
    out = tool_usage.summarize(records)
    assert out["by_task"]["repair"]["with_tools"] == 1
    assert out["by_task"]["repair"]["calls"] == 1
    assert out["by_task"]["repair"]["calls_per_answer"] == 0.5
    assert out["by_task"]["audit"]["errors"] == 1
    assert out["tools"] == {"check_answer": 1}
    assert out["usage"]["cost_nanodollars"] == 100


def test_interesting_picks_a_tool_heavy_and_a_failed_trace():
    records = [
        {"id": "a", "task": "repair", "tool_calls": [{"name": "check_answer"}] * 4, "reasoning": "long " * 20},
        {"id": "b", "task": "repair", "error": "quota"},
    ]
    picked = tool_usage.interesting(records, k=1)
    assert {r["id"] for r in picked} == {"a", "b"}
