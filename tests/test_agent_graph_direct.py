"""Hosted (langgraph-free) direct runner: import safety + the 4-tuple finalize path.

The Benchmarks task venv has NO langgraph. `run_direct` must therefore import and work
with langgraph absent, and must never execute the adapter's tools a second time.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

from thermo_agent.teacher import agent_graph
from thermo_agent.teacher.agent_graph import run_direct
from thermo_agent.teacher.toolkit import tool_names

ROOT = pathlib.Path(__file__).resolve().parents[1]


# Run in a clean subprocess so the langgraph blocker cannot leak into other tests.
_BLOCK_LANGGRAPH = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, r"{root}")

    class _BlockLangGraph:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "langgraph" or fullname.startswith("langgraph."):
                raise ImportError("langgraph is deliberately blocked: " + fullname)
            return None

    sys.meta_path.insert(0, _BlockLangGraph())

    from thermo_agent.teacher import agent_graph as ag

    assert callable(ag.build_context), "build_context must import without langgraph"
    assert callable(ag.finalize), "finalize must import without langgraph"
    assert callable(ag.run_direct), "run_direct must import without langgraph"

    # build_graph is the only langgraph consumer; it may (and does) fail here.
    try:
        ag.build_graph()
    except ImportError:
        pass
    else:
        raise AssertionError("build_graph unexpectedly succeeded with langgraph blocked")

    leaked = sorted(m for m in sys.modules if m == "langgraph" or m.startswith("langgraph."))
    assert not leaked, "langgraph leaked into sys.modules: %r" % leaked
    print("OK-NO-LANGGRAPH")
    """
).format(root=str(ROOT))


def test_import_works_with_langgraph_blocked():
    proc = subprocess.run([sys.executable, "-c", _BLOCK_LANGGRAPH],
                          cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK-NO-LANGGRAPH" in proc.stdout


@pytest.mark.corpus
def test_run_direct_finalizes_four_tuple_without_reexecuting_tools(monkeypatch):
    row = json.loads((ROOT / "data" / "processed" / "repair_train.jsonl")
                     .open(encoding="utf-8").readline())
    answer = row["completion"][0]["content"]

    counter = {"n": 0}

    def spy_tool(candidate):
        counter["n"] += 1
        return "SPY RESULT"

    spy_tool.__name__ = "spy_tool"
    monkeypatch.setattr(agent_graph, "make_tools", lambda task, row_: [spy_tool])

    # Pre-baked trajectory: one tool-using turn, then the final assistant answer.
    tool_call = {"name": "spy_tool", "arguments": {"candidate": "earlier candidate"}}
    turns = [
        {"assistant": "earlier candidate", "reasoning": "checking first law",
         "tool_calls": [tool_call],
         "tool_results": ['{"ok": true, "max_mass_residual_kg_s": 0.0}']},
        {"assistant": answer, "reasoning": "done", "tool_calls": [], "tool_results": []},
    ]

    def respond(messages, tools):
        # The adapter already ran the tool in its native loop. The 4-tuple still carries a
        # call; run_direct must discard it and finalize with the recorded turns.
        return answer, [tool_call], "done", turns

    state = run_direct(row, "repair", respond)

    assert state["final"] == answer
    assert state["done"] is True
    assert counter["n"] == 0, "run_direct executed a tool from the 4-tuple trajectory"
    sft = state["sft_completion"]
    assert "<tool_call" in sft and "<tool_result" in sft
    assert sft.rstrip().endswith(answer.rstrip())
    assert state["turns"] == turns
    assert isinstance(state["notes_used"], list)
    assert state["tools_available"] == tool_names()
    assert state["accepted"] is True
    assert state["verdict"]["accepted"] is True
