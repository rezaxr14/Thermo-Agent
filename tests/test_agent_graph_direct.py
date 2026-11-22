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

from thermo_agent.teacher import agent_graph, transfer
from thermo_agent.teacher.agent_graph import finalize, run_direct
from thermo_agent.teacher.toolkit import SKILL_TOOL_NAMES, tool_names
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES

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
    monkeypatch.setattr(agent_graph, "make_tools", lambda task, row_, **kw: [spy_tool])

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
    assert "<tool_call" in sft and "<tool_response" in sft
    assert sft.rstrip().endswith(answer.rstrip())
    assert state["turns"] == turns
    assert isinstance(state["notes_used"], list)
    assert state["tools_available"] == tool_names()
    # The 16 tools are the default; the legacy 5-tool set is the opt-out.
    assert tool_names() == tool_names(include_dynamic=True)
    assert tool_names() == tool_names(include_dynamic=False) + list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES)
    assert state["accepted"] is True
    assert state["verdict"]["accepted"] is True


@pytest.mark.corpus
def test_run_direct_reports_dynamic_tools_when_opted_in():
    """The hosted runner's include_dynamic flag reaches run_direct's tool inventory."""
    row = json.loads((ROOT / "data" / "processed" / "repair_train.jsonl")
                     .open(encoding="utf-8").readline())
    answer = row["completion"][0]["content"]
    seen = {}

    def respond(messages, tools):
        seen["tools"] = [t.__name__ for t in tools]
        seen["system"] = messages[0]["content"]
        # 4-tuple path: the adapter already executed the tools; the graph must not re-run them.
        return answer, [], "done", [{"assistant": answer, "reasoning": "",
                                     "tool_calls": [], "tool_results": []}]

    state = run_direct(row, "repair", respond, include_dynamic=True)
    assert seen["tools"] == tool_names(include_dynamic=True)
    assert len(seen["tools"]) == 16
    # The tool list is injected by the chat template, not duplicated in prompt text.
    assert '<tool_call name=' not in seen["system"]
    assert state["tools_available"] == tool_names(include_dynamic=True)


# --------------------------------------------------------------------------- #
# Hosted prompt shards: input-only rows defer local validation, never raise.
# --------------------------------------------------------------------------- #
# Mirrors scripts/make_hosted_prompts.FIELD_ALLOWLIST: tool inputs + prompt,
# with every answer-bearing scoring column stripped by the shard builder.
SHARD_ALLOWLIST = {
    "repair": ("id", "split", "topology", "prompt", "plant_broken", "site", "task"),
    "audit": ("id", "split", "topology", "prompt", "plant", "site", "task"),
    "ledger": ("id", "split", "topology", "prompt", "ledger", "task"),
    "design": ("id", "split", "prompt", "site", "task"),
}


def _fixture_row(task: str) -> dict:
    """A committed, full row (all scoring fields + ground-truth completion)."""
    path = ROOT / "tests" / "fixtures" / "corpus" / f"{task}_train.jsonl"
    return json.loads(path.open(encoding="utf-8").readline())


def _shard_row(row: dict, task: str) -> dict:
    return {k: row[k] for k in SHARD_ALLOWLIST[task] if k in row}


def _fake_respond(text: str):
    """The 4-tuple shape the real kbench adapter hands back to run_direct."""
    turns = [{"assistant": text, "reasoning": "fake reasoning about the tool numbers shown",
              "tool_calls": [], "tool_results": []}]

    def respond(messages, tools):
        return text, [], "fake reasoning about the tool numbers shown", turns

    return respond


def test_audit_row_without_ed_json_finalizes_deferred_with_full_trajectory():
    """The exact failure class from the 99-row journal: no ed_json -> used to raise KeyError."""
    row = _shard_row(_fixture_row("audit"), "audit")
    assert "ed_json" not in row and "fuel_w" not in row
    text = ("<ranking>component-1 component-2</ranking>\n"
            "<change>SET component-1.eta = 0.9</change>\n"
            "<prediction>0.3</prediction>\n"
            "<analysis>the plant streams give this effect</analysis>")
    turns = [{"assistant": text, "reasoning": "fake reasoning about the tool numbers shown",
              "tool_calls": [], "tool_results": []}]

    # Direct finalize (the LangGraph path) must not raise and must still record SFT text.
    out = finalize({"task": "audit", "row": row, "last_text": text,
                    "last_reasoning": "fake reasoning about the tool numbers shown", "turns": turns})
    assert out["done"] is True
    assert out["final"] == text
    assert out["turns"] == turns
    assert out["sft_completion"].strip()
    assert out["sft_completion"].rstrip().endswith(text)
    assert out["accepted"] is False
    assert out["verdict"]["deferred"] is True
    assert out["verdict"]["accepted"] is False
    assert out["verdict"]["reward"] == 0.0
    assert "ed_json" in out["verdict"]["reasons"][0]

    # The hosted driver returns the same full trajectory and never skips the row.
    state = run_direct(row, "audit", _fake_respond(text))
    assert state["done"] is True
    assert state["final"] == text
    assert state["sft_completion"].strip()
    assert state["turns"] == turns
    assert state["verdict"]["deferred"] is True
    assert isinstance(state["notes_used"], list)
    assert state["served_skill"]
    assert len(state["system_prompt_sha256"]) == 64
    assert state["tools_available"] == tool_names()


def test_complete_row_still_gets_the_real_verifier_reward():
    """Adding the deferral path must not weaken real validation on local corpora."""
    row = _fixture_row("audit")
    assert all(k in row for k in ("ed_json", "fuel_w", "net_w"))
    text = row["completion"][0]["content"]
    out = finalize({"task": "audit", "row": row, "last_text": text,
                    "last_reasoning": "", "turns": []})
    assert out["verdict"]["deferred"] is False
    assert out["verdict"]["accepted"] is True
    assert out["verdict"]["reward"] > 0.9
    assert out["accepted"] is True


@pytest.mark.parametrize("task", ("repair", "ledger", "audit", "design"))
def test_shard_shaped_row_finalizes_for_every_task(task):
    """All four shards omit answer fields; each must finalize with a full trajectory."""
    row = _shard_row(_fixture_row(task), task)
    text = f"<answer>plausible {task} answer</answer>"
    state = run_direct(row, task, _fake_respond(text))
    assert state["done"] is True
    assert state["final"] == text
    assert state["sft_completion"].strip()
    assert state["sft_completion"].rstrip().endswith(text)
    assert state["turns"]
    assert isinstance(state["notes_used"], list)
    assert state["served_skill"]
    assert len(state["system_prompt_sha256"]) == 64
    assert state["verdict"]["deferred"] is True
    assert state["verdict"]["reasons"][0].startswith("validation deferred: missing scoring fields")


def test_ingest_grades_a_deferred_row_once_scoring_fields_are_joined(tmp_path):
    """transfer.ingest joins the answer-free journal row to the full export by id and grades it."""
    task = "audit"
    row = _fixture_row(task)
    export = tmp_path / "export"
    export.mkdir()
    (export / f"{task}_train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    # The hosted run produces a deferred verdict but a full trajectory/journal record.
    shard = _shard_row(row, task)
    text = row["completion"][0]["content"]
    state = run_direct(shard, task, _fake_respond(text))
    assert state["verdict"]["deferred"] is True
    completions = tmp_path / "completions.jsonl"
    completions.write_text(json.dumps({
        "id": shard["id"], "record_id": shard["id"], "task": task,
        "completion": state["final"],
        "sft_completion": state["sft_completion"],
        "trajectory": {"turns": state["turns"], "notes_used": state["notes_used"],
                       "final": state["final"], "served_skill": state["served_skill"],
                       "system_prompt_sha256": state["system_prompt_sha256"],
                       "memory_block": state["memory_block"]},
    }) + "\n", encoding="utf-8")

    report = transfer.ingest(export, completions, tmp_path / "frontier", model="test-model")
    assert report["attempts"] == 1
    assert report["accepted"] == 1
    produced = [json.loads(line) for line in
                (tmp_path / "frontier" / f"frontier_{task}_train.jsonl").open(encoding="utf-8")]
    assert len(produced) == 1
    assert produced[0]["source"] == "frontier:test-model"
    assert produced[0]["validation"]["reward"] > 0.9
    assert produced[0]["completion"][0]["content"].endswith(text)  # trajectory SFT text kept
    assert produced[0]["trajectory"]["final"] == text
    assert produced[0]["trajectory"]["served_skill"] == state["served_skill"]

