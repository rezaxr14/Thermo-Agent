"""Reasoning replay contract for the multi-turn teacher agent.

The assistant's own per-turn text IS the reasoning trace on chat-completions models:
it must survive capture (adapter), replay (reflect), serialization (finalize) and the
capture audits (show_trace / teacher_campaign_stats) — but raw trajectory markup must
never leak into the compact failure memory the next episode is served.

No network: kbench messages are faked and the verifier is monkeypatched.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# kbench builds a default model at import time when MODEL_PROXY_URL is set (the repo's
# .env sets it). A harmless model name keeps the import offline; nothing is requested.
os.environ.setdefault("LLM_DEFAULT", "gpt-6-luna")

from thermo_agent.teacher import agent_graph
from thermo_agent.teacher.compact_memory import _clean, compact_notes
from thermo_agent.teacher.kbench_adapter import _message_turns, reasoning_for_model
from thermo_agent.teacher.memory import Note

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import teacher_campaign_stats as stats  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "show_trace_replay", ROOT / "scripts" / "show_trace.py")
show_trace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(show_trace)

VISIBLE = "I will check the compressor pressure ratio and exergy destruction first."
FINAL_XML = "<repair>candidate PR = 12</repair>"
TOOL_JSON = json.dumps({"ok": True, "compressor_pr": 12.0, "exergy_destruction_w": 1500.0})


def _msg(content, role="assistant", name="frontier-model", traces=None, calls=()):
    return SimpleNamespace(
        content=content,
        sender=SimpleNamespace(role=role, name=name),
        reasoning_traces=traces,
        tool_calls=[SimpleNamespace(name=c["name"], arguments=c["arguments"]) for c in calls],
    )


def _chat(*messages):
    return SimpleNamespace(messages=list(messages))


# --------------------------------------------------------------------------- #
# 1. capture: visible prose becomes reasoning; tool I/O stays with its turn
# --------------------------------------------------------------------------- #
def test_message_turns_mirrors_visible_prose_and_keeps_tool_io():
    chat = _chat(
        _msg(VISIBLE, calls=[{"name": "first_law_check", "arguments": {"candidate": "draft"}}]),
        _msg(TOOL_JSON, role="tool", name="first_law_check"),
        _msg(FINAL_XML),
    )
    reasoning, turns = _message_turns(chat)

    assert len(turns) == 2
    assert turns[0]["assistant"] == VISIBLE
    assert turns[0]["reasoning"] == VISIBLE          # no hidden trace -> mirror visible prose
    assert turns[0]["tool_calls"][0]["name"] == "first_law_check"
    assert turns[0]["tool_calls"][0]["arguments"] == {"candidate": "draft"}
    assert TOOL_JSON in turns[0]["tool_results"]      # result attached to the calling turn
    assert turns[1]["assistant"] == FINAL_XML
    assert turns[1]["reasoning"] == FINAL_XML
    assert VISIBLE in reasoning and FINAL_XML in reasoning  # aggregate replay transcript


# --------------------------------------------------------------------------- #
# 2. a provider's hidden trace wins over the visible text
# --------------------------------------------------------------------------- #
def test_hidden_trace_wins_over_visible_prose():
    _reasoning, turns = _message_turns(_chat(_msg("visible", traces="hidden chain")))
    assert len(turns) == 1
    assert turns[0]["assistant"] == "visible"
    assert turns[0]["reasoning"] == "hidden chain"


# --------------------------------------------------------------------------- #
# 3. replay: reflect hands the model its own reasoning exactly once + the numbers
# --------------------------------------------------------------------------- #
def _reflect_state(last_text, last_reasoning):
    turn = {
        "assistant": last_text,
        "reasoning": last_reasoning,
        "tool_calls": [{"name": "first_law_check", "arguments": {"candidate": last_text}}],
        "tool_results": [TOOL_JSON],
    }
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    state = {"messages": messages, "turns": [turn], "last_text": last_text,
             "last_reasoning": last_reasoning, "rounds": 0}
    return state


def test_reflect_replays_hidden_reasoning_once_and_attaches_tool_result():
    hidden = "hidden chain about pressure ratios"
    state = _reflect_state(VISIBLE, hidden)
    out = agent_graph.reflect(state)

    assistant_msg, user_msg = out["messages"][-2:]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"].count(hidden) == 1
    assert "<think>" in assistant_msg["content"]
    assert VISIBLE in assistant_msg["content"]
    assert user_msg["role"] == "user"
    assert TOOL_JSON in user_msg["content"]
    assert "Tool results:" in user_msg["content"]
    assert out["rounds"] == 1
    assert len(state["messages"]) == 2        # original conversation left intact


def test_reflect_does_not_duplicate_reasoning_equal_to_assistant_text():
    state = _reflect_state(VISIBLE, VISIBLE)
    content = agent_graph.reflect(state)["messages"][-2]["content"]
    assert content.count(VISIBLE) == 1
    assert "<think>" not in content


# --------------------------------------------------------------------------- #
# 4. serialization: SFT text emits each turn's reasoning once, with tool I/O
# --------------------------------------------------------------------------- #
def test_finalize_sft_emits_each_reasoning_once(monkeypatch):
    verdict = {"accepted": True, "reward": 1.0, "reasons": []}
    monkeypatch.setattr("thermo_agent.teacher.validate.validate", lambda *a, **k: verdict)

    same = "I will check the compressor pressure ratio."
    hidden = "Residuals are zero, so the candidate stands."
    turns = [
        {"assistant": same, "reasoning": same,       # chat-completions turn
         "tool_calls": [{"name": "first_law_check", "arguments": {"candidate": "draft"}}],
         "tool_results": [TOOL_JSON]},
        {"assistant": FINAL_XML, "reasoning": hidden,  # turn with a hidden trace
         "tool_calls": [], "tool_results": []},
    ]
    state = {"task": "repair", "row": {}, "last_text": FINAL_XML,
             "last_reasoning": hidden, "turns": turns}

    out = agent_graph.finalize(state)
    sft = out["sft_completion"]
    assert sft.count(same) == 1
    assert sft.count(hidden) == 1
    assert "<tool_call" in sft and "<tool_result>" in sft
    assert TOOL_JSON in sft
    assert sft.rstrip().endswith(FINAL_XML)
    assert out["verdict"]["accepted"] is True


# --------------------------------------------------------------------------- #
# 5. reasoning level must be a valid string (or None) alongside function tools
# --------------------------------------------------------------------------- #
def test_reasoning_for_model_levels():
    assert reasoning_for_model("gpt-6-luna") == "none"
    assert reasoning_for_model("openai/gpt-6.2-sol") == "none"
    assert reasoning_for_model("claude-haiku-5-5") is None
    assert reasoning_for_model("google/gemini-3.8-flash") is None
    for model in ("gpt-6-luna", "claude-haiku-5-5", "openai/gpt-5.6-luna", ""):
        assert reasoning_for_model(model) is not True


# --------------------------------------------------------------------------- #
# 6. served memory must never carry raw trajectory markup or bookkeeping
# --------------------------------------------------------------------------- #
def test_compact_notes_strip_trajectory_markup_and_bookkeeping():
    raw_blocks = [
        "<think>secret chain</think>",
        '<tool_call>{"candidate":"draft","model":"gpt-6-luna"}</tool_call>',
        '<tool_result>{"ok":false,"ts":"2026-01-14","cost":0.25,'
        '"tokens":900,"latency":1200}</tool_result>',
    ]
    embedded = " " + " ".join(raw_blocks) + " "
    note = Note(
        id="n-1", task="repair", fault_type="UNIT MISMATCH", architecture="cycles",
        symptoms="residual too high",
        lesson="Attempt rejected (format)." + embedded + " Correct pattern: recheck units.",
        fix="Use kJ/kg consistently and re-check the first-law residual." + embedded +
            " ts=2026-01-14 cost=0.25 tokens=900 latency=1200 gpt-6-luna google-gemini-3.8-flash",
        source="frontier:gpt-6-luna", accepted=False, record_id="row-1",
        extra={"cost": 1.0, "latency_ms": 1200},
    )

    block = compact_notes([note], k=1)
    low = block.lower()
    assert "UNIT MISMATCH" in block               # the reusable fault class survives
    assert "Use kJ/kg consistently" in block      # the reusable fix survives
    for raw in raw_blocks:
        assert raw not in block
    assert "<think>" not in low and "secret chain" not in low
    assert "<tool_call>" not in low and '"candidate"' not in low
    assert "<tool_result>" not in low and '"ok":false' not in low
    assert "draft" not in block
    for model in ("gpt-6-luna", "gemini", "claude", "haiku", "luna"):
        assert model not in low
    assert "2026-01-14" not in block
    for field in ("cost=", "tokens=", "latency", "ts="):
        assert field not in low

    # The same cleanup applies to any lesson/fix field, not only via compact_notes.
    assert _clean("keep me <THINK>drop</THINK> <TOOL_CALL>{}</TOOL_CALL> "
                  "<tool_result>{}</tool_result>") == "keep me"


# --------------------------------------------------------------------------- #
# 7. capture audits count assistant prose as reasoning when reasoning is empty
# --------------------------------------------------------------------------- #
def test_capture_audits_count_assistant_prose_as_reasoning():
    row = {
        "id": "rep-1", "record_id": "rep-1", "task": "repair",
        "model": "gpt-6-luna", "provider": "openai", "run_id": "run-1",
        "source": "frontier:gpt-6-luna",
        "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
        "trajectory": {
            "final": FINAL_XML,
            "notes_used": [], "tools_available": [],
            "turns": [{"assistant": VISIBLE, "reasoning": "",
                       "tool_calls": [], "tool_results": []}],
        },
        "validation": {"reward": 1.0, "reasons": []},
    }

    text = show_trace.render(row, {})
    assert "reasoning: " + VISIBLE in text
    assert "reasoning turns captured : 1/1" in text
    assert "assistant prose turns    : 1/1" in text

    # teacher_campaign_stats applies the same rule to journal rows.
    assert stats.has_thinking(row) is True
    assert stats.has_thinking({"trajectory": {"turns": [
        {"assistant": "", "reasoning": ""}]}}) is False
