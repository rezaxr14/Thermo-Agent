"""The kbench adapter must hand the frontier model the full system prompt and the full
conversation history, and must not re-execute tools the internal tool loop already ran.

No network and no real model: kbench's chat/actors/LLM are faked, while the *real*
``native_tool_agent`` loop is driven so we can observe exactly what the model is shown.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import json

import kaggle_benchmarks
import pytest

from thermo_agent.teacher import agent_graph
from thermo_agent.teacher.agent_graph import build_context, run
from thermo_agent.teacher.kbench_adapter import _message_turns, make_respond
from thermo_agent.teacher.toolkit import tool_names

from _rows import make_row


# --------------------------------------------------------------------------- #
# fake kbench (chats / messages / actors / LLM)
# --------------------------------------------------------------------------- #
class _Holder:
    def __init__(self):
        self.current = None


class _Message:
    def __init__(self, content, sender, visible=True):
        self.content = content
        self.sender = sender
        self.is_visible_to_llm = visible
        self.reasoning_traces = None
        self.tool_calls = []


class _Actor:
    def __init__(self, holder, name, role):
        self._holder = holder
        self.name = name
        self.role = role

    def send(self, content, is_visible_to_llm=True):
        msg = _Message(content, self, is_visible_to_llm)
        self._holder.current.messages.append(msg)
        return msg


class _Actors:
    """Stands in for the ``kaggle_benchmarks.actors`` module the harness passes in."""

    def __init__(self, holder):
        self.user = _Actor(holder, "User", "user")
        self.Actor = lambda name=None, role="assistant", **_: _Actor(
            holder, name or "Assistant", role)
        self.Tool = lambda name="tool", **_: _Actor(holder, name, "tool")


class _Chat:
    def __init__(self, holder, name, system_instructions):
        self.name = name
        self.system_instructions = system_instructions
        self.messages = []
        # Real kbench.chats.new sends system_instructions as a System message in the chat.
        if system_instructions:
            self.messages.append(
                _Message(system_instructions, _Actor(holder, "System", "system")))


class _Chats:
    def __init__(self, holder, recorded):
        self._holder = holder
        self._recorded = recorded

    @contextmanager
    def new(self, name, system_instructions=None):
        chat = _Chat(self._holder, name, system_instructions)
        self._recorded.append({"name": name, "system_instructions": system_instructions})
        self._holder.current = chat
        try:
            yield chat
        finally:
            self._holder.current = None


class _Kbench:
    def __init__(self, holder, recorded):
        self.chats = _Chats(holder, recorded)


class _LLM:
    """Records every ``input_messages`` view the real native_tool_agent hands the model."""

    def __init__(self, holder, final="FINAL ANSWER"):
        self._holder = holder
        self.final = final
        self.calls = []

    def respond(self, **kwargs):
        self.calls.append(kwargs)
        # Real LLMChat.respond appends its reply to the current chat.
        msg = _Message(self.final, _Actor(self._holder, "frontier-model", "assistant"))
        self._holder.current.messages.append(msg)
        return msg


def _env(monkeypatch):
    holder, recorded = _Holder(), []
    actors = _Actors(holder)
    llm = _LLM(holder)
    kbench = _Kbench(holder, recorded)
    # native_tool_agent reads chats.get_current_chat() off the real kbench module.
    monkeypatch.setattr(kaggle_benchmarks.chats, "get_current_chat", lambda: holder.current)
    return holder, recorded, actors, llm, kbench


def _system_prompt(row):
    return build_context(
        {"row": row, "task": "repair", "notes": [], "mode": "agents+skills"}
    )["system_prompt"]


# --------------------------------------------------------------------------- #
# (a) + (b): the model sees the full prompt and the whole conversation
# --------------------------------------------------------------------------- #
def test_full_system_prompt_and_history_replayed(monkeypatch):
    row = make_row()
    sys_prompt = _system_prompt(row)
    _, recorded, actors, llm, kbench = _env(monkeypatch)
    respond = make_respond(llm, kbench, actors, system_prompt=sys_prompt)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "TASK TURN 1: repair the plant"},
        {"role": "assistant", "content": "ASSISTANT TURN 1: I will inventory the tools."},
        {"role": "user", "content": "USER TURN 2: use the numbers to revise"},
    ]
    text, calls, _reasoning, _turns = respond(messages, [])

    # (a) full system prompt (skills + guidance/compact memory + tool contract)
    # is delivered as system_instructions.
    assert len(recorded) == 1
    si = recorded[0]["system_instructions"]
    assert isinstance(si, str) and si.strip()
    assert si == sys_prompt
    assert "FIRST inventory" in si                  # compact_memory.agent_guidance_block
    assert "REAL physics tools" in si               # tool contract
    assert all(name in si for name in tool_names())

    # (b) every message (system + all user/assistant turns) is replayed to the
    # model, not just the last one.
    assert llm.calls, "the model was never called"
    view = llm.calls[0]["input_messages"]
    assert [m.content for m in view] == [
        sys_prompt,
        "TASK TURN 1: repair the plant",
        "ASSISTANT TURN 1: I will inventory the tools.",
        "USER TURN 2: use the numbers to revise",
    ]
    assert len(view) == len(messages)  # 1 system + 3 non-system turns
    assert text == "FINAL ANSWER"
    assert calls == []                 # the adapter never asks LangGraph to re-run tools


# --------------------------------------------------------------------------- #
# (c): a 4-tuple finalizes immediately; tools are not executed a second time
# --------------------------------------------------------------------------- #
def test_four_tuple_finalizes_without_reexecuting_tools(monkeypatch):
    row = make_row()
    _, _, actors, llm, kbench = _env(monkeypatch)
    base = make_respond(llm, kbench, actors, system_prompt=_system_prompt(row))

    counter = {"n": 0}

    def spy_tool(candidate):
        counter["n"] += 1
        return "SPY RESULT"

    spy_tool.__name__ = "spy_tool"
    # make_tools is a module global looked up inside build_context.
    monkeypatch.setattr(agent_graph, "make_tools", lambda task, row_: [spy_tool])

    def respond(messages, tools):
        text, _calls, reasoning, turns = base(messages, tools)
        # A 4-tuple may carry the tool calls the internal loop already executed.
        return text, [{"name": "spy_tool", "arguments": {"candidate": "REEXEC_MARKER"}}], reasoning, turns

    state = run(row, "repair", respond)

    assert counter["n"] == 0, "LangGraph executed the tools a second time"
    assert state["done"] is True
    assert state["rounds"] == 0
    assert state["last_calls"] == []   # the 4-tuple's calls were discarded, not run
    # Per-turn visible assistant prose IS the reasoning for chat-completions models.
    assert state["turns"] == [{"assistant": "FINAL ANSWER", "reasoning": "FINAL ANSWER",
                               "tool_calls": [], "tool_results": []}]


def test_spy_would_catch_a_second_execution(monkeypatch):
    """Control: the spy above really does observe a tool run on the 3-tuple path."""
    row = make_row()
    counter = {"n": 0}

    def spy_tool(candidate):
        counter["n"] += 1
        return "SPY RESULT"

    spy_tool.__name__ = "spy_tool"
    monkeypatch.setattr(agent_graph, "make_tools", lambda task, row_: [spy_tool])

    def respond(messages, tools):
        if any("Tool results:" in m["content"] for m in messages):
            return "FINAL ANSWER", [], "done"
        return "candidate", [{"name": "spy_tool", "arguments": {"candidate": "x"}}], "checking"

    state = run(row, "repair", respond)
    assert counter["n"] == 1
    assert state["rounds"] == 1


# --------------------------------------------------------------------------- #
# regression: dispatch on sender.role, not sender.name (the pilot bug)
# --------------------------------------------------------------------------- #
def test_message_turns_dispatch_on_role_captures_tool_result_and_final_answer():
    """kbench names a tool sender after the tool ("first_law_check", not "Tool") and the
    model actor after its slug (not "Assistant"). Dispatching on sender.name therefore lost
    both the tool result and the final assistant answer, leaving sft_completion empty."""
    row = make_row()
    final_xml = row["completion"][0]["content"]
    tool_json = json.dumps({"ok": True, "max_mass_residual_kg_s": 0.0,
                            "max_energy_residual_W": 0.0})
    chat = SimpleNamespace(messages=[
        # assistant message with tool_calls and empty content, as kbench builds it
        SimpleNamespace(
            content="",
            sender=SimpleNamespace(role="assistant", name="frontier-model"),
            reasoning_traces=None,
            tool_calls=[SimpleNamespace(name="first_law_check",
                                        arguments={"candidate": "draft"})]),
        # tool result sender: role="tool", name = the tool's name (NOT "Tool")
        SimpleNamespace(content=tool_json,
                        sender=SimpleNamespace(role="tool", name="first_law_check"),
                        reasoning_traces=None, tool_calls=[]),
        # final assistant message: role="assistant", name = model slug (NOT "Assistant")
        SimpleNamespace(content=final_xml,
                        sender=SimpleNamespace(role="assistant", name="frontier-model"),
                        reasoning_traces=None, tool_calls=[]),
    ])

    reasoning, turns = _message_turns(chat)

    assert len(turns) == 2
    assert turns[0]["tool_calls"][0]["name"] == "first_law_check"
    assert turns[0]["tool_calls"][0]["arguments"] == {"candidate": "draft"}
    assert tool_json in turns[0]["tool_results"]
    assert turns[1]["assistant"] == final_xml

    state = {"task": "repair", "row": row, "last_text": final_xml,
             "last_reasoning": reasoning, "turns": turns}
    sft = agent_graph.finalize(state)["sft_completion"]
    assert "<tool_call" in sft
    assert "<tool_result" in sft
    assert tool_json in sft
    assert sft.rstrip().endswith(final_xml)
