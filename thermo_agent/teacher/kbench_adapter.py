"""kbench adapter for the LangGraph CoT harness.

Bugs this fixes vs the old adapter:
  1. It delivered NO system_instructions and only messages[-1], so the frontier model never saw
     AGENTS.md, SKILL.md, failure memory or the "use tools to self-correct" instruction, and forgot
     its own candidate between rounds. Here the full system prompt is passed to chats.new and every
     prior message is replayed into one persistent chat.
  2. native_tool_agent already executes tools internally and keeps one chat across its tool rounds.
     Re-executing them in LangGraph's call_tools is wasteful and can desync the trajectory. So the
     adapter returns the COMPLETE trajectory (4-tuple) straight from the chat, and the graph
     finalizes immediately without a second execution.

The result: one persistent chat, the full prompt, tools executed exactly once, and a faithful
SFT trajectory (reasoning + tool calls + numeric results + revisions).
"""
from __future__ import annotations

from kaggle_benchmarks.tools.base import ToolInvocationLimitExhausted
from kaggle_benchmarks.tools.native import native_tool_agent


def _message_turns(chat) -> tuple[str, list[dict]]:
    """Read the assistant answers, tool calls, tool results and reasoning off the chat."""
    turns, reasoning = [], ""
    current = None
    for m in chat.messages:
        sender = getattr(m, "sender", None)
        name = getattr(sender, "name", "") if sender is not None else ""
        content = str(getattr(m, "content", "") or "")
        traces = getattr(m, "reasoning_traces", None)
        if traces:
            reasoning += str(traces) + "\n"
        calls = []
        for c in getattr(m, "tool_calls", None) or []:
            calls.append({"name": getattr(c, "name", "?"), "arguments": getattr(c, "arguments", {})})
        if name == "Assistant" or (calls and name not in ("User", "System", "Tool")):
            if current is not None:
                turns.append(current)
            current = {"assistant": content, "reasoning": str(traces) if traces else "",
                       "tool_calls": calls, "tool_results": []}
        elif name == "Tool":
            if current is None:
                current = {"assistant": "", "reasoning": "", "tool_calls": [], "tool_results": []}
            current["tool_results"].append(content)
    if current is not None:
        turns.append(current)
    return reasoning, turns


def make_respond(llm, kbench, actors, *, system_prompt: str, max_tool_rounds: int = 3,
                 usage_holder: dict | None = None):
    """Build respond(messages, tools) -> (text, tool_calls, reasoning, turns).

    `messages` is the full LangGraph conversation; the system message is installed as the
    kbench chat's system_instructions and the unique user task is sent once. native_tool_agent
    then owns the multi-round tool loop inside the single chat.
    """
    def respond(messages, tools):
        sys = next((m["content"] for m in messages if m["role"] == "system"), system_prompt)
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        with kbench.chats.new("agent", system_instructions=sys or None) as chat:
            actors.user.send(user)
            try:
                msg = native_tool_agent(llm, tools, max_tool_rounds=max_tool_rounds)
            except ToolInvocationLimitExhausted:
                actors.user.send("Stop calling tools. Return your final XML answer now.")
                msg = llm.respond()
            text = str(msg.content)
            reasoning, turns = _message_turns(chat)
            if not turns:
                turns = [{"assistant": text, "reasoning": reasoning.strip(),
                          "tool_calls": [], "tool_results": []}]
            if usage_holder is not None:
                usage_holder["chat"] = chat
            return text, [], reasoning.strip(), turns
    return respond


def usage_of(chat) -> dict:
    try:
        u = chat.usage
        return {"input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "total_cost_nanodollars": getattr(u, "total_cost_nanodollars", None),
                "total_backend_latency_ms": getattr(u, "total_backend_latency_ms", None)}
    except Exception:
        return {}
