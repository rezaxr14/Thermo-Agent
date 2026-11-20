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
    """Read the assistant answers, tool calls, tool results and reasoning off the chat.

    The per-turn assistant text IS the reasoning trace for chat-completions models
    (luna runs with ``reasoning_effort="none"`` and puts its CoT in the visible
    assistant message). Only when the provider returns a separate hidden trace do
    we prefer that trace. Keeping the visible text as ``reasoning`` is what lets a
    later turn replay the model's own prior reasoning, tool use and momentum.
    """
    turns, reasoning = [], ""
    current = None
    for m in chat.messages:
        sender = getattr(m, "sender", None)
        # kbench names a tool sender after the tool (e.g. "first_law_check", not "Tool") and
        # the model actor after its slug (not "Assistant"). The stable discriminator is the
        # actor's role, not its name.
        role = getattr(sender, "role", "") if sender is not None else ""
        content = str(getattr(m, "content", "") or "")
        traces = getattr(m, "reasoning_traces", None)
        trace_text = str(traces).strip() if traces else ""
        # Visible per-turn prose is the reasoning when the provider hides none.
        turn_reasoning = trace_text or content.strip()
        if turn_reasoning:
            reasoning = (reasoning + "\n" + turn_reasoning).strip()
        calls = []
        for c in getattr(m, "tool_calls", None) or []:
            calls.append({"name": getattr(c, "name", "?"), "arguments": getattr(c, "arguments", {})})
        if role == "assistant" or (calls and role not in ("user", "system", "tool")):
            if current is not None:
                turns.append(current)
            current = {"assistant": content, "reasoning": turn_reasoning,
                       "tool_calls": calls, "tool_results": []}
        elif role == "tool":
            if current is None:
                current = {"assistant": "", "reasoning": "", "tool_calls": [], "tool_results": []}
            current["tool_results"].append(content)
    if current is not None:
        turns.append(current)
    return reasoning, turns


def reasoning_for_model(model: str) -> str | None:
    """Reasoning level that is valid alongside native function tools.

    gpt-6-luna (and the rest of the gpt-6 chat-completions family) rejects
    ``tools`` + ``reasoning_effort`` on /v1/chat/completions with a hard 400:

        Function tools with reasoning_effort are not supported ... To use function
        tools, use /v1/responses or set reasoning_effort to 'none'.

    kbench's native tool loop always goes through chat.completions, so the only
    supported combination on that path is ``reasoning_effort="none"``. Returning
    None means "do not override the backend default" (correct for models that do
    support tools and reasoning together, e.g. Claude and Gemini).
    """
    m = str(model or "").lower()
    if m.startswith(("openai/gpt-6", "openai/gpt-6.", "gpt-6", "openai/gpt-5.6")):
        return "none"
    return None


def make_respond(llm, kbench, actors, *, system_prompt: str, max_tool_rounds: int = 3,
                 usage_holder: dict | None = None, reasoning: str | None = None):
    """Build respond(messages, tools) -> (text, tool_calls, reasoning, turns).

    `messages` is the full LangGraph conversation; the system message is installed as the
    kbench chat's system_instructions and the unique user task is sent once. native_tool_agent
    then owns the multi-round tool loop inside the single chat.

    `reasoning` is forwarded to kbench only when it is not None, so models that accept
    tools and reasoning together keep their traces; gpt-6 chat-completions needs "none".
    """
    tool_kwargs = {"max_tool_rounds": max_tool_rounds}
    if reasoning is not None:
        tool_kwargs["reasoning"] = reasoning

    def respond(messages, tools):
        sys = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or system_prompt
        with kbench.chats.new("agent", system_instructions=sys or None) as chat:
            # Replay the WHOLE conversation into this one chat, not just the last message:
            # system text is already installed as system_instructions by chats.new; every
            # user/assistant turn is sent with its own role so the model keeps its history.
            for m in messages:
                role = m.get("role")
                if role == "system":
                    continue
                if role == "assistant":
                    actors.Actor(name="Assistant", role="assistant").send(m.get("content", ""))
                else:
                    actors.user.send(m.get("content", ""))
            try:
                msg = native_tool_agent(llm, tools, **tool_kwargs)
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
