"""LangGraph state machine for the self-correcting CoT agent.

   START -> build_context -> propose -> decide
                                ^          | (tools) -> call_tools -> reflect -> propose
                                |          | (done)  -> finalize -> END

build_context  AGENTS.md + component/task SKILL.md + freshly retrieved failure memory + tool contract
propose        the model emits a candidate answer and (optionally) tool calls
call_tools     executes the REAL thermodynamics tools (first/second law, validity, solve) on the candidate
reflect        appends the numeric tool results to the conversation so the model can self-correct
finalize       records the verifier-accepted answer and the full trajectory for SFT

The graph is model-agnostic: `respond(state) -> {text, tool_calls, reasoning}` is injected.
Every transition is a node so the whole thing is inspectable, resumable and testable.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from ..agent.harness import augment
from ..core.schema import Plant
from . import memory as mem
from .toolkit import make_tools, tool_names


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class Turn(TypedDict, total=False):
    assistant: str
    reasoning: str
    tool_calls: list[dict]
    tool_results: list[str]


class AgentState(TypedDict, total=False):
    # inputs
    row: dict
    task: str
    notes: list
    max_rounds: int
    k_memory: int
    mode: str
    respond: Any            # injected model callable (declared so LangGraph keeps it in state)
    last_tool_results: list[str]
    route: str
    # tools (bound to row, not serialised into checkpoints)
    tools: list
    tool_map: dict
    # working
    messages: list[dict]
    system_prompt: str
    turns: list[Turn]
    notes_used: list[str]
    rounds: int
    last_text: str
    last_calls: list[dict]
    last_reasoning: str
    # output
    final: str
    done: bool
    accepted: bool
    verdict: dict
    sft_completion: str


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #
def build_context(state: AgentState) -> dict:
    """System prompt = AGENTS.md + skills + fresh failure memory + tool contract."""
    row, task = state["row"], state["task"]
    original = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] == "system")
    plant = None
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                plant = Plant.model_validate_json(row[key]); break
            except Exception:
                pass
    sys = augment([{"role": "system", "content": original}], state.get("mode", "agents+skills"), task, plant)[0]["content"]
    notes = state.get("notes") or []
    # D10/D11: guidance first (inventory skills/tools, then examine the cycle against memory),
    # then a TOKEN-EFFICIENT memory that carries only what failed and what works — no model ids,
    # timestamps, cost or run logs. The full ledger stays on disk for our own analysis.
    from .compact_memory import agent_guidance_block, compact_notes
    sys += "\n\n" + agent_guidance_block()
    if notes:
        compact = compact_notes(mem.retrieve(
            " ".join(m["content"] for m in row["prompt"]), notes,
            k=state.get("k_memory", 3), task=task, exclude_record=str(row.get("id", ""))),
            k=state.get("k_memory", 3))
        if compact:
            sys += "\n\n" + compact
    sys += (
        "\n\nYou have REAL physics tools and you SHOULD use them to solve and self-correct. "
        "Call them on candidate answers and USE the returned numbers (mass/energy residuals in kg/s and W, "
        "exergy destruction in MW, Carnot/COP limits, stream states) to find what is wrong, revise, and call "
        "again (up to %d rounds). Tools: %s. You also have skills and retrieved failure memory above: use them "
        "to choose a better design/repair. Then STOP calling tools and return the final answer in the task's "
        "XML tags exactly as specified above. Never invent numbers; use tool output, skills and memory."
    ) % (state.get("max_rounds", 4), ", ".join(tool_names()))
    tools = make_tools(task, row)
    notes_used = [n.id for n in (mem.retrieve(" ".join(m["content"] for m in row["prompt"]), notes,
                                             k=state.get("k_memory", 3), task=task) if notes else [])]
    user = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] != "system")
    return {
        "system_prompt": sys, "tools": tools, "tool_map": {t.__name__: t for t in tools},
        "notes_used": notes_used,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        "turns": [], "rounds": 0, "done": False,
    }


def propose(state: AgentState) -> dict:
    """The injected model produces a candidate + tool calls.

    The adapter may return either (text, calls, reasoning) — in which case LangGraph drives the
    tool loop via call_tools/reflect — or (text, calls, reasoning, turns), where `turns` is the
    complete trajectory already produced by an internal tool loop (e.g. kbench native_tool_agent).
    In the latter case the tools were executed exactly once and we finalize immediately.
    """
    out = state["respond"](state["messages"], state["tools"])
    if len(out) == 4:
        text, calls, reasoning, turns = out
        return {"last_text": text or "", "last_calls": [], "last_reasoning": reasoning or "",
                "turns": turns}
    text, calls, reasoning = out
    return {"last_text": text or "", "last_calls": calls or [], "last_reasoning": reasoning or ""}


def decide(state: AgentState) -> str:
    """Conditional-edge path: tool calls and rounds left -> call_tools; else finalize."""
    if state.get("last_calls") and state.get("rounds", 0) < state.get("max_rounds", 4):
        return "call_tools"
    return "finalize"


def call_tools(state: AgentState) -> dict:
    """Execute the real tools; append results to the trajectory."""
    results, calls_rec = [], []
    for c in state["last_calls"]:
        name = c.get("name", "")
        args = c.get("arguments", {})
        candidate = args.get("candidate") if isinstance(args, dict) else None
        if candidate is None:
            candidate = state["last_text"]
        fn = state["tool_map"].get(name)
        result = fn(candidate) if fn else json.dumps({"error": f"unknown tool {name}"})
        calls_rec.append({"name": name, "arguments": args})
        results.append(result)
    turn: Turn = {"assistant": state["last_text"], "reasoning": state["last_reasoning"],
                  "tool_calls": calls_rec, "tool_results": results}
    return {"turns": state["turns"] + [turn], "last_tool_results": results}


def reflect(state: AgentState) -> dict:
    """Feed numeric results back as a user turn so the model revises its answer."""
    feedback = "\n".join(f"[{c['name']}] {r}" for c, r in
                         zip(state["turns"][-1]["tool_calls"], state["turns"][-1]["tool_results"]))
    think = f"\n<think>\n{state['last_reasoning']}\n</think>" if state["last_reasoning"] else ""
    msgs = state["messages"] + [
        {"role": "assistant", "content": state["last_text"] + think},
        {"role": "user", "content": "Tool results:\n" + feedback +
         "\nRevise the answer using these numbers, or if it is correct, STOP calling tools and return the final answer in XML tags."},
    ]
    return {"messages": msgs, "rounds": state["rounds"] + 1}


def finalize(state: AgentState) -> dict:
    """Verifier verdict + full SFT trajectory (reasoning, tool calls, numeric results, revisions)."""
    from .validate import validate
    last_assistant = state["last_text"] or (state["turns"][-1]["assistant"] if state["turns"] else "")
    final = last_assistant
    turns = state["turns"]
    # If the adapter returned the full trajectory (tools already executed once), don't append a
    # duplicate final turn — its last turn is the model's final answer.
    if not turns and final:
        turns = [{"assistant": final, "reasoning": state["last_reasoning"],
                  "tool_calls": [], "tool_results": []}]
    parts = []
    for t in turns:
        if t.get("reasoning"):
            parts.append(f"<think>\n{t['reasoning']}\n</think>")
        if t.get("assistant"):
            parts.append(t["assistant"])
        for call, result in zip(t.get("tool_calls", []), t.get("tool_results", [])):
            args = call.get("arguments", "")
            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            parts.append(f"<tool_call name=\"{call.get('name','tool')}\">{args}</tool_call>")
            parts.append(f"<tool_result>{result}</tool_result>")
    sft = "\n".join(parts)
    verdict = validate(state["task"], final, state["row"])
    return {"final": final, "turns": turns, "verdict": verdict,
            "accepted": verdict["accepted"], "sft_completion": sft, "done": True}


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("build_context", build_context)
    g.add_node("propose", propose)
    g.add_node("call_tools", call_tools)
    g.add_node("reflect", reflect)
    g.add_node("finalize", finalize)
    g.add_edge(START, "build_context")
    g.add_edge("build_context", "propose")
    g.add_conditional_edges("propose", decide, {"call_tools": "call_tools", "finalize": "finalize"})
    g.add_edge("call_tools", "reflect")
    g.add_edge("reflect", "propose")
    g.add_edge("finalize", END)
    return g.compile()


GRAPH = build_graph()


def run(row: dict, task: str, respond, notes=None, max_rounds=4, k_memory=3,
        mode="agents+skills") -> AgentState:
    """Convenience entry point. `respond(messages, tools) -> (text, tool_calls, reasoning)`."""
    init: AgentState = {
        "row": row, "task": task, "respond": respond, "notes": notes or [],
        "max_rounds": max_rounds, "k_memory": k_memory, "mode": mode,
    }
    return GRAPH.invoke(init)
