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

The native tool loop (e.g. kbench's `native_tool_agent` inside the adapter) is the DRIVER: it
executes the physics tools exactly once and may hand back a complete 4-tuple trajectory
`(text, calls, reasoning, turns)`. The graph is the STATE MACHINE / VERIFIER: it builds the
context, routes on the adapter output and records the verifier verdict + SFT trajectory. When a
4-tuple arrives, the graph jumps straight to `finalize` — it is NOT a second tool executor, so
tools are never run twice.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, TypedDict

from ..agent.tool_prompt import build_harness_tool_messages
from ..core.schema import Plant
from . import memory as mem
from .toolkit import make_tools, tool_names

# Single source of truth for the round limit. Imported by both drivers and used in the
# system-prompt sentence, so the number the model reads always equals the number enforced.
DEFAULT_MAX_ROUNDS = 6   # D4


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
    """System prompt = task statement + artefact-free compact harness + short memory recaps.

    The harness is appended by the ONE shared builder used by tracegen, the ``*_h``
    training corpora and eval/GRPO, so a given (task, plant, config) always produces
    the same harness text (J2/J3 harness-hash invariant). Only the row's own system
    prompt (the task statement) and the retrieved success/failure recaps are added
    here; row-level bookkeeping (id, run_id, manifests, journal metadata) is never
    read, and the verbose full AGENTS.md + component-skill block is never served.
    """
    row, task = state["row"], state["task"]
    original = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] == "system")
    plant = None
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                plant = Plant.model_validate_json(row[key]); break
            except Exception:
                pass
    # Build the tools first: the same list feeds both the tool contract (so the compact
    # prompt names exactly the tools the model may call) and the executor's tool_map.
    # By default this is the five-plant-tool contract; the hosted frontier harness sets
    # include_dynamic=True to offer the nine continuous-input component tools as well.
    include_dynamic = bool(state.get("include_dynamic", False))
    tools = make_tools(task, row, include_dynamic=include_dynamic)
    # The shared compact builder carries compact AGENTS.md + either the ONE
    # selected skill (default) or the on-demand skill catalog (dynamic mode) +
    # the tool contract. max_turns is the round limit the graph enforces, so the
    # number the model reads always equals the number actually used.
    sys = build_harness_tool_messages(
        [{"role": "system", "content": original}], task, plant,
        mode=state.get("mode", "agents+skills"),
        max_turns=state.get("max_rounds", DEFAULT_MAX_ROUNDS), tools=tools,
        include_dynamic=include_dynamic)[0]["content"]
    # Record WHICH skill was served. The skill bytes ride inside the system prompt, so
    # without this the trace cannot prove a skill was read; the name is deterministic
    # from TASK_SKILLS[task] and the hash covers the exact text sent.
    from ..agent.harness import TASK_SKILLS
    skill_names = list(TASK_SKILLS.get(task) or [])
    served_skill = skill_names[0] if skill_names else ""
    notes = state.get("notes") or []
    # D10/D11: guidance first (inventory skills/tools, then examine the cycle against memory),
    # then a TOKEN-EFFICIENT memory that carries only what failed and what works — no model ids,
    # timestamps, cost or run logs. The full ledger stays on disk for our own analysis.
    from .compact_memory import agent_guidance_block, compact_notes
    sys += "\n\n" + agent_guidance_block()
    # Retrieve failure memory EXACTLY ONCE per episode and exclude notes mined from this same
    # record. The one result feeds both the served (compact) block and `notes_used`, so the
    # journal cannot claim notes the model was never shown (or show a note twice).
    retrieved: list = []
    memory_block = ""
    if notes:
        retrieved = mem.retrieve(
            " ".join(m["content"] for m in row["prompt"]), notes,
            k=state.get("k_memory", 3), task=task,
            exclude_record=str(row.get("id", "")),
        )
        compact = compact_notes(retrieved, k=state.get("k_memory", 3))
        if compact:
            memory_block = compact
            sys += "\n\n" + compact
    # Short reminder only: the tool contract above already lists the five tools, their
    # numbers and the final-answer tags. Keep the enforced round count in the prompt
    # (D4) without re-sending the whole contract or the tool-name list.
    sys += (
        "\n\nYou have REAL physics tools; use their numbers to solve and self-correct "
        "(up to %d rounds). Then STOP and emit the final answer in the task's XML tags."
    ) % state.get("max_rounds", DEFAULT_MAX_ROUNDS)
    notes_used = [n.id for n in retrieved]
    user = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] != "system")
    return {
        "system_prompt": sys, "tools": tools, "tool_map": {t.__name__: t for t in tools},
        "include_dynamic": include_dynamic,
        "notes_used": notes_used,
        "served_skill": served_skill,
        "memory_block": memory_block,
        "system_prompt_sha256": hashlib.sha256(sys.encode("utf-8")).hexdigest(),
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
    if state.get("last_calls") and state.get("rounds", 0) < state.get("max_rounds", DEFAULT_MAX_ROUNDS):
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
    """Feed numeric results back so the model revises while keeping its own reasoning.

    The next turn must see the assistant text/reasoning that produced these tool
    calls. That is the whole point of a multi-turn agentic episode: the model
    carries its reasoning, the skill it chose and its momentum into the revision.
    Emit the reasoning once: if it is already the assistant text, do not repeat it.
    """
    feedback = "\n".join(f"[{c['name']}] {r}" for c, r in
                         zip(state["turns"][-1]["tool_calls"], state["turns"][-1]["tool_results"]))
    assistant_text = state["last_text"] or ""
    reason = (state["last_reasoning"] or "").strip()
    if reason and reason not in assistant_text:
        assistant_text = f"<think>\n{reason}\n</think>\n{assistant_text}".strip()
    msgs = state["messages"] + [
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": "Tool results:\n" + feedback +
         "\nRevise the answer using these numbers, or if it is correct, STOP calling tools and return the final answer in XML tags."},
    ]
    return {"messages": msgs, "rounds": state["rounds"] + 1}


def finalize(state: AgentState) -> dict:
    """Verifier verdict + full SFT trajectory (reasoning, tool calls, numeric results, revisions).

    Hosted prompt shards carry the conversation plus tool inputs but deliberately omit
    answer-bearing scoring columns. For those rows the local verifier cannot produce a
    trustworthy verdict (it used to raise ``KeyError: 'ed_json'``), so the verdict is
    DEFERRED to ``transfer.ingest``, which joins the full export by id. The trajectory
    is still recorded in full. Local corpus rows keep the exact real verifier verdict.
    """
    from .validate import deferred_verdict, missing_scoring_fields, validate
    last_assistant = state["last_text"] or (state["turns"][-1]["assistant"] if state["turns"] else "")
    final = last_assistant or ""
    turns = state["turns"]
    # If no trajectory was recorded (e.g. the adapter never started a turn), fall back to a
    # single assistant turn holding the final answer so the SFT text is never empty.
    if not turns:
        turns = [{"assistant": final, "reasoning": state["last_reasoning"],
                  "tool_calls": [], "tool_results": []}]
    parts = []
    for t in turns:
        reason = str(t.get("reasoning") or "").strip()
        assistant = str(t.get("assistant") or "").strip()
        # The visible per-turn prose is the reasoning for chat-completions models.
        # Emit it once, not as both <think> and assistant text.
        if reason and reason != assistant:
            parts.append(f"<think>\n{reason}\n</think>")
        if assistant:
            parts.append(assistant)
        for call, result in zip(t.get("tool_calls", []), t.get("tool_results", [])):
            args = call.get("arguments", "")
            args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            parts.append(f"<tool_call name=\"{call.get('name','tool')}\">{args}</tool_call>")
            parts.append(f"<tool_result>{result}</tool_result>")
    sft = "\n".join(parts)
    # Guarantee: a non-empty final answer yields non-empty SFT text that ends with that answer.
    # Don't duplicate it when the last turn already ends with it.
    if final:
        if not sft.strip():
            sft = final
        elif not sft.rstrip().endswith(final.rstrip()):
            sft = sft.rstrip() + "\n" + final
    row = state["row"]
    missing = missing_scoring_fields(state["task"], row)
    # An input-only hosted row is recognisable by its carried prompt; a bare/partial
    # dict is an ordinary verifier input and keeps the previous local behaviour.
    if missing and isinstance(row.get("prompt"), list):
        verdict = deferred_verdict(missing)
    else:
        verdict = validate(state["task"], final, row)
    return {"final": final, "turns": turns, "verdict": verdict,
            "accepted": verdict["accepted"], "sft_completion": sft, "done": True}


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
def build_graph():
    # LAZY import: the hosted native-tool-loop path (run_direct) has no langgraph in its
    # venv. Importing this module must therefore never require langgraph; only the local
    # graph driver (run / build_graph) does.
    from langgraph.graph import StateGraph, START, END

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


# The compiled graph is built on first use, NOT at import time. Building it needs langgraph,
# which the hosted Benchmarks venv does not have; its native-tool-loop path uses run_direct.
GRAPH = None


def _get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    return GRAPH


def run(row: dict, task: str, respond, notes=None, max_rounds=DEFAULT_MAX_ROUNDS, k_memory=3,
        mode="agents+skills", include_dynamic=False) -> AgentState:
    """Convenience entry point. `respond(messages, tools) -> (text, tool_calls, reasoning)`.

    Requires langgraph (local driver only). The hosted path uses `run_direct` instead.
    """
    init: AgentState = {
        "row": row, "task": task, "respond": respond, "notes": notes or [],
        "max_rounds": max_rounds, "k_memory": k_memory, "mode": mode,
        "include_dynamic": include_dynamic,
    }
    return _get_graph().invoke(init)


def run_direct(row: dict, task: str, respond, *, notes=None, k_memory=3,
               mode="agents+skills", max_rounds=DEFAULT_MAX_ROUNDS,
               include_dynamic=False) -> dict:
    """Drive the native tool loop WITHOUT langgraph; reproduce the graph's 4-tuple path exactly.

    The adapter (`kbench_adapter.make_respond`) is the DRIVER: it runs the whole multi-turn
    tool loop inside one chat and returns the complete trajectory as
    `(text, calls, reasoning, turns)`. Here we do what the graph does on that path:
      build_context -> respond -> finalize
    The returned `calls` are deliberately discarded (the adapter already executed them, once),
    so tools are never run a second time. `respond(messages, tools) -> (text, calls, reasoning, turns)`.

    Returns the same fields the harness reads from `run()`: `final`, `turns`, `verdict`,
    `accepted`, `sft_completion`, `notes_used`, `tools_available`, `done`.
    """
    init: AgentState = {
        "row": row, "task": task, "respond": respond, "notes": notes or [],
        "max_rounds": max_rounds, "k_memory": k_memory, "mode": mode,
        "include_dynamic": include_dynamic,
    }
    ctx = build_context(init)
    text, calls, reasoning, turns = respond(ctx["messages"], ctx["tools"])
    # Mirror propose()'s 4-tuple branch: last_calls=[] so tools are NOT re-executed.
    del calls
    state: AgentState = {
        **init,
        **ctx,
        "last_text": text or "",
        "last_calls": [],
        "last_reasoning": reasoning or "",
        "turns": turns,
        "rounds": 0,
    }
    out = finalize(state)
    return {
        "final": out["final"],
        "turns": out["turns"],
        "verdict": out["verdict"],
        "accepted": out["accepted"],
        "sft_completion": out["sft_completion"],
        "notes_used": ctx["notes_used"],
        "served_skill": ctx.get("served_skill", ""),
        "memory_block": ctx.get("memory_block", ""),
        "system_prompt_sha256": ctx.get("system_prompt_sha256", ""),
        "tools_available": tool_names(include_dynamic=include_dynamic),
        "done": out["done"],
    }
