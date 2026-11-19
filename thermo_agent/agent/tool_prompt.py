"""Shared tool-using system prompt for the H eval arm and the SFT/GRPO corpora.

Training rows must see the *same* system prompt the model sees at eval time in
``ours_harness``: AGENTS.md + the selected skills plus the multi-turn tool
contract. Keeping the text in one module avoids eval/training prompt drift.
"""
from __future__ import annotations

import inspect as _inspect

from . import harness as H

# Final-answer XML contracts mirror agent/AGENTS.md ("Answer formats") and
# train/prompts.SYSTEM. Kept here as the single source for the tool contract.
FINAL_TAGS = {
    "repair": ("<diagnosis>", "<edits>"),
    "design": ("<rationale>", "<design>"),
    "audit": ("<analysis>", "<ranking>", "<change>", "<prediction>"),
    "ledger": ("<diagnosis>", "<fix>"),
}


def tool_contract(tools, task, max_turns, max_tool_calls) -> str:
    """The five tools' schemas/contracts as text (Google-style docstrings from the toolkit).

    This is copied verbatim from ``ToolLoop.tool_contract`` in
    ``thermo_agent/eval/model_eval.py``; that method is now a thin alias to it.
    """
    tags = FINAL_TAGS.get(task, ("the task's XML tags",))
    lines = [
        "MULTI-TURN TOOL ENVIRONMENT (physics only; there is NO answer-oracle tool).",
        "To call a tool, emit EXACTLY this strict block (quoted name, JSON body, closing tag):",
        '  <tool_call name="TOOL_NAME">{"candidate": "..."}</tool_call>',
        'Use candidate="" for inspect_plant; every other tool takes your candidate answer XML.',
        f"Limits: at most {max_turns} assistant turns and {max_tool_calls} tool calls.",
        "Each result appears verbatim inside <tool_result>...</tool_result>; read the numbers and revise.",
        "When done, STOP calling tools and emit ONLY the final answer XML tags: " + " + ".join(tags),
        "Available tools (name: contract):",
    ]
    for fn in tools:
        doc = " ".join((_inspect.getdoc(fn) or "").split())
        lines.append(f"- {fn.__name__}: {doc}")
    return "\n".join(lines)


def _default_tools():
    """The five physics tools in canonical order, for text-only prompt building."""
    from .env import ThermoEnv
    return [ThermoEnv.inspect_plant, ThermoEnv.first_law_check, ThermoEnv.second_law_check,
            ThermoEnv.validity_check, ThermoEnv.solve_plant]


def build_harness_tool_messages(messages, task, plant, mode="agents+skills",
                                max_turns=4, max_tool_calls=6, tools=None) -> list[dict]:
    """A deep copy of ``messages`` with the harness and the tool contract appended.

    Mirrors ``ToolLoop._base_messages`` exactly: H.augment adds AGENTS.md/skills to
    the system message, then the tool contract is appended to that same system
    message (or inserted as one when there is no system message). The input list
    and its message dicts are never mutated.
    """
    augmented = [dict(m) for m in H.augment(list(messages), mode, task, plant)]
    contract = tool_contract(tools if tools is not None else _default_tools(),
                             task, max_turns, max_tool_calls)
    if augmented and augmented[0].get("role") == "system":
        augmented[0]["content"] = augmented[0]["content"] + "\n\n" + contract
    else:
        augmented.insert(0, {"role": "system", "content": contract})
    return augmented
