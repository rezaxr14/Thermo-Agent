"""Render training pairs exactly as GRPO and inference will see the prompt.

Measured with the cached Qwen3.5-4B tokenizer: with ``enable_thinking=True``
``apply_chat_template(sys+user, add_generation_prompt=True)`` is a LITERAL PREFIX of
``apply_chat_template(sys+user+assistant_containing_<think>)``. With ``False`` the template
injects an empty closed ``<think>\n\n</think>\n\n`` into the generation prompt, which is NOT
a prefix. Our stored trajectories keep ``<think>...</think>`` inside the assistant content,
so ``completion_only_loss`` masking aligns only when reasoning is enabled. TRAIN_ON_REASONING
is exported so SFT and GRPO share one value.

Prompt hygiene (2026-11-19 training-path audit): the chat template sees only the
``role``/``content`` of each message. Corpus bookkeeping (``id``, ``tier``, ``task``,
``tokens``, ``provenance`` ...) stays in the row columns, which GRPO's environment
and reward dispatch need, and is never handed to the tokenizer. Harness-bearing
rows must have been built by the single J2/J3 builder
``thermo_agent.agent.tool_prompt.build_harness_tool_messages(compact=True)``; a
stale verbose ``*_h`` shard fails loudly instead of training a different prompt
than eval uses.
"""
from __future__ import annotations

TRAIN_ON_REASONING = True
CHAT_KW = {"enable_thinking": TRAIN_ON_REASONING}

# Literal markers of the one compact harness and of the retired verbose builder.
COMPACT_HARNESS_MARKER = "# AGENTS.md: energy-plant tasks (compact)"
VERBOSE_HARNESS_MARKERS = ("## Components", "# AGENTS.md: energy-plant tasks\n")
TOOL_CONTRACT_MARKER = "MULTI-TURN TOOL ENVIRONMENT"


def model_messages(row: dict) -> list[dict]:
    """Only chat fields reach the model; a malformed message is a hard error.

    Rows carry bookkeeping columns (task, tier, id, tokens, provenance ...) that TRL
    forwards to the environment and the reward dispatcher. Dropping anything else
    from each message here is what keeps that metadata out of the model prompt.
    ``tool_calls``/``name`` are part of the native protocol and are preserved.
    """
    messages = row["prompt"]
    if isinstance(messages, str):  # a few legacy plain-prompt rows predate the chat list
        return [{"role": "user", "content": messages}]
    out = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise ValueError(f"prompt message {i} has no role/content chat fields: {message!r:.200}")
        role = message["role"]
        content = message.get("content", "")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"prompt message {i} has non-string role/content")
        clean = {"role": role, "content": content}
        for key in ("tool_calls", "name", "tool_call_id"):
            if message.get(key):
                clean[key] = message[key]
        out.append(clean)
    if not out:
        raise ValueError("empty model prompt")
    return out


def completion_messages(row: dict) -> list[dict]:
    """The stored completion as chat messages (one assistant message for plain rows)."""
    completion = row.get("completion")
    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]
    if not isinstance(completion, list) or not completion:
        raise ValueError(f"row {row.get('id', '<no id>')!r} has no completion")
    return [dict(message, role=message.get("role") or "assistant") for message in completion]


def row_tools(row: dict) -> list[dict]:
    """Tool schemas for a row, or ``[]`` for a plain (no-tool) row.

    A row is a tool row when it stores native ``tool_calls``, carries a trajectory, or
    was built on the harness. The schemas come from the one shared builder, so training
    and eval inject exactly the same 16 tool definitions.
    """
    structured = any(isinstance(m, dict) and m.get("tool_calls")
                     for m in (row.get("completion") or [])
                     if isinstance(m, dict))
    harness = bool(row.get("harness") and row["harness"] != "none")
    toolish = structured or harness or bool(row.get("trajectory")) or \
        row.get("prompt_variant") == "harness_tools"
    task = row.get("task")
    if not toolish or not task:
        return []
    from ..agent.tool_protocol import tool_schemas
    return tool_schemas(task)


def prompt_text(row: dict) -> str:
    """The exact text the model is trained on (all messages, no row metadata)."""
    return "\n".join(m["content"] for m in model_messages(row))


def assert_compact_harness(row: dict) -> None:
    """Fail if a tool row carries the retired verbose harness instead of the compact one.

    A row is a harness row when its metadata says so or its system prompt carries
    the tool contract. Plain/A-slice rows (``harness == 'none'``) are not checked.
    """
    text = prompt_text(row)
    is_harness = (row.get("prompt_variant") == "harness_tools"
                  or row.get("harness", "none") not in ("none", "", None)
                  or TOOL_CONTRACT_MARKER in text)
    if not is_harness:
        return
    if COMPACT_HARNESS_MARKER not in text or any(m in text for m in VERBOSE_HARNESS_MARKERS):
        raise ValueError(
            f"row {row.get('id', '<no id>')!r} claims a tool harness but its prompt was not "
            "built by tool_prompt.build_harness_tool_messages(compact=True). Rebuild the *_h "
            "corpus on the J2/J3 compact harness before training; a verbose harness changes "
            "the eval prompt and pushes an 8,192-token row over the T4 memory budget."
        )


def generation_prompt(tok, messages: list[dict], tools: list[dict] | None = None) -> str:
    return tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                   tokenize=False, **CHAT_KW)


def render_pair(tok, row: dict) -> dict:
    """Render prompt and completion through the native template, with tools injected."""
    messages = model_messages(row)
    completion = completion_messages(row)
    tools = row_tools(row)
    prompt = generation_prompt(tok, messages, tools)
    full = tok.apply_chat_template(messages + completion, tools=tools or None,
                                   tokenize=False, **CHAT_KW)
    if not full.startswith(prompt):
        raise ValueError(
            f"row {row.get('id', '<no id>')!r}: the rendered completion is not a suffix of the "
            "generation prompt; the tool template and the training renderer disagree")
    return {"prompt": prompt, "completion": full[len(prompt):]}


def prefix_report(tok, row: dict) -> dict:
    """Does the whole-conversation render start with the generation prompt? False means conversational SFT would mismatch GRPO."""
    messages = model_messages(row)
    completion = completion_messages(row)
    tools = row_tools(row)
    full = tok.apply_chat_template(messages + completion, tools=tools or None, tokenize=False, **CHAT_KW)
    gp = generation_prompt(tok, messages, tools)
    return {"prefix_matches": full.startswith(gp), "generation_prompt_tail": gp[-80:], "full_render_at_that_point": full[len(gp) - 80: len(gp) + 40] if len(full) >= len(gp) else full[-120:]}
