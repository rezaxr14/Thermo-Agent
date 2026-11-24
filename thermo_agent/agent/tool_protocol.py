"""One tool protocol everywhere: Qwen3.5 native structured tool calls.

Training rows, deterministic traces, frontier ingest, GRPO and the eval loop all use
OpenAI-style messages:

    {"role": "assistant", "content": "<think>...</think>", 
     "tool_calls": [{"type": "function",
                     "function": {"name": "inspect_plant", "arguments": {"candidate": ""}}}]}
    {"role": "tool", "name": "inspect_plant", "content": "<verbatim JSON result>"}

Rendering goes through the real tokenizer's ``apply_chat_template(tools=...)``. The
Qwen3.5 template owns the wire format (``<tool_call><function=...>``); no prompt text
duplicates the call syntax or the tool list. Tool results are rendered as
``<tool_response>`` and masked by SFT.

``tokenizer.parse_response`` parses the generated text back into a message. The cached
Qwen3.5 tokenizer has no ``response_template`` of its own, so this module supplies one
that matches the template's native format. The legacy ``<tool_call name=...>`` text is
deliberately NOT recognised.
"""
from __future__ import annotations

import inspect
import json
import re

# Response template for transformers' ``tokenizer.parse_response``. It matches the
# Qwen3.5 chat template exactly: one ``<tool_call>`` per call, an inner
# ``<function=NAME>`` and ``<parameter=KEY>value</parameter>`` arguments.
RESPONSE_TEMPLATE = {
    "version": 1,
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"<\|im_start\|>assistant\n",
    "fields": {
        "content": {"content": "text"},
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<tool_name>[^>]+)>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter>",
                "value_parser": {"name": "text", "args": {"strip": True}},
                "merge_duplicates": True,
            },
            "transform": {
                "type": "function",
                "function": {"name": "{tool_name}", "arguments": "{content}"},
            },
        },
    },
}


def _description(fn) -> str:
    text = " ".join((inspect.getdoc(fn) or "").split())
    return text[:700]


def tool_schemas(task: str, row: dict | None = None, *, include_dynamic: bool = True,
                 reason: bool = False) -> list[dict]:
    """OpenAI-style schemas for the student (or teacher) toolset.

    ``reason=False`` is the student/eval contract: one ``candidate`` string argument.
    ``reason=True`` is the hosted TEACHER contract: a required ``reason`` sentence and
    the candidate. Arguments are plain strings; the toolkit unwraps JSON objects from
    ``candidate`` for the component tools.
    """
    from ..teacher.toolkit import make_tools

    if row is None:
        # Schemas only need the callables' names/docstrings, not a real plant row.
        from ..agent.env import ThermoEnv
        fns = [ThermoEnv.inspect_plant, ThermoEnv.first_law_check, ThermoEnv.second_law_check,
               ThermoEnv.validity_check, ThermoEnv.solve_plant]
        if include_dynamic:
            from ..teacher import toolkit
            fns = fns + toolkit.dynamic_tools() + toolkit.skill_tools()
    else:
        fns = make_tools(task, row, include_dynamic=include_dynamic)

    out = []
    for fn in fns:
        name = fn.__name__
        properties = {}
        required = []
        if reason:
            properties["reason"] = {
                "type": "string",
                "description": "One short sentence: what you are checking and why, before this call.",
            }
            required.append("reason")
        properties["candidate"] = {
            "type": "string",
            "description": "Empty for inspect_plant; otherwise the candidate answer XML or a JSON object string.",
        }
        if name != "inspect_plant":
            required.append("candidate")
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": _description(fn),
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        })
    return out


def teacher_tool_wrappers(fns) -> list:
    """Wrap hosted tool callables with a required ``reason`` argument.

    Used only on the TEACHER/hosted path. The wrapper records the model's reason and
    executes the original tool with its ``candidate`` argument; ingest later moves the
    reason into ``<think>`` and strips it from the student call arguments.
    """
    out = []
    for fn in fns:
        def tool(reason: str, candidate: str = "", _fn=fn):
            return _fn(candidate)
        tool.__name__ = fn.__name__
        tool.__qualname__ = f"teacher_tool.{fn.__name__}"
        tool.__doc__ = (
            f"{_description(fn)}\n\n"
            "Args:\n"
            "    reason: One short sentence on what you are checking and why.\n"
            "    candidate: Empty for inspect_plant; otherwise the candidate XML or a JSON object string.\n\n"
            "Returns:\n"
            "    str: JSON with the tool's real numeric result.")
        tool.__annotations__ = {"reason": str, "candidate": str, "return": str}
        out.append(tool)
    return out


def tool_names(schemas: list[dict]) -> list[str]:
    return [s["function"]["name"] for s in schemas]


def assistant_message(content: str = "", calls: list[dict] | None = None) -> dict:
    """One assistant message; ``calls`` is a list of ``{name, arguments}`` dicts."""
    message: dict = {"role": "assistant", "content": content or ""}
    if calls:
        message["tool_calls"] = [
            {"type": "function",
             "function": {"name": c["name"], "arguments": dict(c.get("arguments") or {})}}
            for c in calls
        ]
    return message


def tool_message(name: str, content: str) -> dict:
    return {"role": "tool", "name": name or "tool", "content": content}


def messages_from_turns(turns: list[dict], final: str = "") -> list[dict]:
    """Convert the harness ``turns`` shape into structured chat messages.

    Each turn carries ``assistant`` prose, ``reasoning`` and parallel
    ``tool_calls``/``tool_results`` lists. The reasoning becomes the ``<think>`` block
    of the assistant message; tool results become ``role="tool"`` messages.
    """
    messages: list[dict] = []
    for turn in turns or []:
        reason = str(turn.get("reasoning") or "").strip()
        prose = str(turn.get("assistant") or "").strip()
        content = prose
        if reason:
            if reason.lstrip().startswith("<think"):
                content = reason if not prose else f"{reason}\n{prose}"
            elif reason != prose:
                content = f"<think>\n{reason}\n</think>\n{prose}".strip()
            else:
                content = f"<think>\n{reason}\n</think>"
        calls = []
        for call in turn.get("tool_calls") or []:
            name = str(call.get("name") or "")
            if not name:
                continue
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    args = {"candidate": args}
            calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        if content or calls:
            messages.append(assistant_message(content, calls))
        for call, result in zip(calls, turn.get("tool_results") or []):
            messages.append(tool_message(call["name"], str(result)))
    if final:
        messages.append(assistant_message(final))
    return messages


_LEGACY_TURN = re.compile(
    r"<tool_call\s+name\s*=\s*[\"'](?P<name>[^\"']+)[\"']\s*>(?P<body>.*?)</tool_call>"
    r"|(?P<result_open><tool_result>)(?P<result>.*?)</tool_result>",
    re.S,
)


def messages_from_legacy_text(text: str) -> list[dict]:
    """Migrate a retired ``<tool_call name=...>``/``<tool_result>`` trace to the native protocol.

    This is a one-way conversion of stored traces, not runtime compatibility: the legacy
    syntax is never accepted by the model-facing parser and is not written to new corpora.
    """
    if not text or "<tool_call name=" not in text:
        return []
    messages: list[dict] = []
    buffer: list[str] = []
    last_name = ""
    cursor = 0
    for match in _LEGACY_TURN.finditer(text):
        buffer.append(text[cursor:match.start()])
        if match.group("name"):
            content = "".join(buffer).strip()
            name = match.group("name")
            try:
                args = json.loads(match.group("body"))
            except (TypeError, ValueError):
                args = {"candidate": match.group("body")}
            messages.append(assistant_message(content, [{"name": name, "arguments": args}]))
            buffer.clear()
            last_name = name
        else:
            content = "".join(buffer).strip()
            if content or not messages:
                messages.append(assistant_message(content))
            messages.append(tool_message(last_name or "tool", match.group("result")))
            buffer.clear()
        cursor = match.end()
    tail = "".join(buffer).strip()
    if tail:
        messages.append(assistant_message(tail))
    return messages


def final_from_messages(messages: list[dict]) -> str:
    """The last assistant message that is not a tool call."""
    for message in reversed(messages or []):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message.get("content") or "")
    return ""


def strip_reason_arguments(messages: list[dict]) -> list[str]:
    """Move each tool call's teacher ``reason`` into the turn's ``<think>`` block.

    Returns the collected reasons in call order. The stored call arguments never keep
    a ``reason`` key, so the student is never trained to emit the teacher-only schema.
    """
    reasons: list[str] = []
    for message in messages or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            reason = str(args.pop("reason", "") or "").strip() if isinstance(args, dict) else ""
            if not reason:
                continue
            reasons.append(reason)
            content = str(message.get("content") or "")
            if reason not in content:
                message["content"] = f"<think>\n{reason}\n</think>\n{content}".strip()
    return reasons


def strip_reason_turns(turns: list[dict]) -> int:
    """Move teacher ``reason`` arguments into each harness turn's ``reasoning`` field.

    The hosted journal stores ``trajectory.turns`` (no ``messages``). Every teacher
    tool call carries a mandatory ``reason``; that sentence is the per-turn
    chain-of-thought. Return the number of turns that gained reasoning.
    """
    moved = 0
    for turn in turns or []:
        calls = turn.get("tool_calls") or []
        reasons: list[str] = []
        for call in calls:
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = None
            if isinstance(args, dict):
                reason = str(args.pop("reason", "") or "").strip()
                if reason:
                    reasons.append(reason)
        if reasons and not str(turn.get("reasoning") or "").strip():
            turn["reasoning"] = " ".join(reasons)
            moved += 1
    return moved



def native_text(messages: list[dict]) -> str:
    """Render structured messages in the Qwen3.5 native textual wire format.

    Used only for metrics and for callers that have no chat template at hand. Training
    always renders the structured messages through ``apply_chat_template(tools=...)``.
    The legacy ``<tool_call name=...>`` form is never produced.
    """
    parts: list[str] = []
    for message in messages or []:
        role = message.get("role")
        if role == "assistant":
            content = str(message.get("content") or "")
            if content:
                parts.append(content)
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                args = fn.get("arguments") or {}
                body = ["<tool_call>", f"<function={name}>"]
                for key, value in args.items():
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    body += [f"<parameter={key}>", text, "</parameter>"]
                body += ["</function>", "</tool_call>"]
                parts.append("\n".join(body))
        elif role == "tool":
            parts.append(f"<tool_response>\n{message.get('content', '')}\n</tool_response>")
    return "\n".join(part for part in parts if part)


def render_messages(tok, messages: list[dict], tools: list[dict], *, add_generation_prompt: bool = False,
                    enable_thinking: bool = True) -> str:
    return tok.apply_chat_template(
        messages, tools=tools or None, add_generation_prompt=add_generation_prompt,
        tokenize=False, enable_thinking=enable_thinking)


def parse_generated(tok, text: str, prefix: str, tools: list[dict]) -> list[tuple[str, dict]]:
    """Parse generated text with ``tokenizer.parse_response`` (native protocol only).

    A tokenizer without ``parse_response`` falls back to the shared native parser. The
    legacy ``<tool_call name=...>`` syntax is never accepted: ``parse_response`` only
    sees the template's ``<function=...>`` blocks.
    """
    parser = getattr(tok, "parse_response", None)
    if callable(parser):
        try:
            message = parser(text, schema=RESPONSE_TEMPLATE, prefix=prefix, tools=tools)
        except (AttributeError, ValueError, TypeError):
            message = None
        if isinstance(message, dict):
            calls = []
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                if fn.get("name"):
                    calls.append((fn["name"], fn.get("arguments") or {}))
            return calls
    from .toolcall import parse_tool_calls
    return [(c.name, c.arguments) for c in parse_tool_calls(text)]
