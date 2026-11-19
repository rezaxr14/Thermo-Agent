"""One shared parser/emitter for the tool-call blocks that appear in this project.

Two wire formats occur in the wild and BOTH are accepted by :func:`parse_tool_calls`:

* our SFT / harness / eval format (XML with a JSON body)::

      <tool_call name="inspect_plant">{"candidate": ""}</tool_call>

* Qwen3.5's native chat-template format (nested ``<function>`` / ``<parameter>``)::

      <tool_call>
      <function=inspect_plant>
      <parameter=candidate>
      ...
      </parameter>
      </function>
      </tool_call>

The module is deliberately format-only: it contains no model ids, timestamps, costs
or any other bookkeeping, and it does not depend on a particular tokenizer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

XML_JSON = "xml-json"
QWEN_NATIVE = "qwen-native"


@dataclass
class ToolCall:
    """A parsed tool call.

    ``format`` is either ``"xml-json"`` (our inline format) or ``"qwen-native"``.
    ``raw`` is the exact matched block, useful for logging/debugging without losing
    the original text.
    """

    name: str
    arguments: dict
    raw: str
    format: str


# Our format: a name attribute (either quote style) and a JSON body. Whitespace is
# tolerated around the ``=``, inside the open tag and around the body.
# The tempered body `(?:(?!<tool_call\\b).)*?` prevents a well-formed opening tag from
# swallowing a later nested/unclosed call and producing a false positive.
_XML_CALL_RE = re.compile(
    r"<tool_call\s+name\s*=\s*(?P<q>[\"'])(?P<name>.*?)(?P=q)\s*>"
    r"(?P<body>(?:(?!<tool_call\b).)*?)</tool_call>",
    re.S,
)

# Qwen3.5 native: <tool_call><function=NAME><parameter=K>v</parameter>...</function></tool_call>
_NATIVE_CALL_RE = re.compile(
    r"<tool_call\s*>\s*<function=(?P<name>[^>]+)>"
    r"(?P<body>(?:(?!<tool_call\b).)*?)</function>\s*</tool_call>",
    re.S,
)
_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter>",
    re.S,
)

# Blocks removed by strip_tool_calls (kept permissive: both quote styles / attr-free).
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call\b.*?</tool_call>", re.S)
_TOOL_RESULT_BLOCK_RE = re.compile(r"<tool_result\b[^>]*>.*?(?:</tool_result>|$)", re.S)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think>|$)", re.S)


def _json_or_str(value: str):
    """Coerce a trimmed string to its JSON type when possible, else keep the string."""
    v = value.strip()
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def _from_xml(match: re.Match) -> ToolCall:
    body = match.group("body").strip()
    if not body:
        arguments: dict = {}
    else:
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            # Malformed JSON body: keep behaviour compatible with the old inline parser.
            payload = {"candidate": body}
        if not isinstance(payload, dict):
            payload = {"candidate": str(payload)}
        arguments = payload
    return ToolCall(match.group("name").strip(), arguments, match.group(0), XML_JSON)


def _from_native(match: re.Match) -> ToolCall:
    arguments = {
        p.group("key").strip(): _json_or_str(p.group("value"))
        for p in _PARAM_RE.finditer(match.group("body"))
    }
    return ToolCall(match.group("name").strip(), arguments, match.group(0), QWEN_NATIVE)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Return every well-formed tool call in ``text``, in source order.

    Structural errors (missing name, missing closing tag, etc.) are ignored, never
    raised. A JSON body that fails to decode is kept as ``{"candidate": <body>}``
    for our format; native ``<parameter>`` values that are not JSON stay strings.
    """
    if not text:
        return []
    found: list[tuple[int, ToolCall]] = []
    for m in _XML_CALL_RE.finditer(text):
        found.append((m.start(), _from_xml(m)))
    for m in _NATIVE_CALL_RE.finditer(text):
        found.append((m.start(), _from_native(m)))
    found.sort(key=lambda pair: pair[0])
    return [call for _, call in found]


def is_tool_call(text: str) -> bool:
    """True when ``text`` contains at least one well-formed tool call."""
    return bool(parse_tool_calls(text))


def strip_tool_calls(text: str) -> str:
    """Remove tool calls, tool results and reasoning, leaving only the final answer."""
    if not text:
        return ""
    s = _TOOL_CALL_BLOCK_RE.sub("", text)
    s = _TOOL_RESULT_BLOCK_RE.sub("", s)
    s = _THINK_BLOCK_RE.sub("", s)
    return s.strip()


def format_tool_result(name: str, result_json: str) -> str:
    """Wrap a tool result in our ``<tool_result>...</tool_result>`` block.

    ``name`` is accepted for symmetry with tool-call emitters; our wire format does
    not carry a name attribute on the result block, so it is intentionally unused.
    """
    return f"<tool_result>{result_json}</tool_result>"
