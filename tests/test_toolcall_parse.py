"""The one shared tool-call parser must accept our XML format and Qwen3.5's native format."""
from __future__ import annotations

from thermo_agent.agent.toolcall import (
    XML_JSON,
    QWEN_NATIVE,
    ToolCall,
    format_tool_result,
    is_tool_call,
    parse_tool_calls,
    strip_tool_calls,
)


# ---------------------------------------------------------------------------
# our format: <tool_call name="X">{json}</tool_call>
# ---------------------------------------------------------------------------
def test_xml_single_call():
    text = '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    c = calls[0]
    assert isinstance(c, ToolCall)
    assert c.name == "inspect_plant"
    assert c.arguments == {"candidate": ""}
    assert c.format == XML_JSON
    assert c.raw == text


def test_xml_multiple_calls_keep_source_order():
    text = (
        '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>\n'
        'and <tool_call name="first_law_check">{"candidate": "<diagnosis>x</diagnosis>"}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["inspect_plant", "first_law_check"]
    assert calls[1].arguments["candidate"] == "<diagnosis>x</diagnosis>"
    assert all(c.format == XML_JSON for c in calls)


def test_xml_empty_body_is_empty_arguments():
    calls = parse_tool_calls('<tool_call name="inspect_plant"></tool_call>')
    assert len(calls) == 1
    assert calls[0].arguments == {}


def test_xml_single_quoted_name_and_whitespace():
    calls = parse_tool_calls("  <tool_call name = 'validity_check' >  {\"candidate\": \"c\"}  </tool_call>  ")
    assert len(calls) == 1
    assert calls[0].name == "validity_check"
    assert calls[0].arguments == {"candidate": "c"}


def test_xml_malformed_or_unclosed_is_ignored():
    # no closing tag -> ignored; no name attribute -> ignored; stray prose -> ignored
    text = (
        '<tool_call name="inspect_plant">{"candidate": ""}'
        ' <tool_call>{"candidate": "no name"}</tool_call>'
        ' just some prose'
    )
    assert parse_tool_calls(text) == []


def test_xml_non_json_body_falls_back_to_candidate():
    (call,) = parse_tool_calls('<tool_call name="validity_check">not json</tool_call>')
    assert call.arguments == {"candidate": "not json"}


def test_xml_non_object_json_is_wrapped():
    (call,) = parse_tool_calls('<tool_call name="validity_check">42</tool_call>')
    assert call.arguments == {"candidate": "42"}


# ---------------------------------------------------------------------------
# Qwen3.5 native: <tool_call><function=NAME><parameter=K>v</parameter>...</function></tool_call>
# ---------------------------------------------------------------------------
def test_native_one_parameter():
    text = (
        "<tool_call>\n<function=inspect_plant>\n<parameter=candidate>\n\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    (call,) = parse_tool_calls(text)
    assert call.name == "inspect_plant"
    assert call.arguments == {"candidate": ""}
    assert call.format == QWEN_NATIVE
    assert call.raw == text


def test_native_two_parameters_with_json_typed_values():
    text = (
        "<tool_call>\n<function=first_law_check>\n"
        "<parameter=candidate>\n<diagnosis>bad</diagnosis>\n</parameter>\n"
        "<parameter=retries>\n3\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    (call,) = parse_tool_calls(text)
    assert call.name == "first_law_check"
    assert call.arguments == {"candidate": "<diagnosis>bad</diagnosis>", "retries": 3}
    assert call.format == QWEN_NATIVE


def test_native_json_object_value_is_coerced():
    text = (
        "<tool_call><function=solve_plant>"
        "<parameter=payload>{\"candidate\": \"x\", \"n\": 2}</parameter>"
        "</function></tool_call>"
    )
    (call,) = parse_tool_calls(text)
    assert call.arguments == {"payload": {"candidate": "x", "n": 2}}


def test_native_no_parameters_and_whitespace():
    text = "  <tool_call>\n\n <function=solve_plant>\n </function> \n</tool_call>  "
    (call,) = parse_tool_calls(text)
    assert call.name == "solve_plant"
    assert call.arguments == {}


def test_native_and_xml_mixed_in_one_text_are_both_parsed_in_order():
    text = (
        '<tool_call name="inspect_plant">{}</tool_call>\n'
        "<tool_call><function=validity_check><parameter=candidate>x</parameter></function></tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [(c.name, c.format) for c in calls] == [
        ("inspect_plant", XML_JSON),
        ("validity_check", QWEN_NATIVE),
    ]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_parse_returns_empty_for_prose():
    assert parse_tool_calls("I am not sure; there is no tool here.") == []
    assert parse_tool_calls("") == []
    assert parse_tool_calls(None) == []


def test_is_tool_call():
    assert is_tool_call('<tool_call name="inspect_plant">{}</tool_call>')
    assert is_tool_call("<tool_call><function=inspect_plant></function></tool_call>")
    assert not is_tool_call("plain prose")


def test_strip_tool_calls_keeps_only_the_final_answer():
    text = (
        "<think>\nreason about it\n</think>\n"
        '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>\n'
        '<tool_result>{"ok": true}</tool_result>\n'
        "<diagnosis>The fluid is wrong.</diagnosis>\n"
        "<edits>\nFLUID CH.hot_in = water\n</edits>"
    )
    answer = strip_tool_calls(text)
    assert answer == "<diagnosis>The fluid is wrong.</diagnosis>\n<edits>\nFLUID CH.hot_in = water\n</edits>"
    assert "<think>" not in answer and "<tool_call" not in answer and "<tool_result>" not in answer


def test_format_tool_result_round_trips():
    payload = '{"ok": true, "failed_checks": []}'
    block = format_tool_result("inspect_plant", payload)
    assert block == f"<tool_result>{payload}</tool_result>"
    # stripping a formatted result leaves no trace, and prose is untouched
    assert strip_tool_calls(f"before{block}after") == "beforeafter"
    assert strip_tool_calls("only prose") == "only prose"
