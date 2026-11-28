"""One native Qwen3.5 structured tool protocol, tested with the real tokenizer.

The legacy ``<tool_call name=...>`` form must never be accepted: a model trained or
evaluated on the native template would otherwise appear to have called a tool when it
emitted the retired syntax.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from thermo_agent.agent import tool_protocol
from thermo_agent.synth import combine


def _tokenizer():
    path = combine.tokenizer_path()
    if not path:
        return None
    directory = pathlib.Path(path).parent
    if not (directory / "tokenizer_config.json").is_file():
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)


@pytest.fixture(scope="module")
def tok():
    tokenizer = _tokenizer()
    if tokenizer is None:
        pytest.skip("cached Qwen3.5 tokenizer is not available")
    return tokenizer


def _schemas():
    return tool_protocol.tool_schemas("repair")


def test_tool_schemas_expose_all_sixteen_tools():
    names = tool_protocol.tool_names(_schemas())
    assert len(names) == 16
    for required in ("inspect_plant", "first_law_check", "second_law_check",
                     "validity_check", "solve_plant", "fluid_state", "turbomachine",
                     "list_skills", "load_skill"):
        assert required in names


def test_structured_render_and_parse_round_trip(tok):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        tool_protocol.assistant_message(
            "<think>check the plant</think>",
            [{"name": "inspect_plant", "arguments": {"candidate": ""}}]),
        tool_protocol.tool_message("inspect_plant", '{"ok": true}'),
        tool_protocol.assistant_message("done"),
    ]
    tools = _schemas()
    prompt = tok.apply_chat_template(messages[:2], tools=tools, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=True)
    full = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=False,
                                   tokenize=False, enable_thinking=True)
    assert full.startswith(prompt)
    completion = full[len(prompt):]
    assert "<tool_call>" in completion and "<function=inspect_plant>" in completion
    assert "<tool_response>" in completion

    calls = tool_protocol.parse_generated(tok, completion, prompt, tools)
    assert calls, "native structured tool call was not parsed"
    name, arguments = calls[0]
    assert name == "inspect_plant"
    assert arguments == {"candidate": ""}


def test_legacy_call_syntax_is_not_parsed(tok):
    legacy = '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>'
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "u"}], tools=_schemas(),
        add_generation_prompt=True, tokenize=False, enable_thinking=True)
    assert tool_protocol.parse_generated(tok, legacy, prompt, _schemas()) == []


def test_teacher_schema_requires_reason_and_ingest_moves_it_into_think():
    teacher = tool_protocol.tool_schemas("repair", reason=True)
    for schema in teacher:
        function = schema["function"]
        assert "reason" in function["parameters"]["properties"]
        assert "reason" in function["parameters"]["required"]
    student = _schemas()
    assert all("reason" not in s["function"]["parameters"]["properties"] for s in student)

    def fake(candidate):
        return {"candidate": candidate}
    wrapped = tool_protocol.teacher_tool_wrappers([fake])
    assert wrapped[0].__name__ == "fake"
    assert wrapped[0](reason="because", candidate="x") == {"candidate": "x"}

    messages = [tool_protocol.assistant_message(
        "", [{"name": "inspect_plant", "arguments": {"reason": "Check the as-given solver status.",
                                                      "candidate": ""}}])]
    reasons = tool_protocol.strip_reason_arguments(messages)
    assert reasons == ["Check the as-given solver status."]
    call = messages[0]["tool_calls"][0]["function"]
    assert "reason" not in call["arguments"]
    assert call["arguments"] == {"candidate": ""}
    assert "<think>" in messages[0]["content"]
    assert "Check the as-given solver status." in messages[0]["content"]


def test_three_step_grpo_style_trace_parses_a_tool_call_every_step(tok):
    """Cheap 3-step smoke: every GRPO-exposed tool renders and parses natively.

    This does not launch GRPO or a model; it checks the exact parser the rollout loop
    uses against three of the 16 tools the wrapper exposes.
    """
    from thermo_agent.train.grpo import TOOL_NAMES

    assert len(TOOL_NAMES) == 16
    source = pathlib.Path(__file__).resolve().parents[1] / "data/processed/repair_train.jsonl"
    if not source.is_file():
        pytest.skip("no real processed repair row")
    row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    tools = _schemas()
    prefix = tok.apply_chat_template(row["prompt"], tools=tools, add_generation_prompt=True,
                                     tokenize=False, enable_thinking=True)
    for name in TOOL_NAMES[:3]:
        message = tool_protocol.assistant_message(
            "", [{"name": name, "arguments": {"candidate": ""}}])
        full = tok.apply_chat_template(row["prompt"] + [message], tools=tools,
                                       add_generation_prompt=False, tokenize=False,
                                       enable_thinking=True)
        calls = tool_protocol.parse_generated(tok, full[len(prefix):], prefix, tools)
        assert calls and calls[0][0] == name


def test_messages_from_turns_does_not_double_wrap_think():
    turns = [{"reasoning": "<think>already wrapped</think>", "assistant": "",
              "tool_calls": [{"name": "inspect_plant", "arguments": {"candidate": ""}}],
              "tool_results": ['{"ok": true}']}]
    messages = tool_protocol.messages_from_turns(turns)
    assert messages[0]["content"].count("<think>") == 1
    assert messages[1] == {"role": "tool", "name": "inspect_plant", "content": '{"ok": true}'}


def test_tool_result_output_drops_wrapper_call_id_and_reason():
    raw = (
        "ToolInvocationResult(name='inspect_plant', "
        "arguments={'candidate': '', 'reason': 'Check the as-given plant.'}, "
        "call_id='call_secret_123', output='{\"ok\": true, \"solved\": true}', "
        "error=None)")
    assert tool_protocol.tool_result_output(raw) == '{"ok": true, "solved": true}'
    assert tool_protocol.tool_result_output('{"ok": false}') == '{"ok": false}'
    with pytest.raises(ValueError):
        tool_protocol.tool_result_output("ToolInvocationResult(name='x', output=not_a_literal)")


def test_normalize_tool_turns_keeps_the_served_output():
    """Regression: compact_result returned None for unchanged payloads, so cleaning
    a journal turn replaced every real tool result with None (or an empty string)."""
    raw = (
        "ToolInvocationResult(name='inspect_plant', "
        "arguments={'reason': 'Check the as-given plant.'}, "
        "call_id='call_secret_123', output='{\"ok\": true, \"solved\": true}', "
        "error=None)")
    turns = [{
        "assistant": "I inspect the plant first because the solver status is unknown.",
        "reasoning": "",
        "tool_calls": [{"name": "inspect_plant", "arguments": {"reason": "Check it."}}],
        "tool_results": [raw],
    }]
    tool_protocol.normalize_tool_turns(turns)
    assert turns[0]["tool_results"] == ['{"ok": true, "solved": true}']


def test_normalize_trajectory_strips_reasons_and_keeps_final_visible():
    final = "<diagnosis>Only CH.eps failed.</diagnosis>\n<edits>\nSET CH.eps = 1\n</edits>"
    trajectory = {
        "final": final,
        "turns": [
            {"assistant": "", "reasoning": "",
             "tool_calls": [{"name": "inspect_plant",
                             "arguments": {"candidate": "", "reason": "Inspect first."}}],
             "tool_results": [
                 "ToolInvocationResult(name='inspect_plant', "
                 "arguments={'candidate': '', 'reason': 'Inspect first.'}, "
                 "call_id='call_secret_123', output='{\"ok\": true}', error=None)"]},
            {"assistant": final, "reasoning": final, "tool_calls": [], "tool_results": []},
        ],
    }
    clean = tool_protocol.normalize_trajectory(trajectory)
    text = clean["sft_completion"]
    assert "<think>\nInspect first.\n</think>" in text
    assert "<parameter=reason>" not in text
    assert "call_id" not in text and "ToolInvocationResult" not in text
    assert "<tool_response>" in text
    assert text.count(final) == 1
    assert clean["messages"][-1]["content"] == final
    assert "reason" not in clean["messages"][0]["tool_calls"][0]["function"]["arguments"]


def test_normalize_train_row_rejects_legacy_text():
    legacy = {"id": "x", "completion": [
        {"role": "assistant", "content": '<tool_call name="inspect_plant">{}</tool_call>'}]}
    assert tool_protocol.normalize_train_row(legacy) is None
    native = {"id": "y", "completion": [
        {"role": "assistant", "content": "<tool_call><function=inspect_plant></function></tool_call>"}]}
    assert tool_protocol.normalize_train_row(native) is native
