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
