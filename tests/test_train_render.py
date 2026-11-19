"""Training-path audit 2026-11-19 item 2: one compact harness, no metadata in prompts."""
from __future__ import annotations

import pathlib

import pytest

from thermo_agent.agent import tool_prompt
from thermo_agent.agent.env import ThermoEnv
from thermo_agent.train import render


def _compact_row(task="repair"):
    tools = [ThermoEnv.inspect_plant, ThermoEnv.first_law_check, ThermoEnv.second_law_check,
             ThermoEnv.validity_check, ThermoEnv.solve_plant]
    msgs = tool_prompt.build_harness_tool_messages(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "plant"}],
        task, None, tools=tools)
    return {"id": "row-1", "task": task, "tier": "easy", "tokens": 1234,
            "prompt_variant": "harness_tools", "harness": "agents+skills",
            "prompt": msgs, "completion": [{"role": "assistant", "content": "<edits/>"}]}


def test_model_messages_drops_row_bookkeeping_and_keeps_role_content_only():
    row = _compact_row()
    row["prompt"][1]["note"] = "internal bookkeeping"
    row["prompt"][1]["tier"] = "easy"
    msgs = render.model_messages(row)
    assert all(set(m) == {"role", "content"} for m in msgs)
    assert "internal bookkeeping" not in render.prompt_text(row)
    assert "easy" not in render.prompt_text(row)


def test_model_messages_rejects_malformed_messages():
    with pytest.raises(ValueError, match="role/content"):
        render.model_messages({"prompt": [{"content": "no role"}]})
    with pytest.raises(ValueError, match="non-string"):
        render.model_messages({"prompt": [{"role": "user", "content": 7}]})


def test_compact_builder_output_passes_the_guard_for_every_task():
    from thermo_agent.synth.tiers import TASKS
    for task in TASKS:
        render.assert_compact_harness(_compact_row(task))  # no raise


def test_stale_verbose_harness_fails_loudly():
    row = _compact_row()
    row["prompt"][0]["content"] = "base\n\n# AGENTS.md: energy-plant tasks\n\n## Components\n..."
    with pytest.raises(ValueError, match="compact=True"):
        render.assert_compact_harness(row)


def test_plain_prompt_is_not_harness_checked():
    row = {"prompt": [{"role": "user", "content": "plain task"}], "harness": "none"}
    render.assert_compact_harness(row)


def test_both_trainers_fail_loudly_on_a_stale_harness_before_training():
    root = pathlib.Path(__file__).resolve().parents[1] / "thermo_agent" / "train"
    for name in ("sft.py", "grpo.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "assert_compact_harness(row)" in src, f"{name} silently accepts a stale harness prompt"


def test_render_pair_uses_only_model_messages(monkeypatch):
    """A fake tokenizer sees the sanitised messages, never the row's metadata."""

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            seen.append([dict(m) for m in messages])
            return "RENDERED"

    seen = []
    row = _compact_row()
    row["prompt"][0]["leak"] = "SHOULD NOT APPEAR"
    assert render.render_pair(Tok(), row) == {"prompt": "RENDERED", "completion": "<edits/>"}
    assert all(set(m) == {"role", "content"} for m in seen[0])
