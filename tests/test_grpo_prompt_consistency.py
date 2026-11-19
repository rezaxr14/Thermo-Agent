"""GRPO must not append a second tool contract to rows that already carry one.

CPU only: no GPU, no TRL trainer, no untracked corpora. The environment wrapper is
driven exactly as TRL drives it (``reset(**dataset_columns)``), using small dicts.
"""
from __future__ import annotations

import inspect
import logging

from thermo_agent.agent.env import row_has_harness_tools
from thermo_agent.train import grpo

_MARKER = "MULTI-TURN TOOL ENVIRONMENT"


def _plain_row() -> dict:
    """Today's GRPO row: a plain 740-char-style system prompt, no tool contract."""
    return {
        "prompt": [
            {"role": "system", "content": "You are a plant engineer."},
            {"role": "user", "content": "Repair the plant."},
        ],
    }


def _harness_row() -> dict:
    """A harness_tools row: the tool contract is already in the system message."""
    return {
        "prompt": [
            {"role": "system", "content": "You are a plant engineer.\n\n"
                                          + _MARKER + " (physics only; there is NO answer-oracle tool)."},
            {"role": "user", "content": "Repair the plant."},
        ],
        "prompt_variant": "harness_tools",
        "harness": "agents+skills",
    }


def _full_prompt_text(row: dict, observation: str) -> str:
    """What TRL builds: the row's messages plus observation appended to the last one."""
    contents = [m["content"] for m in row["prompt"]
                if isinstance(m, dict) and m.get("content")]
    return "\n".join(contents + [observation])


# --------------------------------------------------------------------------- #
# 1. detection accepts either the row metadata or the system-prompt marker
# --------------------------------------------------------------------------- #
def test_row_has_harness_tools_detection():
    assert row_has_harness_tools(_plain_row()) is False
    assert row_has_harness_tools(_harness_row()) is True

    # metadata alone (system prompt carries no marker)
    assert row_has_harness_tools({
        "prompt_variant": "harness_tools",
        "prompt": [{"role": "system", "content": "You are a plant engineer."}],
    }) is True

    # marker alone, with NO prompt_variant column — what a column-filtering loader emits
    assert row_has_harness_tools({
        "prompt": [{"role": "system", "content": "Skills...\n" + _MARKER + "\nMore."}],
    }) is True

    # a marker in a user message is not a system prompt -> plain behaviour
    assert row_has_harness_tools({
        "prompt": [{"role": "user", "content": _MARKER}],
    }) is False

    # string prompts are treated as the whole prompt text
    assert row_has_harness_tools({"prompt": "plain instruction"}) is False
    assert row_has_harness_tools({"prompt": "plain\n" + _MARKER}) is True


# --------------------------------------------------------------------------- #
# 2. the core consistency contract
# --------------------------------------------------------------------------- #
def test_harness_row_does_not_gain_a_second_tool_contract():
    plain, harness = _plain_row(), _harness_row()
    assert _MARKER not in _full_prompt_text(plain, "")            # no contract before reset
    assert _full_prompt_text(harness, "").count(_MARKER) == 1     # one contract before reset

    plain_observation = grpo.GrpoThermoEnv().reset(**plain)
    harness_observation = grpo.GrpoThermoEnv().reset(**harness)

    # plain still gains exactly one block; harness keeps exactly one (no duplicate)
    assert _full_prompt_text(plain, plain_observation).count(_MARKER) == 1
    assert _full_prompt_text(harness, harness_observation).count(_MARKER) == 1


def test_plain_row_still_gets_the_full_observation_block():
    observation = grpo.GrpoThermoEnv().reset(**_plain_row())
    assert observation.startswith(_MARKER)
    for name in grpo.TOOL_NAMES:
        assert name in observation
    assert "<edits>" in observation


def test_reset_returns_the_same_structure_in_both_cases():
    plain_observation = grpo.GrpoThermoEnv().reset(**_plain_row())
    harness_observation = grpo.GrpoThermoEnv().reset(**_harness_row())
    assert type(plain_observation) is str
    assert type(harness_observation) is str
    assert harness_observation == ""                             # nothing for TRL to append


# --------------------------------------------------------------------------- #
# 3. the flag and its one-time log
# --------------------------------------------------------------------------- #
def test_flag_is_set_only_for_harness_rows_and_logged_once(caplog):
    plain = grpo.GrpoThermoEnv()
    plain.reset(**_plain_row())
    assert plain.prompt_already_has_tools is False

    harness = grpo.GrpoThermoEnv()
    with caplog.at_level(logging.INFO, logger="thermo_agent.train.grpo"):
        harness.reset(**_harness_row())
        harness.reset(**_harness_row())                          # same env, second episode
    assert harness.prompt_already_has_tools is True

    records = [r for r in caplog.records
               if r.name == "thermo_agent.train.grpo" and "harness tool prompt" in r.message]
    assert len(records) == 1


# --------------------------------------------------------------------------- #
# 4. limits, tools and reward/final-answer path are untouched
# --------------------------------------------------------------------------- #
def test_max_turns_and_tool_calls_unchanged_for_both_rows():
    for row in (_plain_row(), _harness_row()):
        wrapper = grpo.GrpoThermoEnv()
        wrapper.reset(**row)
        assert wrapper.max_turns == grpo.DEFAULT_MAX_TOOL_CALLING_ITERATIONS == 6
        assert wrapper.max_tool_calls == grpo.MAX_TOOL_CALLS == 8
        assert wrapper._env.max_turns == wrapper.max_turns
        assert wrapper._env.max_tool_calls == wrapper.max_tool_calls
        assert "6 assistant turns" in wrapper._env.observation_block()
        assert "8 tool calls" in wrapper._env.observation_block()


def test_custom_limits_honoured_for_both_rows():
    for row in (_plain_row(), _harness_row()):
        wrapper = grpo.GrpoThermoEnv(max_turns=7, max_tool_calls=9)
        wrapper.reset(**row)
        assert wrapper._env.max_turns == 7
        assert wrapper._env.max_tool_calls == 9


def test_tool_exposure_and_final_answer_path_unchanged():
    for row in (_plain_row(), _harness_row()):
        wrapper = grpo.GrpoThermoEnv()
        wrapper.reset(**row)

        # TRL discovers the same five public tools, never step/observation_block.
        public = {name for name, _ in inspect.getmembers(type(wrapper), predicate=inspect.isfunction)
                  if not name.startswith("_")}
        assert public - {"reset", "get_reward"} == set(grpo.TOOL_NAMES)
        assert [fn.__name__ for fn in grpo.wrapper_tool_methods(wrapper)] == list(grpo.TOOL_NAMES)

        # step still exists and finalizes only through the verifier (never a tool).
        assert wrapper._env.step is not None
        assert "final" not in [list(h)[0] for h in wrapper._env.history]
