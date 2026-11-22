"""Gate 8: multi-turn GRPO wiring, verified against the installed TRL 1.18.1 source.

CPU only: no model load, no GPU, no TRL training. The environment wrapper, reward contract,
batch guard and config values are checked exactly as TRL probes/calls them.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from _rows import make_row  # noqa: E402
import trl.trainer.grpo_trainer as trl_grpo  # noqa: E402
from trl import GRPOConfig  # noqa: E402
from transformers.utils import get_json_schema  # noqa: E402

from thermo_agent.agent.env import ThermoEnv  # noqa: E402
from thermo_agent.train import ddp, grpo, render, rewards  # noqa: E402
from thermo_agent.train.config import GrpoSettings  # noqa: E402


def _public_functions(cls):
    """The exact predicate/filter TRL uses to discover tools (grpo_trainer.py:750, :2529)."""
    return {name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith("_")}


# --------------------------------------------------------------------------- #
# 1. The TRL probe behaviour the wrapper is written against
# --------------------------------------------------------------------------- #
def test_trl_source_probes_factory_with_no_args_and_reads_public_class_methods():
    src = pathlib.Path(trl_grpo.__file__).read_text(encoding="utf-8")
    assert "EnvironmentFactory = Callable[[], _SupportsReset]" in src
    assert "def reset(self, **kwargs) -> str | None: ..." in src
    assert "instance = factory()" in src
    assert "inspect.getmembers(type(instance), predicate=inspect.isfunction)" in src
    assert "member_name not in (\"reset\", \"get_reward\")" in src
    assert "environment.reset(**reset_kwargs)" in src
    assert "completions/clipped_ratio" in src          # TRL already logs truncation rate


def test_wrapper_exposes_exactly_sixteen_tools_and_no_env_internals():
    factory = grpo.make_env_factory("repair")
    wrapper = factory()                                 # TRL calls this with no arguments
    public = _public_functions(type(wrapper))
    assert public - {"reset", "get_reward"} == set(grpo.TOOL_NAMES)
    assert "step" not in public and "observation_block" not in public
    # We keep the verifier rewards as reward_funcs, so the wrapper must NOT own get_reward.
    assert "get_reward" not in public
    assert len(grpo.TOOL_NAMES) == 16


def test_reset_takes_dataset_columns_and_returns_observation_not_prompt():
    row = make_row()
    wrapper = grpo.GrpoThermoEnv()
    observation = wrapper.reset(**row)                  # TRL forwards every dataset column
    assert isinstance(observation, str)
    assert observation.startswith("MULTI-TURN TOOL ENVIRONMENT")
    assert "inspect_plant" in observation and "<edits>" in observation
    assert wrapper._env is not None                     # the owned ThermoEnv was bound to the row


# --------------------------------------------------------------------------- #
# 2. Tool schemas: type hints + Google docstrings sufficient for JSON schema
# --------------------------------------------------------------------------- #
def test_all_sixteen_tools_generate_a_json_schema_with_candidate():
    schemas = [get_json_schema(tool) for tool in grpo.wrapper_tool_methods()]
    assert len(schemas) == 16
    for schema in schemas:
        fn = schema["function"]
        assert fn["name"] in grpo.TOOL_NAMES
        assert fn["description"]
        candidate = fn["parameters"]["properties"]["candidate"]
        assert candidate["type"] == "string"
        assert candidate["description"]


def test_raw_thermoenv_docstrings_are_not_schema_ready_in_transformers_5_23():
    """Why the wrapper needs its own docstrings: `candidate (str):` is not parseable here."""
    env = ThermoEnv(make_row(), "repair")
    with pytest.raises(Exception, match="docstring has no description"):
        get_json_schema(env.inspect_plant)


# --------------------------------------------------------------------------- #
# 3. k-sweep batch divisibility
# --------------------------------------------------------------------------- #
def test_k_sweep_batch_divisibility_and_exact_error_message():
    grad_accum = GrpoSettings().grad_accum
    for k in (4, 8, 16):
        # One sequence per step on one rank cannot hold a group of k.
        assert not ddp.grpo_batches_divisible(1, 1, 1, k)
        with pytest.raises(ValueError) as exc:
            ddp.assert_grpo_batches(1, 1, 1, k)
        assert str(exc.value) == ddp.grpo_batch_error(1, 1, 1, k)
        assert "per_device_train_batch_size=1" in str(exc.value)
        assert "divisible" in str(exc.value) and f"k={k}" in str(exc.value)
        # The decided fix: per-device 2, accum = k on two ranks (4 prompts per optimizer step).
        ddp.assert_grpo_batches(2, k, 2, k)


def test_recipe_is_fp16_lora_ddp_no_qlora_fsdp_zeRo():
    ddp.assert_fp16_lora_ddp("float16", {}, object())
    with pytest.raises(ValueError, match="float16"):
        ddp.assert_fp16_lora_ddp("bfloat16", {}, object())
    with pytest.raises(ValueError, match="QLoRA"):
        ddp.assert_fp16_lora_ddp("float16", {"quantization_config": object()}, object())
    with pytest.raises(ValueError, match="ZeRO"):
        ddp.assert_fp16_lora_ddp("float16", {}, object(), deepspeed={"config": "ds.json"})
    with pytest.raises(ValueError, match="FSDP"):
        ddp.assert_fp16_lora_ddp("float16", {}, object(), fsdp="full_shard")
    with pytest.raises(ValueError, match="LoRA"):
        ddp.assert_fp16_lora_ddp("float16", {}, None)


# --------------------------------------------------------------------------- #
# 4. Reward contract: last message is the final answer, truncated rollouts rejected
# --------------------------------------------------------------------------- #
_FINAL = "<diagnosis>residuals ~0</diagnosis>\n<edits>SET T0.P_it = 20e6</edits>"
_TOOL_TURN = {"role": "assistant", "content": "",
              "tool_calls": [{"type": "function", "function": {"name": "inspect_plant", "arguments": {}}}]}
_TOOL_RESULT = {"role": "tool", "name": "inspect_plant", "content": '{"ok": false}'}


def test_completion_text_final_answer_for_full_one_and_multi_turn_rollouts():
    assert rewards.completion_text([{"role": "assistant", "content": _FINAL}]) == _FINAL
    full = [_TOOL_TURN, _TOOL_RESULT, {"role": "assistant", "content": _FINAL}]
    assert rewards.completion_text(full) == _FINAL
    assert rewards.truncation_reason(full) is None
    assert rewards.completion_text(_FINAL) == _FINAL             # legacy plain-text path


def test_completion_text_rejects_truncated_rollouts_with_a_clear_reason():
    truncated = [_TOOL_TURN, _TOOL_RESULT]                       # ran out of length after the tool
    assert rewards.completion_text(truncated) == ""              # never scored as a final answer
    with pytest.raises(rewards.TruncatedRollout, match="tool result"):
        rewards.completion_text(truncated, strict=True)
    assert rewards.r_format(None, [truncated]) == [0.0]
    # An unanswered native/legacy tool call is also not a final answer.
    native = ("<tool_call>\n<function=solve_plant>\n<parameter=candidate>\nx\n"
              "</parameter>\n</function>\n</tool_call>")
    assert rewards._is_well_formed_tool_call is not None  # shared agent/toolcall.py parser in use
    assert rewards.is_tool_call_content(native)
    assert rewards.is_tool_call_content('<tool_call name="first_law_check">{}</tool_call>')
    # Incomplete native markup must be rejected too, even though it is not "well formed".
    assert rewards.is_tool_call_content("<tool_call>\n<function=solve_plant>\n<parameter=candidate>")
    with pytest.raises(rewards.TruncatedRollout, match="tool call"):
        rewards.completion_text([{"role": "assistant", "content": native}], strict=True)


# --------------------------------------------------------------------------- #
# 5. Config values (single thinking constant, multi-turn defaults, configurable)
# --------------------------------------------------------------------------- #
def test_multiturn_config_values_are_set_on_grpo_config():
    values = grpo.multiturn_config()
    assert values["max_tool_calling_iterations"] == grpo.DEFAULT_MAX_TOOL_CALLING_ITERATIONS == 6
    assert values["max_completion_length"] >= 2048 and values["max_completion_length"] != 320
    assert values["mask_truncated_completions"] is True

    cfg = GRPOConfig(output_dir="out", num_generations=4, **values)
    assert cfg.max_tool_calling_iterations == values["max_tool_calling_iterations"]
    assert cfg.max_completion_length == values["max_completion_length"]
    assert cfg.mask_truncated_completions is True

    custom = grpo.multiturn_config(max_tool_calling_iterations=8, max_completion_length=4096)
    assert custom["max_tool_calling_iterations"] == 8
    assert custom["max_completion_length"] == 4096


def test_thinking_is_one_shared_value_and_reasoning_is_in_targets():
    # Measured on the cached Qwen3.5 tokenizer: False pre-closes <think> and breaks prompt-prefix
    # alignment for reasoning-bearing teacher rows; True keeps the alignment exact.
    assert render.TRAIN_ON_REASONING is True
    assert grpo.TRAIN_ON_REASONING is render.TRAIN_ON_REASONING  # one shared value, not a copy
    assert render.CHAT_KW == {"enable_thinking": True}           # SFT uses this
    assert grpo.multiturn_config()["chat_template_kwargs"] == {"enable_thinking": True}


# --------------------------------------------------------------------------- #
# 6. Diagnostics: zero-variance share, warning above 50%, no behaviour change
# --------------------------------------------------------------------------- #
def _func_a(prompts=None, completions=None, **kwargs):
    return [1.0, 1.0]


def _func_b(prompts=None, completions=None, **kwargs):
    return [0.0, 0.0]


def test_reward_diagnostics_logs_zero_variance_share_and_warns():
    diag = grpo.RewardDiagnostics([1.0, 1.0], num_generations=2)
    funcs = diag.wrap([_func_a, _func_b])
    assert [f.__name__ for f in funcs] == ["_func_a", "_func_b"]  # reward_weights map by name
    logged = []
    with pytest.warns(UserWarning, match="zero reward variance"):
        for fn in funcs:
            fn(log_metric=lambda name, value: logged.append((name, value)))
    assert logged == [("reward_diagnostics/zero_variance_share", 1.0)]
    assert diag.latest == {"zero_variance_share": 1.0, "groups": 1}


def test_reward_diagnostics_quiet_below_threshold():
    diag = grpo.RewardDiagnostics([1.0, 1.0], num_generations=2)
    funcs = diag.wrap([_func_a, lambda *a, **k: [1.0, 0.0]])
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for fn in funcs:
            fn(log_metric=lambda name, value: None)
    assert not caught
    assert diag.latest["zero_variance_share"] == 0.0
