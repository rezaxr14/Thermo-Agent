"""Static guard for evaluation-arm separation in ``kaggle/eval_runner.py``.

Pure-python: the Kaggle notebook source is parsed with ``ast`` (never imported or
executed, so no Kaggle bootstrap / GPU / network is involved).  The only package
import is ``thermo_agent.eval.model_eval`` for one signature check; torch and
transformers are imported lazily inside HFGenerator and are never touched here.
"""
from __future__ import annotations

import ast
import inspect
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "kaggle" / "eval_runner.py"

# The four RL arms: each records the task its checkpoint was trained on (``train_task``).
EXPECTED_GRPO_ARMS = {"ours_grpo", "ours_sft_grpo_k4", "ours_sft_grpo_k8", "ours_sft_grpo_k16"}
_UNSET = object()


def _module():
    return ast.parse(EVAL_RUNNER.read_text(encoding="utf-8"), filename=str(EVAL_RUNNER))


def _literal(node):
    """Literal value of an AST node, or _UNSET for names/calls (env vars, paths, ...)."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNSET


def _specs():
    """Read the module-level SPECS dict from the source without importing kaggle."""
    for node in _module().body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SPECS" for t in node.targets):
            assert isinstance(node.value, ast.Dict), "SPECS must be a dict literal"
            specs = {}
            for key, val in zip(node.value.keys, node.value.values):
                name = _literal(key)
                if not isinstance(name, str):
                    continue
                assert isinstance(val, ast.Dict), f"SPECS[{name}] must be a dict literal"
                specs[name] = {_literal(fk): _literal(fv)
                               for fk, fv in zip(val.keys, val.values)}
            return specs
    raise AssertionError("SPECS assignment not found in kaggle/eval_runner.py")


def test_exactly_four_grpo_arms_use_tools():
    """GRPO checkpoints were trained inside the tool env, so all 4 must be scored with tools."""
    specs = _specs()
    # Derive the GRPO set from ``train_task`` (unique to the RL arms) rather than from
    # tool_access, so the assertion would catch a missing/extra GRPO spec.
    grpo = {name for name, spec in specs.items() if "train_task" in spec}
    assert grpo == EXPECTED_GRPO_ARMS, f"unexpected GRPO arms: {sorted(grpo)}"
    assert len(grpo) == 4
    for name in sorted(grpo):
        spec = specs[name]
        assert spec["backend"] == "hf", f"{name} must be an HF arm"
        assert spec["tool_access"] == "tools", f"{name} must be scored with tools"
        assert spec["harness"] == "agents+skills", f"{name} must use the tool-env prompt"
        assert spec["zero_shot"] is False, f"{name} has weight updates (not zero-shot)"


def test_ours_sft_is_prompt_only_harness_contrast():
    """ours_sft carries the AGENTS.md/skills harness text but no tool loop (the Z side)."""
    spec = _specs()["ours_sft"]
    assert spec["tool_access"] == "none"
    assert spec["harness"] == "agents+skills"


def test_ours_base_is_no_harness_no_tools_zero_shot():
    """ours_base is the raw base model: no harness, no tools, no weight updates."""
    spec = _specs()["ours_base"]
    assert spec["harness"] == "none"
    assert spec["tool_access"] == "none"
    assert spec["zero_shot"] is True


def test_hf_arms_never_disable_thinking():
    """No arm may pass --no-thinking (or --thinking): HFGenerator must keep its default True."""
    strings = {n.value for n in ast.walk(_module())
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "--no-thinking" not in strings, "eval_runner must not render prompts with thinking off"
    assert "--thinking" not in strings, "eval_runner must not override the thinking mode"

    specs = _specs()
    hf_arms = {name: spec for name, spec in specs.items() if spec.get("backend") == "hf"}
    assert hf_arms, "expected at least one HF arm"
    for name, spec in hf_arms.items():
        assert not any("think" in str(k).lower() for k in spec), \
            f"{name} special-cases thinking in its spec"


def test_hf_generator_defaults_to_enable_thinking_true():
    """The eval generator renders the same chat-template prefix the corpora were trained with."""
    from thermo_agent.eval.model_eval import HFGenerator

    params = inspect.signature(HFGenerator.__init__).parameters
    assert "enable_thinking" in params
    assert params["enable_thinking"].default is True
