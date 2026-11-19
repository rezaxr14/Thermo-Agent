"""Tests for the generic Kaggle notebook's pure helpers (no Kaggle/network calls)."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "notebooks/kaggle_model_runner.py"
    spec = importlib.util.spec_from_file_location("kaggle_model_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_config_is_budget_only_and_harness():
    m = _load()
    assert m.CFG.mode == "harness"
    assert m.CFG.uses_tools is True
    assert m.CFG.allowed_models == ("gpt-6-luna", "claude-haiku-5-5")
    assert m.CFG.rounds == 6


def test_gemini_models_are_hard_blocked_even_if_configured():
    m = _load()
    assert not m.is_allowed("gemini-3.8-flash", ("gemini-3.8-flash",))
    assert not m.is_allowed("google/gemini-3.5-flash-lite", ("gemini-3.5-flash-lite",))
    assert m.is_allowed("openai/gpt-6-luna")
    assert m.is_allowed("anthropic/claude-haiku-5-5")


def test_mode_mapping():
    m = _load()
    for mode in m.TOOL_MODES:
        assert m.Config.from_dict({"mode": mode}).uses_tools is True
    for mode in m.PLAIN_MODES:
        cfg = m.Config.from_dict({"mode": mode})
        assert cfg.uses_tools is False
        assert cfg.rounds == 1


def test_config_from_dict_coerces_strings():
    cfg = _load().Config.from_dict({"mode": "plain", "limit": "100", "max_cost_usd": "2.5",
                                    "allowed_models": "gpt-6-luna,claude-haiku-5-5"})
    assert cfg.limit == 100
    assert cfg.max_cost_usd == 2.5
    assert cfg.allowed_models == ("gpt-6-luna", "claude-haiku-5-5")


def test_task_file_and_parse_models():
    m = _load()
    assert m.task_file("audit") == "audit_train.jsonl"
    assert m.parse_models(None) == ("gpt-6-luna", "claude-haiku-5-5")
    assert m.parse_models(" gpt-6-luna , claude-haiku-5-5 ") == (
        "gpt-6-luna", "claude-haiku-5-5")


def test_redaction_masks_full_assignment():
    m = _load()
    assert "hf_secret" not in m.redact("HF_TOKEN=hf_secret123")
    assert "apikey" not in m.redact("KAGGLE_API_TOKEN=apikey123")


def test_notebook_and_example_exist():
    assert (ROOT / "notebooks/kaggle_model_runner.ipynb").is_file()
    example = json.loads((ROOT / "notebooks/harness_config.example.json").read_text(encoding="utf-8"))
    assert example["allowed_models"] == ["gpt-6-luna", "claude-haiku-5-5"]
