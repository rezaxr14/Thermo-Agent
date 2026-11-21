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


def test_per_model_shard_mapping():
    cfg = _load().Config.from_dict({
        "shard": "default.jsonl",
        "model_shards": {"gpt-6-luna": "luna_batch.jsonl",
                         "claude-haiku-5-5": "haiku_batch.jsonl"},
    })
    assert cfg.shard_for("openai/gpt-6-luna") == "luna_batch.jsonl"
    assert cfg.shard_for("claude-haiku-5-5") == "haiku_batch.jsonl"
    assert cfg.shard_for("some-other-model") == "default.jsonl"


def test_equal_tiers_selection_and_pipeline_exposed():
    m = _load()
    cfg = m.Config.from_dict({"tiers": ["easy", "medium", "hard"], "equal_tiers": True,
                              "pipeline": {"gate_tokens": 8192, "ingest": True,
                                           "memory": True, "mix_refresh": True}})
    rows = ([{"id": f"e{i}", "tier": "easy"} for i in range(50)]
            + [{"id": f"m{i}", "cell": "repair/medium"} for i in range(40)]
            + [{"id": f"h{i}", "tier": "hard"} for i in range(60)])
    chosen = cfg.select_rows(rows)
    counts = {t: sum(1 for r in chosen if cfg.tier_of(r) == t) for t in m.TIERS}
    assert counts == {"easy": 40, "medium": 40, "hard": 40}
    assert cfg.pipeline["gate_tokens"] == 8192
    assert cfg.pipeline["mix_refresh"] is True


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


def test_explicit_shard_beats_export_directory():
    """Regression: attached training mixes must never replace the batch shard."""
    m = _load()
    shard = pathlib.Path("prompts_batch11_gpt-6-luna.jsonl")
    export = pathlib.Path("data/sft_mix_v3")
    assert m.choose_rows_source(shard.name, shard, export) == ("shard", shard)


def test_export_directory_is_only_a_fallback():
    m = _load()
    export = pathlib.Path("data/sft_mix_v3")
    assert m.choose_rows_source("missing.jsonl", None, export) == ("export", export)


def test_no_shard_and_no_export_fails_loudly():
    import pytest
    m = _load()
    with pytest.raises(SystemExit):
        m.choose_rows_source("missing.jsonl", None, None)


def test_notebook_matches_canonical_runner_source():
    """The .ipynb is derived; edit the .py and run notebooks/build_runner_notebook.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_runner_notebook", ROOT / "notebooks/build_runner_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    head, tail = builder.split_source(
        (ROOT / "notebooks/kaggle_model_runner.py").read_text(encoding="utf-8"))
    assert builder.notebook_code() == head + tail


def test_notebook_and_example_exist():
    assert (ROOT / "notebooks/kaggle_model_runner.ipynb").is_file()
    example = json.loads((ROOT / "notebooks/harness_config.example.json").read_text(encoding="utf-8"))
    assert example["allowed_models"] == ["gpt-6-luna", "claude-haiku-5-5"]

def test_partial_tier_shard_is_not_balanced_to_zero():
    """Regression: batch 11's haiku shard had easy+medium only; min() over all
    tiers was 0, so every row was dropped and the journal stayed empty."""
    m = _load()
    cfg = m.Config.from_dict({"tiers": ["easy", "medium", "hard"], "equal_tiers": True})
    rows = ([{"id": f"e{i}", "tier": "easy"} for i in range(3)]
            + [{"id": f"m{i}", "tier": "medium"} for i in range(3)])
    assert cfg.select_rows(rows) == rows


def test_full_tier_shard_is_trimmed_to_equal_thirds():
    m = _load()
    cfg = m.Config.from_dict({"tiers": ["easy", "medium", "hard"], "equal_tiers": True})
    rows = ([{"id": f"e{i}", "tier": "easy"} for i in range(5)]
            + [{"id": f"m{i}", "tier": "medium"} for i in range(5)]
            + [{"id": f"h{i}", "tier": "hard"} for i in range(2)])
    selected = cfg.select_rows(rows)
    assert len(selected) == 6
    assert {r["tier"] for r in selected} == {"easy", "medium", "hard"}


def test_untiered_shard_selects_nothing_so_the_runner_fails_loudly():
    m = _load()
    cfg = m.Config.from_dict({"tiers": ["easy", "medium", "hard"], "equal_tiers": True})
    assert cfg.select_rows([{"id": "x"}]) == []
