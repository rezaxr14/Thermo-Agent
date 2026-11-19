"""F4 evaluation arms and sampled pass@k (DECISIONS §2F).

The arm table is checked statically through the AST (no Kaggle bootstrap / GPU), and
generation is checked with fake tokenizer/model objects so no weights are downloaded.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import pathlib
import sys

import pytest

from thermo_agent.eval import model_eval as ME

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "kaggle" / "eval_runner.py"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_prompt_separation.py"

F4_FAMILIES = ("ours_base", "ours_sft", "ours_sft_grpo_k4",
               "ours_sft_grpo_k8", "ours_sft_grpo_k16")
QWEN_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}


@pytest.fixture(scope="module")
def audit_mod():
    spec = importlib.util.spec_from_file_location("audit_prompt_separation", AUDIT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. static arm-table guard
# ---------------------------------------------------------------------------
def test_every_local_family_has_z_and_h_counterpart(audit_mod):
    specs = audit_mod.read_specs(EVAL_RUNNER)
    for family in F4_FAMILIES:
        z, h = f"{family}_z", f"{family}_h"
        assert z in specs, f"missing Z arm {z}"
        assert h in specs, f"missing H arm {h}"
        assert specs[z]["backend"] == "hf" and specs[h]["backend"] == "hf"
        assert specs[z]["harness"] == "none" and specs[z]["tool_access"] == "none"
        assert specs[h]["harness"] == "agents+skills" and specs[h]["tool_access"] == "tools"
        # GRPO arms keep tools in H only (the explicit table arms carry grpo_k).
        if "grpo" in family:
            assert specs[z]["grpo_k"] == specs[h]["grpo_k"]
            assert specs[z]["tool_access"] == "none" and specs[h]["tool_access"] == "tools"


def test_no_z_arm_carries_harness_or_tool_access(audit_mod):
    specs = audit_mod.read_specs(EVAL_RUNNER)
    z_arms = [n for n, s in specs.items()
              if s.get("mode") == "Z" or n.endswith("_z")]
    assert len(z_arms) >= len(F4_FAMILIES) + 1, "expected the F4 Z table plus frontier Z"
    for name in z_arms:
        spec = specs[name]
        assert spec["harness"] == "none", f"{name} is a Z arm but has harness {spec['harness']!r}"
        assert spec["tool_access"] == "none", f"{name} is a Z arm but has tools"
        assert "zero_shot" in spec and "tool_access" in spec and "harness" in spec


def test_legacy_report_arms_and_replay_files_are_kept(audit_mod):
    specs = audit_mod.read_specs(EVAL_RUNNER)
    legacy = ("ours_base", "ours_sft", "ours_harness", "ours_grpo", "ours_sft_grpo_k4",
              "ours_sft_grpo_k8", "ours_sft_grpo_k16", "frontier_zero", "frontier_tools",
              "frontier_harness", "other_model", "other_model_api")
    assert set(legacy) <= set(specs)
    # Replay paths are expressions (str(WORK / "...")), so check them in the AST source.
    source = EVAL_RUNNER.read_text(encoding="utf-8")
    for name in ("eval_all.jsonl", "harness_completions.jsonl",
                 "harness_eval_completions2.jsonl", "eval_gptnano_replay.jsonl"):
        assert f'"{name}"' in source, f"lost replay file {name}"
    # DECISIONS: frontier Z replays eval_all.jsonl, frontier H replays the v6 tools journal.
    assert 'str(WORK / "eval_all.jsonl")' in source
    for arm in ("frontier_h", "frontier_harness", "frontier_tools"):
        i = source.index(f'"{arm}":')
        chunk = source[i:i + 300]
        assert ('harness_eval_completions2.jsonl' in chunk or
                'harness_completions.jsonl' in chunk), f"{arm} lost its replay file"
    assert specs["frontier_zero"]["harness"] == "none"
    assert specs["frontier_h"]["tool_access"] == "tools"


def test_audit_script_passes_repo_and_fails_a_mislabeled_z_arm(audit_mod, tmp_path):
    rows, failures = audit_mod.audit(audit_mod.read_specs(EVAL_RUNNER))
    assert not failures, failures
    assert any(r["arm"] == "ours_sft_h" and r["tool_contract"] for r in rows)
    assert any(r["arm"] == "ours_sft_z" and not r["harness_marker"] for r in rows)

    bad = tmp_path / "bad_runner.py"
    bad.write_text(
        'SPECS = {"bad_z": {"backend": "hf", "mode": "Z", "harness": "agents+skills", '
        '"tool_access": "tools", "zero_shot": False}}\n', encoding="utf-8")
    _rows, bad_failures = audit_mod.audit(audit_mod.read_specs(bad))
    assert bad_failures and "Z prompt contains the harness marker" in bad_failures[0]
    assert "Z prompt contains the tool contract" in bad_failures[0]


# ---------------------------------------------------------------------------
# 2. greedy default / exact sampling values
# ---------------------------------------------------------------------------
class _Inputs:
    def __init__(self):
        self._data = {"input_ids": _Tensor(4)}

    def to(self, device):
        return self

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class _Tensor:
    def __init__(self, length):
        self.shape = (1, length)


class _Seq:
    """Minimal torch-tensor stand-in: indexable with a `.shape`."""

    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _Seq(self.values[item])
        return self.values[item]


class _Tok:
    def apply_chat_template(self, messages, **kwargs):
        return "rendered prompt"

    def __call__(self, text, **kwargs):
        return _Inputs()

    def decode(self, tokens, **kwargs):
        return "final answer"


class _Model:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [_Seq([0, 0, 0, 1])]


def _fake_generator(**sampling):
    gen = object.__new__(ME.HFGenerator)
    gen.tok, gen.model = _Tok(), _Model()
    gen.max_new_tokens = 4096
    gen.enable_thinking = True
    gen.do_sample = bool(sampling.pop("_do_sample", sampling))
    gen.temperature = sampling.get("temperature")
    gen.top_p = sampling.get("top_p")
    gen.top_k = sampling.get("top_k")
    gen.min_p = sampling.get("min_p")
    gen.num_samples = sampling.get("num_samples", 1)
    gen.seed = sampling.get("seed")
    gen.repetition_penalty = 1.0
    gen.presence_penalty = None
    gen.last_prompt_tokens = gen.last_completion_tokens = 0
    gen.last_truncated = False
    return gen


def test_greedy_is_default_and_generate_kwargs_are_unchanged():
    params = inspect.signature(ME.HFGenerator.__init__).parameters
    for name in ("temperature", "top_p", "top_k", "min_p", "seed"):
        assert params[name].default is None
    assert params["num_samples"].default == 1
    gen = _fake_generator(_do_sample=False)
    assert gen._decode_kwargs() == {"max_new_tokens": 4096, "do_sample": False}
    assert gen([{"role": "user", "content": "hello"}]) == "final answer"
    assert {k: v for k, v in gen.model.kwargs.items() if k != "input_ids"} == {
        "max_new_tokens": 4096, "do_sample": False}


def test_sampling_sets_exact_qwen_values_and_never_presence_penalty():
    assert ME.HFGenerator.PASS_AT_K_SAMPLING == QWEN_SAMPLING
    gen = _fake_generator(**QWEN_SAMPLING, _do_sample=True)
    kw = gen._decode_kwargs()
    assert kw == {"max_new_tokens": 4096, "do_sample": True, **QWEN_SAMPLING,
                  "repetition_penalty": 1.0}
    assert "presence_penalty" not in kw
    gen([{"role": "user", "content": "hello"}])
    assert {k: v for k, v in gen.model.kwargs.items() if k != "input_ids"} == kw


# ---------------------------------------------------------------------------
# 3. CLI pass@k end to end (fake generator, no GPU)
# ---------------------------------------------------------------------------
def test_pass_at_k_cli_writes_k_samples_and_a_pass_at_k_field(tmp_path, monkeypatch,
                                                              corpus_fixture_rows, capsys):
    row = corpus_fixture_rows["repair"][0]
    truth = row["completion"][0]["content"]
    data = tmp_path / "data"
    data.mkdir()
    (data / "repair_test_id.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "pass8.json"
    made = []

    class FakeHF:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.max_new_tokens = 4096
            self.seed = None
            made.append(self)

        def set_seed(self, seed):
            self.seed = seed

        def __call__(self, messages):
            return truth

    monkeypatch.setattr(ME, "HFGenerator", FakeHF)
    monkeypatch.setattr(sys, "argv", [
        "model_eval", str(data), "--tasks", "repair", "--splits", "test_id", "--limit", "1",
        "--out", str(out), "--pass-at-k", "8"])
    ME.main()
    capsys.readouterr()

    assert len(made) == 1
    assert made[0].kwargs == {**QWEN_SAMPLING, "seed": 0, "enable_thinking": True}
    assert made[0].seed == 7, "the eight passes must use seeds 0..7"
    report = json.loads(out.read_text(encoding="utf-8"))
    pak = report["pass_at_k"]
    assert pak["k"] == 8 and pak["seeds"] == list(range(8))
    assert pak["sampling"] == QWEN_SAMPLING
    assert pak["per_task_split"]["repair/test_id"]["n"] == 1
    assert pak["pass_at_k"] == 1.0
    assert report["meta"]["sampling"]["enabled"] is True
    samples = [json.loads(x) for x in out.with_suffix(".samples.jsonl").read_text().splitlines()]
    assert len(samples) == 8 and [s["seed"] for s in samples] == list(range(8))
    assert all(s["sample"] == i for i, s in enumerate(samples))


def test_greedy_cli_has_no_pass_at_k_field(tmp_path, monkeypatch, corpus_fixture_rows, capsys):
    row = corpus_fixture_rows["repair"][0]
    data = tmp_path / "data"
    data.mkdir()
    (data / "repair_test_id.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "greedy.json"
    made = []

    class FakeHF:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.max_new_tokens = 4096
            made.append(self)

        def __call__(self, messages):
            return row["completion"][0]["content"]

    monkeypatch.setattr(ME, "HFGenerator", FakeHF)
    monkeypatch.setattr(sys, "argv", [
        "model_eval", str(data), "--tasks", "repair", "--splits", "test_id", "--limit", "1",
        "--out", str(out)])
    ME.main()
    capsys.readouterr()
    assert not any(k in made[0].kwargs for k in
                   ("temperature", "top_p", "top_k", "min_p", "seed", "num_samples"))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "pass_at_k" not in report
    assert report["meta"]["sampling"]["enabled"] is False
    assert not out.with_suffix(".samples.jsonl").exists()
