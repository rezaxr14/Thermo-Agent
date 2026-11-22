"""Eval-arm prompt construction: no invalid harness reaches H.augment, replay stays frozen.

Covers the Kaggle replay arms whose ``harness: "tools"`` label is not a real
``H.MODES`` value.  The runner is imported by path with all Kaggle I/O stubbed,
so no Kaggle worker / GPU / network is involved.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import types
from unittest import mock

from thermo_agent.agent import harness as H
from thermo_agent.agent import tool_prompt
from thermo_agent.eval import model_eval as ME

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "kaggle" / "eval_runner.py"


# ---------------------------------------------------------------------------
# Load eval_runner.py by path without executing any real Kaggle bootstrap code.
# ---------------------------------------------------------------------------
def _load_runner(tmp_path, monkeypatch, name="arm_prompt_runner"):
    root = tmp_path / "worker" / "thermo-agent"
    (root / "data" / "heldout").mkdir(parents=True, exist_ok=True)
    for sub in ("reports", "runs", "paper/assets"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "data" / "heldout" / "repair_test_id.jsonl").write_text('{"id":"held-1"}\n', encoding="utf-8")
    for k in (4, 8, 16):
        ck = root / "runs" / f"grpo_ledger_k{k}" / "checkpoint-1"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "adapter_config.json").write_text("{}", encoding="utf-8")
    merged = root / "runs" / "sft_merged"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")

    fake = types.ModuleType("thermo_agent.kaggle_notebook")
    fake.setup = lambda *a, **k: root
    fake.find_input = lambda *a, **k: root
    fake.run = lambda *a, **k: None
    fake.token = lambda *a, **k: ""
    fake.assert_pinned_environment = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "thermo_agent.kaggle_notebook", fake)
    for key in ("HF_TOKEN", "ADAPTER", "GRPO_ADAPTER", "TASK", "FORCE", "SPLITS", "LIMIT", "DATA_DIRS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TASK", "ledger")
    monkeypatch.setenv("ARMS", "ours_base")  # keep import-time work minimal

    walk = [("/kaggle/input", [], ["kaggle_bundle.zip"])]
    with mock.patch("os.walk", side_effect=lambda *a, **k: iter(walk)), \
         mock.patch("zipfile.ZipFile"), mock.patch("shutil.rmtree"), mock.patch("os.chdir"), \
         mock.patch("subprocess.run", side_effect=lambda argv, *a, **k: subprocess.CompletedProcess(argv, 0)):
        spec = importlib.util.spec_from_file_location(name, EVAL_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class _FakeGen:
    """Fake ``generate(messages) -> str`` that records every prompt it is handed."""

    def __init__(self, reply="final"):
        self.reply = reply
        self.prompts = []
        self.current_id = None

    def set_current_id(self, rid):
        self.current_id = rid

    def __call__(self, messages):
        self.prompts.append(messages)
        return self.reply


def _data_dir(tmp_path, row):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "repair_test_id.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# 1. every arm constructs; no invalid harness reaches the shared builder
# ---------------------------------------------------------------------------
def test_every_runner_arm_constructs_with_a_valid_harness(tmp_path, monkeypatch, corpus_fixture_rows):
    module = _load_runner(tmp_path, monkeypatch, name="arm_prompt_all")
    data = _data_dir(tmp_path / "all", corpus_fixture_rows["repair"][0])

    seen = []
    real_build = ME.build_harness_tool_messages
    real_compact = ME._build_compact_harness_messages

    def recording_build(messages, task, plant, mode="agents+skills", *args, **kwargs):
        seen.append(mode)
        return real_build(messages, task, plant, mode=mode, *args, **kwargs)

    def recording_compact(messages, mode, task, plant, *args, **kwargs):
        seen.append(mode)
        return real_compact(messages, mode, task, plant, *args, **kwargs)

    monkeypatch.setattr(ME, "build_harness_tool_messages", recording_build)
    monkeypatch.setattr(ME, "_build_compact_harness_messages", recording_compact)

    for name, spec in module.SPECS.items():
        gen = _FakeGen("not an answer")
        ME.run(gen, data, ["repair"], splits=["test_id"], limit=1,
               mode=spec["harness"], tool_access=spec["tool_access"])
        assert gen.prompts, f"{name} never built a prompt"

    assert seen, "expected at least one augmentation call"
    invalid = set(seen) - set(H.MODES)
    assert not invalid, f"invalid harness reached the shared builder: {invalid}"


# ---------------------------------------------------------------------------
# 2. normalization + CLI alias for the legacy "tools" label
# ---------------------------------------------------------------------------
def test_normalize_harness_maps_tools_to_full_harness():
    assert ME.normalize_harness("tools") == "agents+skills"
    assert ME.normalize_harness("none") == "none"
    assert ME.normalize_harness("agents+skills") == "agents+skills"
    assert ME.normalize_harness(None) == "none"


def test_cli_tools_alias_notes_label_and_records_both_labels(tmp_path, monkeypatch, capsys,
                                                             corpus_fixture_rows):
    row = corpus_fixture_rows["repair"][0]
    data = _data_dir(tmp_path / "cli", row)
    replay = tmp_path / "cli" / "replay.jsonl"
    replay.write_text(json.dumps({"id": row["id"], "completion": "stored answer"}) + "\n", encoding="utf-8")

    seen = []
    real_build = ME.build_harness_tool_messages

    def recording_build(messages, task, plant, mode="agents+skills", *args, **kwargs):
        seen.append(mode)
        return real_build(messages, task, plant, mode=mode, *args, **kwargs)

    monkeypatch.setattr(ME, "build_harness_tool_messages", recording_build)
    out = tmp_path / "cli" / "tools_alias.json"
    monkeypatch.setattr(sys, "argv", [
        "model_eval", str(data), "--backend", "replay", "--replay", str(replay),
        "--tasks", "repair", "--splits", "test_id", "--limit", "1",
        # exact shape of the frontier_tools / frontier_harness runner command:
        "--harness", "tools", "--tool-access", "tools", "--out", str(out),
    ])
    ME.main()

    note = capsys.readouterr().out.splitlines()[0]
    assert "--harness tools" in note and "agents+skills" in note and "harness_label=tools" in note
    # the tool episode uses the shared compact builder with the normalized mode.
    assert "agents+skills" in seen
    assert set(seen) <= set(H.MODES), f"invalid harness reached the shared builder: {set(seen) - set(H.MODES)}"

    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["harness_config"]["mode"] == "agents+skills"
    assert rep["meta"]["harness"] == "agents+skills"
    assert rep["meta"]["harness_mode"] == "agents+skills"
    assert rep["meta"]["harness_label"] == "tools"
    assert rep["arm_config"]["harness"] == "agents+skills"
    assert rep["arm_config"]["harness_label"] == "tools"
    assert rep["arm_config"]["harness_mode"] == "agents+skills"


# ---------------------------------------------------------------------------
# 3. replay is message-ignoring: the stored completion is always returned
# ---------------------------------------------------------------------------
def test_replay_generator_ignores_messages_and_returns_stored_completion(tmp_path):
    stored = "<diagnosis>stored</diagnosis><edits>SET x = 1</edits>"
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps({"id": "r1", "completion": stored}) + "\n", encoding="utf-8")
    gen = ME.ReplayGenerator(replay)
    gen.set_current_id("r1")

    a = gen([{"role": "system", "content": "one"}, {"role": "user", "content": "prompt A"}])
    b = gen([{"role": "system", "content": "totally different"}, {"role": "user", "content": "prompt B"}])
    c = gen([])
    assert a == b == c == stored


# ---------------------------------------------------------------------------
# 4. ours_harness uses the shared tool prompt; ours_sft does not
# ---------------------------------------------------------------------------
def test_ours_harness_uses_shared_tool_prompt_ours_sft_does_not(tmp_path, monkeypatch, corpus_fixture_rows):
    module = _load_runner(tmp_path, monkeypatch, name="arm_prompt_ours")
    assert ME.build_harness_tool_messages is tool_prompt.build_harness_tool_messages

    harness_spec = module.SPECS["ours_harness"]
    sft_spec = module.SPECS["ours_sft"]
    assert harness_spec["tool_access"] == "tools" and sft_spec["tool_access"] == "none"
    assert harness_spec["harness"] == sft_spec["harness"] == "agents+skills"

    data = _data_dir(tmp_path / "ours", corpus_fixture_rows["repair"][0])
    prompts = {}
    for name, spec in (("ours_harness", harness_spec), ("ours_sft", sft_spec)):
        gen = _FakeGen("final")
        ME.run(gen, data, ["repair"], splits=["test_id"], limit=1,
               mode=spec["harness"], tool_access=spec["tool_access"])
        prompts[name] = gen.prompts[0][0]["content"]

    h_sys, s_sys = prompts["ours_harness"], prompts["ours_sft"]
    # Both arms carry the same compact AGENTS block; only the tool arm gets the contract.
    assert "# AGENTS.md: energy-plant tasks (compact)" in h_sys
    assert "# AGENTS.md: energy-plant tasks (compact)" in s_sys
    assert "MULTI-TURN TOOL ENVIRONMENT" in h_sys
    assert "MULTI-TURN TOOL ENVIRONMENT" in h_sys and "inspect_plant" in h_sys
    assert "<tool_call name=" not in h_sys
    assert "MULTI-TURN TOOL ENVIRONMENT" not in s_sys
    assert "<tool_call name=" not in s_sys
