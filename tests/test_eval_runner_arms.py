"""Prove the Kaggle evaluation runner scores exactly the required arms on the held-out set.

`kaggle/eval_runner.py` is notebook source: it bootstraps from /kaggle/input and runs every arm at
import time.  We load it by path with the Kaggle helpers stubbed and subprocess calls captured, then
inspect the arm specs, checkpoint resolution and the `model_eval` command lines it would execute.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import types
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "kaggle" / "eval_runner.py"

REQUIRED_ARMS = (
    "ours_base", "ours_sft", "ours_harness",
    "ours_sft_grpo_k4", "ours_sft_grpo_k8", "ours_sft_grpo_k16",
    "frontier_zero", "frontier_tools", "frontier_harness", "other_model",
)


def _fake_kaggle_notebook(root, replay_source):
    """The import-time Kaggle helpers: harmless values rooted in a temp worker dir."""
    mod = types.ModuleType("thermo_agent.kaggle_notebook")
    mod.setup = lambda *a, **k: root
    mod.find_input = lambda name, *a, **k: replay_source
    mod.run = lambda cmd, *a, **k: None
    mod.token = lambda *a, **k: ""
    mod.assert_pinned_environment = lambda *a, **k: None
    mod.restore_checkpoints = lambda *a, **k: {}
    mod.status = lambda *a, **k: root / "status.json"
    return mod


def _load(tmp_path, monkeypatch, *, data_dirs=None, name="eval_runner_under_test",
          create_adapters=True, task="ledger", premade_reports=None):
    """Import eval_runner.py by path with Kaggle bootstrap / subprocess fully stubbed.

    Returns ``(module, cmds)`` where ``cmds`` captures every subprocess argv the module issued.
    ``create_adapters`` writes plausible checkpoint dirs so the GRPO adapter guard passes; turn it
    off to exercise the missing-adapter failure. ``premade_reports`` pre-creates report JSONs.
    """
    root = tmp_path / "worker" / "thermo-agent"
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("paper/assets", "reports", "runs", "data", "replay_inputs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # eval_runner refuses to score before checking the frozen held-out corpus at import time.
    heldout = root / "data" / "heldout"
    heldout.mkdir(parents=True, exist_ok=True)
    (heldout / "repair_test_id.jsonl").write_text('{"id": "held-1"}\n', encoding="utf-8")

    # A CPU worker with no secrets and no local adapter env: GRPO reads runs/grpo_k<k> first,
    # falling back to the legacy runs/grpo_<task>_k<k> when only an old checkpoint exists.
    for key in ("HF_TOKEN", "ADAPTER", "GRPO_ADAPTER", "TASK", "FORCE", "SPLITS", "LIMIT"):
        monkeypatch.delenv(key, raising=False)
    if task:
        monkeypatch.setenv("TASK", task)
    if data_dirs is None:
        monkeypatch.delenv("DATA_DIRS", raising=False)
    else:
        monkeypatch.setenv("DATA_DIRS", data_dirs)
    monkeypatch.setenv("ARMS", ",".join(REQUIRED_ARMS))

    if create_adapters:
        for k in (4, 8, 16):
            for stage in (f"grpo_k{k}", f"grpo_{task}_k{k}"):
                ck = root / "runs" / stage / "checkpoint-1"
                ck.mkdir(parents=True, exist_ok=True)
                (ck / "adapter_config.json").write_text("{}", encoding="utf-8")
        merged = root / "runs" / "sft_merged"
        merged.mkdir(parents=True, exist_ok=True)
        (merged / "config.json").write_text("{}", encoding="utf-8")
    for arm, report in (premade_reports or {}).items():
        (root / "reports" / f"{arm}.json").write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setitem(sys.modules, "thermo_agent.kaggle_notebook",
                        _fake_kaggle_notebook(root, root / "replay_inputs"))

    cmds = []

    def fake_run(argv, *a, **k):
        cmds.append([str(x) for x in argv])
        return subprocess.CompletedProcess(argv, 0)

    walk_result = [("/kaggle/input", [], ["kaggle_bundle.zip"])]
    with mock.patch("os.walk", side_effect=lambda *a, **k: iter(walk_result)), \
         mock.patch("zipfile.ZipFile"), \
         mock.patch("shutil.rmtree"), \
         mock.patch("os.chdir"), \
         mock.patch("subprocess.run", side_effect=fake_run):
        spec = importlib.util.spec_from_file_location(name, EVAL_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, cmds


def _run_arm_captured(module, arm, sink):
    """Call ``run_arm`` after import with subprocess.run stubbed, capturing its command."""
    def fake_run(argv, *a, **k):
        sink.append([str(x) for x in argv])
        return subprocess.CompletedProcess(argv, 0)

    with mock.patch("subprocess.run", side_effect=fake_run):
        module.run_arm(arm)
    return sink[-1]


def _model_eval_cmds(cmds):
    return [c for c in cmds if "thermo_agent.eval.model_eval" in c]


def _display_name(cmd):
    return cmd[cmd.index("--display-name") + 1]


def test_a_required_arms_are_the_exact_scored_set(tmp_path, monkeypatch):
    module, cmds = _load(tmp_path, monkeypatch, data_dirs="heldout", name="er_arms")
    assert set(REQUIRED_ARMS) <= set(module.ARMS)
    assert set(REQUIRED_ARMS) <= set(module.SPECS)
    # exactly these arms are invoked at import time: no required arm skipped, no extra arm scored
    assert {_display_name(c) for c in _model_eval_cmds(cmds)} == set(REQUIRED_ARMS)


def test_b_data_dirs_default_is_the_frozen_heldout_set(tmp_path, monkeypatch):
    held, _ = _load(tmp_path / "held", monkeypatch, data_dirs="heldout", name="er_held")
    assert held.DATA_DIRS == [held.ROOT / "data" / "heldout"]

    default, _ = _load(tmp_path / "default", monkeypatch, data_dirs=None, name="er_default")
    assert default.DATA_DIRS == [default.ROOT / "data" / "heldout"]
    assert default.DEFAULT_DATA_DIRS == "heldout"


def test_c_grpo_k_arms_resolve_distinct_checkpoint_dirs(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch, data_dirs="heldout", name="er_grpo")
    adapters = module.GRPO_K_ADAPTERS
    assert set(adapters) == {4, 8, 16}
    expected = {n: module.ROOT / "runs" / f"grpo_k{n}" for n in (4, 8, 16)}
    assert {n: adapters[n] for n in (4, 8, 16)} == expected
    assert len({str(adapters[n]) for n in (4, 8, 16)}) == 3, "k=4/8/16 must not share weights"

    # each arm passes its OWN adapter AND the merged SFT base, never base Qwen
    for n in (4, 8, 16):
        cmd = _run_arm_captured(module, f"ours_sft_grpo_k{n}", [])
        eval_cmd = _model_eval_cmds([cmd])[0]
        assert eval_cmd[eval_cmd.index("--adapter") + 1] == str(adapters[n])
        assert eval_cmd[eval_cmd.index("--model") + 1] == module.SFT_MERGED
        assert "Qwen/Qwen3.5-4B" not in eval_cmd


def test_d_replay_paths_resolve_to_the_new_completions_file(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch, data_dirs="heldout", name="er_replay")
    specs = module.SPECS
    assert specs["frontier_tools"]["replay"] == str(module.WORK / "harness_eval_completions2.jsonl")
    assert "harness_eval_completions.jsonl" not in specs["frontier_tools"]["replay"]
    assert "harness_eval_completions2.jsonl" in module.REPLAY_FILES
    assert "harness_eval_completions.jsonl" not in module.REPLAY_FILES
    replays = {name: s["replay"] for name, s in specs.items() if "replay" in s}
    assert replays, "expected replay-backed arms"
    assert all(isinstance(p, str) for p in replays.values())

    cmd = _run_arm_captured(module, "frontier_tools", [])
    assert cmd[cmd.index("--replay") + 1] == specs["frontier_tools"]["replay"]


def test_e_model_eval_cmd_uses_data_dirs_not_hardcoded(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch, data_dirs="heldout", name="er_cmd")
    cmd = _run_arm_captured(module, "ours_sft", [])
    dirs = [str(d) for d in module.DATA_DIRS]
    i = cmd.index(dirs[0])
    assert cmd[i:i + len(dirs)] == dirs
    assert not any("processed" in arg for arg in cmd), "no hardcoded processed/processed_arch"


def test_f_ours_sft_and_grpo_arms_share_harness_and_splits(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch, data_dirs="heldout", name="er_harness")
    names = ("ours_sft", "ours_sft_grpo_k4", "ours_sft_grpo_k8", "ours_sft_grpo_k16")
    assert {module.SPECS[n]["harness"] for n in names} == {"agents+skills"}
    for n in names:
        cmd = _model_eval_cmds([_run_arm_captured(module, n, [])])[0]
        assert cmd[cmd.index("--harness") + 1] == "agents+skills"
        assert "--splits" not in cmd, "no arm is restricted to a subset of splits"
    # frontier_tools used to be pinned to test_id; it now scores the same splits as every arm.
    ft = _model_eval_cmds([_run_arm_captured(module, "frontier_tools", [])])[0]
    assert "--splits" not in ft


def test_h_missing_grpo_adapter_raises(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="GRPO adapter"):
        _load(tmp_path, monkeypatch, data_dirs="heldout", name="er_missing",
              create_adapters=False)


def test_i_arm_config_records_train_task_in_report_and_manifest(tmp_path, monkeypatch):
    module, _ = _load(
        tmp_path / "cfg", monkeypatch, data_dirs="heldout", name="er_cfg",
        premade_reports={"ours_sft_grpo_k4": {"meta": {"display_name": "ours_sft_grpo_k4"}}},
    )
    report = json.loads((module.ROOT / "reports" / "ours_sft_grpo_k4.json").read_text(encoding="utf-8"))
    cfg = report["arm_config"]
    assert cfg["train_task"] == "ledger"
    assert cfg["grpo_k"] == 4
    assert cfg["harness"] == "agents+skills"
    assert cfg["data_dirs"] == [str(module.ROOT / "data" / "heldout")]
    manifest = json.loads(
        (module.ROOT / "paper" / "assets" / "model_results_manifest.json").read_text(encoding="utf-8"))
    assert manifest["arms"]["ours_sft_grpo_k4"]["train_task"] == "ledger"
