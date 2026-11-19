"""P0-10: one grpo_{task}_k{k} stage name across trainer, HF sync and eval.

Three group sizes must produce three distinct remote prefixes, the eval runner must pull those
same names, and a missing adapter must raise instead of scoring the base model under a GRPO label.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from thermo_agent.train import grpo as grpo_mod  # noqa: E402
from thermo_agent.train import hf_sync  # noqa: E402
from test_eval_runner_arms import _load  # noqa: E402

K_VALUES = (4, 8, 16)
STAGES = [grpo_mod.stage_name("ledger", k) for k in K_VALUES]


def test_stage_name_encodes_task_and_k():
    assert STAGES == ["grpo_ledger_k4", "grpo_ledger_k8", "grpo_ledger_k16"]
    assert len(set(STAGES)) == 3
    assert grpo_mod.stage_name("repair", 4) == "grpo_repair_k4"
    assert grpo_mod.stage_name("repair", 4) != grpo_mod.stage_name("audit", 4)


def test_three_k_values_give_three_distinct_remote_stages():
    # hf_sync.remote_steps keys off "<stage>/checkpoint-". Each stage must select only its own
    # checkpoint, so one synthetic remote listing yields three disjoint step sets.
    files = [f"{stage}/checkpoint-{k}/adapter_model.safetensors" for k, stage in zip(K_VALUES, STAGES)]
    selected = [hf_sync.remote_steps(files, stage) for stage in STAGES]
    assert selected == [[4], [8], [16]]
    assert len({tuple(steps) for steps in selected}) == 3


def test_trainer_uses_the_same_stage_name_and_merged_sft_init():
    src = (ROOT / "thermo_agent" / "train" / "grpo.py").read_text(encoding="utf-8")
    assert "stage = stage_name(a.task, g.num_generations)" in src
    assert grpo_mod.DEFAULT_INIT == "runs/sft_merged"


def test_train_runner_passes_merged_sft_as_init_and_the_k_stage():
    src = (ROOT / "kaggle" / "train_runner.py").read_text(encoding="utf-8")
    assert "INIT = str(merged)" in src
    assert "hf:merged/sft" not in src
    assert '"--init", INIT' in src
    assert 'GRPO_STAGE = f"grpo_{TASK}_k{GRPO_K}"' in src


def test_eval_pull_uses_the_same_stage_names(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch, data_dirs="heldout", name="stages_ok")
    assert {k: str(module.GRPO_K_ADAPTERS[k]) for k in K_VALUES} == {
        k: str(module.ROOT / "runs" / stage) for k, stage in zip(K_VALUES, STAGES)
    }


def test_missing_adapter_raises_instead_of_scoring_the_base_model(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="GRPO adapter"):
        _load(tmp_path, monkeypatch, data_dirs="heldout", name="stages_missing",
              create_adapters=False)
