"""Training-path audit 2026-11-19 item 7: the smoke v28 OOM fix is in the training path.

v28 died in `accelerator.backward` on an 8,192-token row with 13.55 GiB in use and
1.10 GiB requested at 1.01 GiB free. The saved activations for
`layers x sequence x hidden` dominate at that length; offloading them to pinned
host memory removes that peak without changing the model, labels or the 8,192 gate.
"""
from __future__ import annotations

import pathlib

from thermo_agent.train import config, grpo

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_checkpointing_keeps_reentrant_off_and_offloads_saved_activations():
    kw = config.activation_checkpointing_kwargs()
    assert kw["use_reentrant"] is False
    assert kw["offload"] is True


def test_both_trainers_pass_the_offload_settings_to_their_config():
    src = [(ROOT / "thermo_agent" / "train" / name).read_text(encoding="utf-8")
           for name in ("sft.py", "grpo.py")]
    assert all("gradient_checkpointing_kwargs=activation_checkpointing_kwargs()" in text for text in src)


def test_gate_and_completion_length_are_unchanged_by_the_fix():
    """The memory fix must not hide the 8,192 gate or shrink GRPO's 4,096 budget."""
    assert config.SftSettings().max_length == 8192
    assert config.GrpoSettings().max_completion_length == 4096
    assert grpo.DEFAULT_MAX_COMPLETION_LENGTH == 4096
