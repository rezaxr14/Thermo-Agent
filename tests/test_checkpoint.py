"""A Kaggle session cannot finish an epoch, so resuming must be automatic and must pick the newest checkpoint.

`train/sft.py` and `train/grpo.py` call `train(resume_from_checkpoint=latest(out))`; a first session starts with no
directory, and a restarted one continues from the highest `checkpoint-<step>` the Trainer wrote.
"""
import pathlib

from thermo_agent.train import config as C
from thermo_agent.train.checkpoint import latest


def test_no_directory_is_a_fresh_run(tmp_path):
    assert latest(tmp_path / "does-not-exist") is None


def test_empty_directory_is_a_fresh_run(tmp_path):
    assert latest(tmp_path) is None


def test_newest_checkpoint_wins_not_the_alphabetical_last(tmp_path):
    for step in (50, 1000, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert pathlib.Path(latest(tmp_path)).name == "checkpoint-1000"


def test_only_directories_named_like_checkpoints_count(tmp_path):
    (tmp_path / "checkpoint-300").write_text("a file, not a checkpoint", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "checkpoint-last").mkdir()
    assert latest(tmp_path) is None
    (tmp_path / "checkpoint-7").mkdir()
    assert pathlib.Path(latest(tmp_path)).name == "checkpoint-7"


def test_sft_saves_often_enough_for_a_twelve_hour_session():
    """The checkpoint interval must be short in wall-clock terms: 50 optimizer steps is about an hour (est.)."""
    s = C.SftSettings()
    assert s.epochs == 3 and s.early_stopping_patience == 3, "A1/A10: 3 epochs upper bound, early stopping decides"
    assert 0 < s.save_steps <= 500, "a 12 h session must contain several checkpoints, not one epoch boundary"
    assert s.eval_steps == s.save_steps


def test_grpo_checkpoints_on_steps_too():
    g = C.GrpoSettings()
    assert 0 < g.save_steps <= 100
