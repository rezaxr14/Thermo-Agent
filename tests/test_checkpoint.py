"""A Kaggle session cannot finish an epoch, so resuming must be automatic and must pick the newest checkpoint.

`train/sft.py` and `train/grpo.py` call `train(resume_from_checkpoint=latest(out))`; a first session starts with no
directory, and a restarted one continues from the highest `checkpoint-<step>` the Trainer wrote.
"""
import pathlib

from thermo_agent.train import config as C
from thermo_agent.train.checkpoint import is_complete, latest


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


def _complete_ckpt(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    (path / "trainer_state.json").write_text("{}", encoding="utf-8")
    (path / "optimizer.pt").write_bytes(b"state")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def test_complete_checkpoint_needs_optimizer_rng_and_weights(tmp_path):
    full = _complete_ckpt(tmp_path / "checkpoint-100")
    assert is_complete(full)
    for missing in ("trainer_state.json", "optimizer.pt", "adapter_model.safetensors"):
        saved = (full / missing).read_bytes()
        (full / missing).unlink()
        assert not is_complete(full)
        (full / missing).write_bytes(saved)


def test_resume_skips_an_interrupted_higher_step_checkpoint(tmp_path):
    _complete_ckpt(tmp_path / "checkpoint-50")
    # A session killed mid-save leaves the newest dir without optimizer/RNG state.
    partial = tmp_path / "checkpoint-100"
    partial.mkdir()
    (partial / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert latest(tmp_path) == str(partial)              # historical enumeration behaviour
    assert pathlib.Path(latest(tmp_path, complete=True)).name == "checkpoint-50"


def test_both_stages_resume_only_from_complete_checkpoints():
    for name in ("sft.py", "grpo.py"):
        src = (pathlib.Path(__file__).resolve().parents[1] / "thermo_agent" / "train" / name).read_text(encoding="utf-8")
        assert "latest(a.out, complete=True)" in src, f"{name} may resume from a broken checkpoint"

