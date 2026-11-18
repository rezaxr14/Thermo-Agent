"""Resume-state relay: only complete checkpoints are restored, newest first."""
import json

from thermo_agent.kaggle_notebook import restore_checkpoints


def _checkpoint(path, step, complete=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    if complete:
        (path / "adapter_model.safetensors").write_bytes(b"adapter")
        (path / "optimizer.pt").write_bytes(b"optimizer")
    return path


def test_restores_newest_complete_checkpoint(tmp_path):
    source = tmp_path / "resume"
    old = _checkpoint(source / "sft" / "checkpoint-100", 100)
    newest = _checkpoint(source / "sft" / "checkpoint-300", 300)
    _checkpoint(source / "sft" / "checkpoint-400", 400, complete=False)
    root = tmp_path / "work"

    restored = restore_checkpoints(root, stages=("sft",), source=source)

    assert list(restored) == ["sft"]
    assert restored["sft"].endswith("checkpoint-300")
    assert (root / "runs" / "sft" / "checkpoint-300" / "optimizer.pt").exists()
    assert not (root / "runs" / "sft" / "checkpoint-100").exists()
    assert not (root / "runs" / "sft" / "checkpoint-400").exists()


def test_does_not_overwrite_a_local_checkpoint(tmp_path):
    source = tmp_path / "resume"
    _checkpoint(source / "sft" / "checkpoint-200", 200)
    local = _checkpoint(tmp_path / "work" / "runs" / "sft" / "checkpoint-200", 200)
    (local / "optimizer.pt").write_bytes(b"local state")

    restore_checkpoints(tmp_path / "work", stages=("sft",), source=source)

    assert (local / "optimizer.pt").read_bytes() == b"local state"


def test_missing_resume_dataset_is_a_fresh_run(tmp_path):
    assert restore_checkpoints(tmp_path, stages=("sft",), source=tmp_path / "absent") == {}
