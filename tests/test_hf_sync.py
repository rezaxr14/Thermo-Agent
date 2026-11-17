"""Resumable training state must go to exactly one private HF repo, and a restart must restore it.

The tests use a fake HfApi and a fake downloader: no network call and no token is needed. They pin the safety
rule (only `<token owner>/thermo-agent-checkpoints`), the full-state file list, the "newer checkpoint wins"
logic, and the Trainer hook used by SFT and GRPO.
"""
import json
import pathlib
import types

import pytest

from thermo_agent.train import ddp, hf_sync

TOKEN = "hf_test_token"


class FakeApi:
    def __init__(self, exists=True, private=True, files=()):
        self.exists, self.private = exists, private
        self.files = list(files)
        self.uploads, self.created = [], []

    def whoami(self):
        return {"name": "Reza-Nadimi"}

    def repo_info(self, repo_id, token=None):
        if not self.exists:
            raise Missing("not found")
        return types.SimpleNamespace(private=self.private)

    def create_repo(self, repo_id, token=None, private=None, exist_ok=False):
        self.created.append((repo_id, private))
        self.exists, self.private = True, private
        return None

    def list_repo_files(self, repo_id, token=None):
        return self.files

    def upload_folder(self, **kw):
        self.uploads.append(("folder", kw))

    def upload_file(self, **kw):
        self.uploads.append(("file", kw))


class Missing(Exception):
    pass


@pytest.fixture(autouse=True)
def _repo_not_found(monkeypatch):
    monkeypatch.setattr("huggingface_hub.errors.RepositoryNotFoundError", Missing, raising=False)


def test_token_read_from_env_then_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("# comment\nHF_TOKEN=hf_from_dotenv\nOTHER=x\n", encoding="utf-8")
    assert hf_sync.read_token(tmp_path / ".env") == "hf_from_dotenv"
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert hf_sync.read_token(tmp_path / ".env") == "hf_from_env"


def test_only_the_project_repo_is_allowed(monkeypatch):
    api = FakeApi()
    assert hf_sync.resolve_repo(None, TOKEN, api) == "Reza-Nadimi/thermo-agent-checkpoints"
    assert hf_sync.resolve_repo("Reza-Nadimi/thermo-agent-checkpoints", TOKEN, api).endswith(hf_sync.DEFAULT_REPO_NAME)
    for bad in ("someone-else/thermo-agent-checkpoints", "Reza-Nadimi/other-repo", "org/repo"):
        with pytest.raises(ValueError, match="refusing HF repo"):
            hf_sync.resolve_repo(bad, TOKEN, api)
    monkeypatch.setenv("HF_REPO_ID", "someone-else/thermo-agent-checkpoints")
    with pytest.raises(ValueError):
        hf_sync.resolve_repo(None, TOKEN, api)


def test_repo_is_created_private_and_a_public_repo_is_refused():
    api = FakeApi(exists=False)
    repo = hf_sync.ensure_private_repo(None, TOKEN, api)
    assert repo == "Reza-Nadimi/thermo-agent-checkpoints"
    assert api.created == [(repo, True)]
    api = FakeApi(exists=True, private=False)
    with pytest.raises(SystemExit, match="not private"):
        hf_sync.ensure_private_repo(None, TOKEN, api)


def test_remote_and_local_steps_are_numbers_not_names(tmp_path):
    files = ["sft/checkpoint-5/optimizer.pt", "sft/checkpoint-100/trainer_state.json",
             "sft/checkpoint-7/adapter_model.safetensors", "grpo_ledger/checkpoint-9/optimizer.pt",
             "sft/run.json", "sft/checkpoint-1000"]
    assert hf_sync.remote_steps(files, "sft") == [5, 7, 100]
    for step in (5, 1000, 7):
        (tmp_path / "sft" / f"checkpoint-{step}").mkdir(parents=True)
    assert hf_sync.local_steps(tmp_path, "sft") == [5, 7, 1000]
    assert hf_sync.remote_files_for(files, "sft", 100) == ["sft/checkpoint-100/trainer_state.json"]


def _cache(tmp_path, name):
    """Stand in for hf_hub_download: a cached file the puller copies into the Trainer checkpoint."""
    p = tmp_path / "cache" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": 100}), encoding="utf-8")
    return str(p)


def test_pull_downloads_only_a_newer_checkpoint_and_only_state_files(tmp_path):
    files = ["sft/checkpoint-100/trainer_state.json", "sft/checkpoint-100/optimizer.pt",
             "sft/checkpoint-100/adapter_model.safetensors", "sft/checkpoint-100/tokenizer.json",
             "sft/checkpoint-50/trainer_state.json"]
    seen = []
    out = hf_sync.pull_checkpoint(tmp_path, "r", "sft", TOKEN, files=files,
                                  download=lambda f: seen.append(f) or _cache(tmp_path, pathlib.PurePosixPath(f).name))
    assert pathlib.Path(out).name == "checkpoint-100"
    assert sorted(pathlib.PurePosixPath(f).name for f in seen) == [
        "adapter_model.safetensors", "optimizer.pt", "trainer_state.json"]
    assert json.loads((tmp_path / "sft" / "checkpoint-100" / "trainer_state.json").read_text()) == {"step": 100}
    # a local checkpoint at the same or a higher step means nothing to download
    assert hf_sync.pull_checkpoint(tmp_path, "r", "sft", TOKEN, files=files,
                                   download=lambda f: pytest.fail("must not re-download a local checkpoint")) is None


def test_push_checkpoint_uploads_the_whole_resume_state(tmp_path):
    ckpt = tmp_path / "runs" / "sft" / "checkpoint-200"
    ckpt.mkdir(parents=True)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth",
                 "adapter_model.safetensors", "tokenizer_config.json"):
        (ckpt / name).write_text("{}", encoding="utf-8")
    (ckpt.parent / "trainer_state.json").write_text("{}", encoding="utf-8")
    api = FakeApi()
    prefix = hf_sync.push_checkpoint(ckpt, "r", "sft", TOKEN, api)
    assert prefix == "sft/checkpoint-200"
    folder = [kw for kind, kw in api.uploads if kind == "folder"][0]
    assert folder["path_in_repo"] == prefix
    assert set(folder["allow_patterns"]) == set(hf_sync.UPLOAD_PATTERNS)
    assert any(kw["path_in_repo"] == "sft/trainer_state.json" for kind, kw in api.uploads if kind == "file")


def test_merged_model_round_trip(tmp_path):
    api = FakeApi()
    src = tmp_path / "merged_src"; src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"weights")
    assert hf_sync.push_merged(src, "r", TOKEN, api, prefix="merged/sft") == "merged/sft"
    files = ["merged/sft/config.json", "merged/sft/model.safetensors", "merged/sft/tokenizer.json"]
    target = tmp_path / "sft_merged"
    def download(f):
        name = pathlib.PurePosixPath(f).name
        return str(src / name) if (src / name).exists() else _cache(tmp_path, name)
    out = hf_sync.pull_merged(target, "r", TOKEN, api, files=files, download=download)
    assert pathlib.Path(out) == target
    assert (target / "config.json").exists() and (target / "model.safetensors").exists()


def test_callback_pushes_on_save_only_on_rank_zero(tmp_path, monkeypatch):
    api = FakeApi()
    cb = hf_sync.make_callback("r", "sft", TOKEN, api)
    ckpt = tmp_path / "checkpoint-200"; ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text("{}", encoding="utf-8")
    args, state = types.SimpleNamespace(output_dir=str(tmp_path)), types.SimpleNamespace(global_step=200)
    monkeypatch.setattr(ddp, "local_rank_from_env", lambda env=None: 0)
    cb.on_save(args, state, None)
    assert any(kw.get("path_in_repo") == "sft/checkpoint-200" for _, kw in api.uploads)
    api.uploads.clear()
    monkeypatch.setattr(ddp, "local_rank_from_env", lambda env=None: 1)
    cb.on_save(args, state, None)
    assert api.uploads == []


def test_barrier_is_a_no_op_without_distributed(monkeypatch):
    import sys
    stub = types.ModuleType("torch.distributed")
    stub.is_available = lambda: False
    stub.is_initialized = lambda: False
    monkeypatch.setitem(sys.modules, "torch.distributed", stub)
    ddp.barrier()

def test_prepare_auto_skips_without_a_token_but_on_requires_one(tmp_path, monkeypatch):
    def no_token(*a, **k):
        raise SystemExit("no token")
    monkeypatch.setattr(hf_sync, "read_token", no_token)
    assert hf_sync.prepare(None, "sft", tmp_path, "auto") == (None, None)
    with pytest.raises(SystemExit):
        hf_sync.prepare(None, "sft", tmp_path, "on")
    assert hf_sync.prepare(None, "sft", tmp_path, "off") == (None, None)
