"""Resume training across Kaggle sessions with one private Hugging Face repo.

A 12 h Kaggle session cannot finish a 30 h SFT epoch, so the full Trainer state -- LoRA weights, optimizer,
scheduler, GradScaler, RNG and trainer_state -- is pushed to a *single* private repo after every save. A restarted
session pulls the highest checkpoint back before the Trainer is built, so `train(resume_from_checkpoint=latest(...))`
continues exactly where the previous session stopped. The merged model is pushed under `merged/` so the next
session can start GRPO without redoing the merge.

Safety: this module never guesses a repo. The repo id must equal `<token owner>/thermo-agent-checkpoints`; a
different id (including an organisation or another user's repo) is refused before any network call. An existing
repo that is not private is also refused. The token is read from the environment or the repository's `.env`,
never from a notebook cell.

    python -m thermo_agent.train.hf_sync init              # create the private repo once
    python -m thermo_agent.train.hf_sync status             # local vs remote steps
    python -m thermo_agent.train.hf_sync push --stage sft   # manual push (also done by the callback)
    python -m thermo_agent.train.hf_sync pull --stage sft   # manual pull

Training enables this automatically when a token is available (`--hf-sync auto`, the default); pass
`--hf-sync off` for a purely local run.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import time

DEFAULT_REPO_NAME = "thermo-agent-checkpoints"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

# The complete state of a resumable Trainer run. Nothing outside this list is uploaded: no data, no .env, no logs.
STATE_FILES = (
    "trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth",
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin",
)
UPLOAD_PATTERNS = ["*.json", "*.safetensors", "*.bin", "*.pt", "*.pth"]

# The xet transfer backend is an optional huggingface_hub extra; plain HTTP works everywhere and is easier to
# retry on a throttled Kaggle connection.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# --------------------------------------------------------------------------------------------- credentials
def read_token(env_path=None) -> str:
    """HF_TOKEN from the environment, else from a `KEY=value` line in `.env` (no dependency on python-dotenv)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    path = pathlib.Path(env_path or ".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "HF_TOKEN":
                token = value.strip().strip('"').strip("'")
                if token:
                    return token
    raise SystemExit("HF_TOKEN is not set: put it in .env or the Kaggle secret store (THERMO_HF_SYNC=off disables sync)")


def owner_of(token: str | None = None, api=None) -> str:
    api = api or _api(token)
    return api.whoami()["name"]


def _api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token or read_token())


def default_repo(token: str | None = None, api=None) -> str:
    return f"{owner_of(token, api)}/{DEFAULT_REPO_NAME}"


def resolve_repo(repo_id: str | None = None, token: str | None = None, api=None) -> str:
    """Accept only this project's private repo, never an arbitrary id from an env var or a notebook cell."""
    repo = repo_id or os.environ.get("HF_REPO_ID") or default_repo(token, api)
    expected = default_repo(token, api)
    if repo != expected:
        raise ValueError(f"refusing HF repo {repo!r}: this project may only use {expected!r}")
    return repo


# --------------------------------------------------------------------------------------------- repo lifecycle
def ensure_private_repo(repo_id: str | None = None, token: str | None = None, api=None) -> str:
    """Create the project repo as private if missing; refuse it if it already exists and is public."""
    token = token or read_token()
    api = api or _api(token)
    repo = resolve_repo(repo_id, token, api)
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = api.repo_info(repo_id=repo, token=token)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo, token=token, private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo, token=token)
    if not getattr(info, "private", False):
        raise SystemExit(f"refusing to upload: {repo} is not private")
    return repo


def repo_files(repo_id: str, token: str | None = None, api=None) -> list[str]:
    api = api or _api(token)
    return list(api.list_repo_files(repo_id=repo_id, token=token))


# --------------------------------------------------------------------------------------------- pure helpers
def remote_steps(files, stage: str) -> list[int]:
    prefix = f"{stage}/checkpoint-"
    return sorted({int(m.group(1)) for f in files
                   if (m := re.match(re.escape(prefix) + r"(\d+)/", str(f)))})


def local_steps(root, stage: str) -> list[int]:
    path = pathlib.Path(root) / stage
    if not path.is_dir():
        return []
    return sorted({int(m.group(1)) for p in path.iterdir()
                   if (m := CHECKPOINT_RE.match(p.name)) and p.is_dir()})


def remote_files_for(files, stage: str, step: int) -> list[str]:
    prefix = f"{stage}/checkpoint-{step}/"
    return [f for f in files if str(f).startswith(prefix) and not str(f).endswith("/")]


# --------------------------------------------------------------------------------------------- upload/download
def _retry(fn, what: str, attempts: int = 3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)


def push_checkpoint(checkpoint_dir, repo_id: str, stage: str, token: str | None = None, api=None) -> str:
    """Upload one Trainer checkpoint (state + weights) under `<stage>/checkpoint-N/`."""
    src = pathlib.Path(checkpoint_dir)
    match = CHECKPOINT_RE.match(src.name)
    if not match or not src.is_dir():
        raise ValueError(f"not a Trainer checkpoint directory: {src}")
    api = api or _api(token)
    step = int(match.group(1))
    prefix = f"{stage}/{src.name}"
    _retry(lambda: api.upload_folder(repo_id=repo_id, token=token, folder_path=str(src), path_in_repo=prefix,
                                     allow_patterns=UPLOAD_PATTERNS,
                                     commit_message=f"{stage}: checkpoint-{step} (full resume state)"),
           f"upload {prefix}")
    # The stage root carries trainer_state.json so `status` works without a checkpoint download.
    for name in ("trainer_state.json",):
        f = src.parent / name
        if f.exists():
            _retry(lambda: api.upload_file(repo_id=repo_id, token=token, path_or_fileobj=str(f),
                                           path_in_repo=f"{stage}/{name}",
                                           commit_message=f"{stage}: training state at step {step}"),
                   f"upload {stage}/{name}")
    return prefix


def push_root_state(root, repo_id: str, stage: str, token: str | None = None, api=None) -> list[str]:
    """Upload small run metadata (settings, git commit, last step) for a stage."""
    src = pathlib.Path(root) / stage / "run.json"
    if not src.exists():
        return []
    api = api or _api(token)
    _retry(lambda: api.upload_file(repo_id=repo_id, token=token, path_or_fileobj=str(src),
                                   path_in_repo=f"{stage}/run.json",
                                   commit_message=f"{stage}: run metadata"), f"upload {stage}/run.json")
    return [f"{stage}/run.json"]


def pull_checkpoint(root, repo_id: str, stage: str, token: str | None = None, api=None,
                    files=None, download=None) -> str | None:
    """Download the newest remote checkpoint if it is newer than the newest local one.

    `files` and `download` are injectable for offline tests; by default they use huggingface_hub. Returns the
    local checkpoint path, or None when nothing needed downloading.
    """
    from huggingface_hub import hf_hub_download

    token = token or read_token()
    files = list(files if files is not None else _retry(lambda: repo_files(repo_id, token, api), "list files"))
    remote = remote_steps(files, stage)
    local = local_steps(root, stage)
    if not remote or (local and max(remote) <= max(local)):
        return None
    step = max(remote)
    items = [f for f in remote_files_for(files, stage, step) if pathlib.PurePosixPath(f).name in STATE_FILES]
    if not items:
        raise SystemExit(f"remote checkpoint {stage}/checkpoint-{step} has no resumable state files")
    target = pathlib.Path(root) / stage / f"checkpoint-{step}"
    tmp = target.parent / f".hf_download_{step}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    download = download or (lambda f: hf_hub_download(repo_id=repo_id, filename=f, token=token, local_dir=str(tmp),
                                                      force_download=True))
    try:
        for f in items:
            path = pathlib.Path(_retry(lambda f=f: download(f), f"download {f}"))
            dest = target / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not (target / "trainer_state.json").exists():
        raise SystemExit(f"checkpoint {target} has no trainer_state.json; refusing an unverified resume")
    return str(target)


def push_merged(local_dir, repo_id: str, token: str | None = None, api=None, prefix: str = "merged/sft") -> str:
    """Upload a merged model so GRPO can start in a new session without repeating SFT + merge."""
    api = api or _api(token)
    _retry(lambda: api.upload_folder(repo_id=repo_id, token=token, folder_path=str(local_dir), path_in_repo=prefix,
                                     allow_patterns=UPLOAD_PATTERNS,
                                     commit_message="merged SFT model for GRPO"), "upload merged model")
    return prefix


def pull_merged(local_dir, repo_id: str, token: str | None = None, api=None,
                prefix: str = "merged/sft", files=None, download=None) -> str | None:
    """Download a merged model if the local directory is missing or incomplete."""
    from huggingface_hub import hf_hub_download

    token = token or read_token()
    target = pathlib.Path(local_dir)
    needed = ("config.json", "model.safetensors")
    if all((target / n).exists() for n in needed):
        return str(target)
    files = list(files if files is not None else _retry(lambda: repo_files(repo_id, token, api), "list files"))
    items = [f for f in files if str(f).startswith(prefix + "/") and not str(f).endswith("/")]
    if not any(str(f).endswith("config.json") for f in items):
        return None
    target.mkdir(parents=True, exist_ok=True)
    download = download or (lambda f: hf_hub_download(repo_id=repo_id, filename=f, token=token,
                                                      local_dir=str(target), force_download=True))
    for f in items:
        path = pathlib.Path(_retry(lambda f=f: download(f), f"download {f}"))
        rel = pathlib.PurePosixPath(f).relative_to(prefix)
        dest = target / str(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != dest.resolve():
            shutil.copyfile(path, dest)
    return str(target) if all((target / n).exists() for n in needed) else None


# --------------------------------------------------------------------------------------------- Trainer hook
def make_callback(repo_id: str, stage: str, token: str | None = None, api=None):
    """A TrainerCallback that pushes after each save and at the end of training (rank 0 only)."""
    from transformers import TrainerCallback
    from . import ddp  # imported lazily: the sync module itself must stay importable on CPU-only boxes

    api = api or _api(token)

    class _HFSyncCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if ddp.local_rank_from_env() != 0:
                return
            checkpoint = pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint.is_dir():
                print(f"[hf_sync] pushing {stage}/{checkpoint.name}", flush=True)
                push_checkpoint(checkpoint, repo_id, stage, token, api)

        def on_train_end(self, args, state, control, **kwargs):
            if ddp.local_rank_from_env() != 0:
                return
            meta = {"stage": stage, "global_step": state.global_step, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            f = pathlib.Path(args.output_dir) / "run.json"
            f.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
            push_root_state(args.output_dir, repo_id, stage, token, api)
            print(f"[hf_sync] {stage} synced at step {state.global_step}", flush=True)

    return _HFSyncCallback()


def prepare(repo_id: str | None, stage: str, root, mode: str = "auto", token: str | None = None, api=None):
    """Resolve the repo, pull the newest checkpoint (rank 0), and return (repo_id, callback).

    `mode`: `off` -> no sync, `on` -> require a token, `auto` -> sync only when a token is available.
    """
    from . import ddp

    if mode == "off":
        return None, None
    try:
        token = token or read_token()
    except SystemExit:
        if mode == "auto":
            print("[hf_sync] no HF token; training stays local (--hf-sync on to require one)", flush=True)
            return None, None
        raise
    # A bad repo id or a public repo is a real error, not a reason to fall back to local-only training.
    resolved = ensure_private_repo(repo_id, token, api)
    if ddp.local_rank_from_env() == 0:
        pulled = pull_checkpoint(root, resolved, stage, token, api)
        print(f"[hf_sync] {stage}: restored from {pulled}" if pulled else f"[hf_sync] {stage}: no remote checkpoint yet", flush=True)
    if ddp.is_distributed():
        ddp.barrier()
    return resolved, make_callback(resolved, stage, token, api)


# --------------------------------------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("init", "status", "push", "pull", "sync", "push-merged", "pull-merged"))
    ap.add_argument("--stage", default="sft", help="sft, grpo_repair, ... (not used by the merged actions)")
    ap.add_argument("--dir", default="runs", help="local root that holds <stage>/checkpoint-N")
    ap.add_argument("--repo", default=None, help="must be <token owner>/thermo-agent-checkpoints")
    ap.add_argument("--prefix", default="merged/sft", help="path of the merged model inside the repo")
    a = ap.parse_args()
    token = read_token()
    repo = resolve_repo(a.repo, token)
    if a.action == "init":
        print(ensure_private_repo(repo, token))
    elif a.action == "status":
        inform = lambda: repo_files(repo, token)
        files = inform()
        print(json.dumps({"repo": repo, "private": True, "stage": a.stage,
                          "remote": remote_steps(files, a.stage), "local": local_steps(a.dir, a.stage)}, indent=1))
    elif a.action in ("push", "sync"):
        repo = ensure_private_repo(repo, token)
        steps = local_steps(a.dir, a.stage)
        if not steps:
            raise SystemExit(f"no local checkpoints under {a.dir}/{a.stage}")
        for step in steps:
            push_checkpoint(pathlib.Path(a.dir) / a.stage / f"checkpoint-{step}", repo, a.stage, token)
        push_root_state(a.dir, repo, a.stage, token)
        print(f"pushed {a.dir}/{a.stage} steps {steps} to {repo}")
    elif a.action == "pull":
        print(pull_checkpoint(a.dir, repo, a.stage, token) or "nothing to pull")
    elif a.action == "push-merged":
        repo = ensure_private_repo(repo, token)
        print(push_merged(a.dir, repo, token, prefix=a.prefix))
    elif a.action == "pull-merged":
        print(pull_merged(a.dir, repo, token, prefix=a.prefix) or "no merged model in the repo")


if __name__ == "__main__":
    main()
