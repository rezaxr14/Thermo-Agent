"""Kaggle training notebook source. Build with `python kaggle/make_notebook.py`.

The notebook is deliberately split into one stage per cell: a 12 h Kaggle session cannot finish an SFT epoch, and
each cell can be re-run in a new session. All state is synced to the private HF repo after every Trainer save, so
re-running a cell resumes from the newest `checkpoint-N` instead of starting over. Repo and corpus arrive as two
Kaggle datasets; the HF token comes from Kaggle Secrets, never from the notebook source.
"""

# %% [markdown]
# # thermo-agent: dual-T4 training with resumable checkpoints
#
# Attach two private Kaggle datasets before running:
#
# 1. `thermo-agent-code` — the zip made by `scripts/make_kaggle_bundle.ps1`.
# 2. `thermo-agent-corpus` — the `data/sft` and `data/grpo` directories from the local build.
#
# Add a Kaggle Secret named `HF_TOKEN` (the same token as `.env` locally). Set **Accelerator: GPU T4 x2** in the
# notebook settings. Run the setup cells once, then run one training cell per session; a re-run resumes from the
# last checkpoint in `Reza-Nadimi/thermo-agent-checkpoints`.

# %%
# Bootstrap with the standard library only: the code bundle is not on sys.path until it is unpacked.
import os
import pathlib
import shutil
import sys
import zipfile

# T4

# 14.56 GiB with a 4B fp16 base leaves little headroom. Expandable segments remove the
# reserved-but-unallocated fragmentation that made smoke v26 fail on the first backward.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "thermo-agent"
code_root = None
bundles = []
for _root, _dirs, _files in os.walk("/kaggle/input", followlinks=True):
    if "thermo_agent" in _dirs and (pathlib.Path(_root) / "thermo_agent" / "train" / "sft.py").exists():
        code_root = pathlib.Path(_root)
        break
    if "kaggle_bundle.zip" in _files:
        bundles.append(pathlib.Path(_root) / "kaggle_bundle.zip")
if code_root:
    shutil.copytree(code_root, ROOT, dirs_exist_ok=True)
elif bundles:
    with zipfile.ZipFile(bundles[0]) as zf:
        zf.extractall(WORK)
else:
    raise SystemExit("attach the Kaggle dataset thermo-agent-code")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("code unpacked at", ROOT)

# %%
from thermo_agent.kaggle_notebook import (
    WORK, assert_pinned_environment, find_input, restore_checkpoints, run, setup, status, token,
    torch_device_count, torch_facts,
)

# Read the corpus selectors BEFORE setup() so a custom SFT_DATA/GRPO_DATA is always staged. The
# merged corpora (sft_all / grpo_all) are the defaults; the original sft / grpo stay available too.
SFT_DATA = os.environ.get("SFT_DATA", "sft_all")
GRPO_DATA = os.environ.get("GRPO_DATA", "grpo_all")
ROOT = setup(corpora=tuple(dict.fromkeys(
    (SFT_DATA, GRPO_DATA, "sft", "grpo", "sft_all", "grpo_all"))))  # install deps + stage corpora

# Pin check in the notebook itself: every version comes from kaggle/environment.lock.json and
# the fused wheels come from kernel_sources. A drift fails here, it is never repaired by compiling.
assert_pinned_environment(ROOT / "kaggle" / "environment.lock.json")
status(stage="setup", state="done", root=str(ROOT))
print("code and corpora ready at", ROOT)

# %%
def require_train_corpus(name, env_var):
    """Fail before DDP if the requested corpus dir is missing or has no *_train.jsonl rows.

    Naming the dir, the files actually present and the env var to set turns the v25 failure
    ("no training rows for [...] in .../data/sft_all") into an immediate, actionable error.
    """
    data_root = ROOT / "data"
    d = data_root / name
    inside = sorted(p.name for p in d.glob("*_train.jsonl")) if d.is_dir() else []
    if not inside:
        present = sorted(p.name for p in data_root.iterdir()) if data_root.is_dir() else []
        existing = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
        raise SystemExit(
            f"[corpus] no *_train.jsonl rows for corpus {name!r} at {d}. "
            f"Files actually present in that dir: {existing}. "
            f"Corpora staged under {data_root}: {present}. "
            f"Set {env_var} to a staged corpus directory name before running this cell."
        )
    print(f"[corpus] {name}: {len(inside)} train split(s): {inside}", flush=True)
    return d


# %%
# Fail fast on a bad env var before any DDP cell is launched.
require_train_corpus(SFT_DATA, "SFT_DATA")
require_train_corpus(GRPO_DATA, "GRPO_DATA")

# %%
# Cross-session resume: with no Kaggle Secret, HF sync cannot relay checkpoints. The local CLI
# relays the newest checkpoint into the private thermo-agent-resume dataset between sessions.
restored = restore_checkpoints(ROOT)
status(stage="resume", state="done", restored=restored)

# %%
# The token is a Kaggle Secret, not a literal in the notebook. `hf_sync` refuses any repo other than
# the single project repo and refuses a public repo.
token()
from thermo_agent.train import ddp, hf_sync

if os.environ.get("HF_TOKEN"):
    print("HF repo:", hf_sync.ensure_private_repo())
else:
    print("no token: --hf-sync auto will keep checkpoints local")

# %%
print(ddp.topology_line())
# Torch facts come from a short subprocess with the pinned PYTHONPATH; the notebook (parent) process
# must never pull torch or numpy into itself after setup(), or it would reload the corrupt preloaded wheel.
_tf = torch_facts()
status(stage="preflight", state="running", torch=_tf["torch"], cuda=_tf["cuda"])
assert torch_device_count() == 2, "start this notebook with Accelerator: GPU T4 x2 (two T4s visible)"
assert ddp.dtype_for_capability(7, 5) == "float16", "T4 must run fp16"
status(stage="preflight", state="done", gpus=torch_device_count())

# %%
# Optional smoke limits: set STEPS=20 and ROWS=64 to validate the whole path before the long run.
STEPS = int(os.environ.get("STEPS", "0"))
ROWS = int(os.environ.get("ROWS", "0"))
SYNC = os.environ.get("THERMO_HF_SYNC", "auto")
LIMITS = (["--max-steps", str(STEPS)] if STEPS else []) + (["--limit-rows", str(ROWS)] if ROWS else [])
PY = sys.executable
NGPU = int(os.environ.get("NGPU", "2"))
# SFT_DATA / GRPO_DATA are read in the setup cell above (before setup() stages the corpora).
SFT_BATCH = os.environ.get("SFT_BATCH", "2")
SFT_ACCUM = os.environ.get("SFT_ACCUM", "8")
MAXLEN = os.environ.get("MAXLEN", "")
LOSS_TOKENS = os.environ.get("LOSS_TOKENS", "512")
LOSS_VOCAB = os.environ.get("LOSS_VOCAB", "8192")
print("[loss] tiles", LOSS_TOKENS, "tokens x", LOSS_VOCAB, "vocabulary entries", flush=True)

# %%
# Stage 1: SFT. Every save pushes the adapter, optimizer, scheduler, scaler and RNG to the private repo.
# Re-check here so a bad env var fails before launching DDP even if setup ran in an earlier cell.
require_train_corpus(SFT_DATA, "SFT_DATA")
status(stage="sft", state="running", steps=STEPS, rows=ROWS, batch=SFT_BATCH, accum=SFT_ACCUM,
       loss_tokens=LOSS_TOKENS, loss_vocab=LOSS_VOCAB)
run([PY, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(NGPU), "--module", "thermo_agent.train.sft",
     ROOT / "data" / SFT_DATA, ROOT / "runs" / "sft", "--dtype", "float16",
     "--tasks", "repair", "design", "audit", "ledger",
     "--per-device-batch", SFT_BATCH, "--grad-accum", SFT_ACCUM, "--num-workers", "2",
     "--loss-token-chunk", LOSS_TOKENS, "--loss-vocab-chunk", LOSS_VOCAB,
     *(["--max-length", MAXLEN] if MAXLEN else []),
     "--hf-sync", SYNC, *LIMITS])
status(stage="sft", state="done", local_steps=hf_sync.local_steps(ROOT / "runs", "sft"))

# %%
# Merge the newest SFT checkpoint; the merged weights also go to the repo so the next session can start GRPO.
latest = hf_sync.local_steps(ROOT / "runs", "sft")
if not latest:
    raise SystemExit("no SFT checkpoint to merge")
adapter = ROOT / "runs" / "sft" / f"checkpoint-{max(latest)}"
merged = ROOT / "runs" / "sft_merged"
status(stage="merge", state="running", adapter=str(adapter))
run([PY, "-m", "thermo_agent.train.merge", adapter, merged,
     "--dtype", "float16", "--hf-sync", SYNC])
status(stage="merge", state="done")

# %%
# Stage 2: GRPO. `--init runs/sft_merged` reuses the merged model from the merge cell, so no SFT is redone here.
# Each task x k has its own checkpoint prefix (`grpo_ledger_k4`, ...), and can run in a separate session.
TASK = os.environ.get("TASK", "ledger")
# GRPO group size k (rollouts per prompt). Each k gets its OWN output dir so checkpoints never
# overwrite: k=4 -> runs/grpo_{TASK}_k4, k=8 -> ..._k8, k=16 -> ..._k16. SFT alone is runs/sft.
GRPO_K = int(os.environ.get("GRPO_K", "4"))
# ddp.assert_grpo_batches requires per_device_batch % k == 0 (and the effective batch too).
# Default to the smallest value that also keeps the >=4 throughput floor, so k=4/8/16 all start
# without extra env vars; GRPO_BATCH/GRPO_ACCUM remain overridable for a custom topology.
GRPO_BATCH = int(os.environ.get("GRPO_BATCH", str(max(4, GRPO_K))))
GRPO_ACCUM = int(os.environ.get("GRPO_ACCUM", "2"))
GRPO_STAGE = f"grpo_{TASK}_k{GRPO_K}"
print(f"[grpo] k={GRPO_K} per-device-batch={GRPO_BATCH} grad-accum={GRPO_ACCUM}", flush=True)
status(stage=GRPO_STAGE, state="running", k=GRPO_K, batch=GRPO_BATCH, accum=GRPO_ACCUM)
# GRPO must start from the merged SFT policy, never from base Qwen. The merge cell writes these
# weights to runs/sft_merged (and also pushes them to HF); pass that local dir as --init in both
# the token and no-token paths so the checkpoint and the eval arms share the same base model.
INIT = str(merged)
# Same pre-DDP guard for GRPO: a bad GRPO_DATA must fail before the second DDP launch too.
require_train_corpus(GRPO_DATA, "GRPO_DATA")
run([PY, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(NGPU), "--module", "thermo_agent.train.grpo",
     ROOT / "data" / GRPO_DATA, ROOT / "runs" / GRPO_STAGE, "--task", TASK,
     "--init", INIT, "--dtype", "float16", "--num-generations", str(GRPO_K),
     "--per-device-batch", str(GRPO_BATCH), "--grad-accum", str(GRPO_ACCUM), "--num-workers", "2",
     "--rollout-log", ROOT / "runs" / GRPO_STAGE / "rollouts.jsonl",
     "--hf-sync", SYNC, *LIMITS])
status(stage=GRPO_STAGE, state="done", k=GRPO_K, local_steps=hf_sync.local_steps(ROOT / "runs", GRPO_STAGE))

# %%
# Status: the remote steps are what the next session will resume from.
if os.environ.get("HF_TOKEN"):
    for stage in ["sft", f"grpo_{TASK}_k{GRPO_K}"]:
        print(stage, "local:", hf_sync.local_steps(ROOT / "runs", stage),
              "remote:", hf_sync.remote_steps(hf_sync.repo_files(hf_sync.resolve_repo()), stage))
else:
    print("remote steps: n/a (no HF token in this worker)")
