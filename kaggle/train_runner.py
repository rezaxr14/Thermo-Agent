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

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "thermo-agent"
bundles = sorted(pathlib.Path("/kaggle/input").rglob("kaggle_bundle.zip"))
if not bundles:
    raise SystemExit("attach the Kaggle dataset thermo-agent-code")
shutil.rmtree(ROOT, ignore_errors=True)
with zipfile.ZipFile(bundles[0]) as zf:
    zf.extractall(WORK)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("code unpacked at", ROOT)

# %%
from thermo_agent.kaggle_notebook import WORK, find_input, run, setup, token

ROOT = setup()   # install dependencies and copy the corpora
print("code and corpora ready at", ROOT)

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
assert ddp.world_size_from_env() == 2, "start this notebook with Accelerator: GPU T4 x2"
assert ddp.dtype_for_capability(7, 5) == "float16", "T4 must run fp16"

# %%
# Optional smoke limits: set STEPS=20 and ROWS=64 to validate the whole path before the long run.
STEPS = int(os.environ.get("STEPS", "0"))
ROWS = int(os.environ.get("ROWS", "0"))
SYNC = os.environ.get("THERMO_HF_SYNC", "auto")
LIMITS = (["--max-steps", str(STEPS)] if STEPS else []) + (["--limit-rows", str(ROWS)] if ROWS else [])
PY = sys.executable

# %%
# Stage 1: SFT. Every save pushes the adapter, optimizer, scheduler, scaler and RNG to the private repo.
run([PY, "-m", "torchrun", "--standalone", "--nproc_per_node=2", "--module", "thermo_agent.train.sft",
     ROOT / "data" / "sft", ROOT / "runs" / "sft", "--dtype", "float16",
     "--tasks", "repair", "design", "audit", "ledger",
     "--per-device-batch", "2", "--grad-accum", "8", "--num-workers", "2",
     "--hf-sync", SYNC, *LIMITS])

# %%
# Merge the newest SFT checkpoint; the merged weights also go to the repo so the next session can start GRPO.
latest = hf_sync.local_steps(ROOT / "runs", "sft")
if not latest:
    raise SystemExit("no SFT checkpoint to merge")
adapter = ROOT / "runs" / "sft" / f"checkpoint-{max(latest)}"
merged = ROOT / "runs" / "sft_merged"
run([PY, "-m", "thermo_agent.train.merge", adapter, merged,
     "--dtype", "float16", "--hf-sync", SYNC])

# %%
# Stage 2: GRPO. `--init hf:merged/sft` reuses the merged model from the repo, so no SFT is redone here.
# Each task has its own checkpoint prefix (`grpo_repair`, ...), and can be run in a separate session.
TASK = os.environ.get("TASK", "ledger")
run([PY, "-m", "torchrun", "--standalone", "--nproc_per_node=2", "--module", "thermo_agent.train.grpo",
     ROOT / "data" / "grpo", ROOT / "runs" / f"grpo_{TASK}", "--task", TASK,
     "--init", "hf:merged/sft", "--dtype", "float16",
     "--per-device-batch", "4", "--grad-accum", "2", "--num-workers", "2",
     "--rollout-log", ROOT / "runs" / f"grpo_{TASK}" / "rollouts.jsonl",
     "--hf-sync", SYNC, *LIMITS])

# %%
# Status: the remote steps are what the next session will resume from.
for stage in ["sft", f"grpo_{TASK}"]:
    print(stage, "local:", hf_sync.local_steps(ROOT / "runs", stage),
          "remote:", hf_sync.remote_steps(hf_sync.repo_files(hf_sync.resolve_repo()), stage))
