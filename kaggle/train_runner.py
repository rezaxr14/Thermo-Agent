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
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "thermo-agent"

# %%
def _input_dir(name: str) -> pathlib.Path:
    candidates = sorted(p for p in pathlib.Path("/kaggle/input").glob(f"**/{name}") if p.is_dir())
    if not candidates:
        raise SystemExit(f"attach the Kaggle dataset {name!r} to this notebook")
    return candidates[0]


code_input = _input_dir("thermo-agent-code")
zip_path = next(code_input.rglob("kaggle_bundle.zip"), None)
if zip_path:
    shutil.rmtree(ROOT, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(WORK)
if not (ROOT / "thermo_agent" / "train" / "sft.py").exists():
    raise SystemExit(f"no code at {ROOT}: attach the kaggle_bundle.zip dataset")

corpus_input = _input_dir("thermo-agent-corpus")
for name in ("sft", "grpo"):
    src = next(corpus_input.rglob(name), None)
    if src is None or not src.is_dir():
        raise SystemExit(f"corpus {name} missing under {corpus_input}")
    dst = ROOT / "data" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copytree(src, dst)
print(f"code: {ROOT}; corpus: {[str(ROOT / 'data' / n) for n in ('sft', 'grpo')]}")

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# %%
# The token is a Kaggle Secret, not a literal in the notebook. `hf_sync` refuses any repo other than
# <token owner>/thermo-agent-checkpoints and refuses a public repo.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    print("no Kaggle secret HF_TOKEN; training will stay local:", exc)

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


def run(cmd, **env):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True,
                   env={**os.environ, **{k: str(v) for k, v in env.items()}})


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
     "--hf-sync", SYNC, *LIMITS])

# %%
# Status: the remote steps are what the next session will resume from.
for stage in ["sft", f"grpo_{TASK}"]:
    print(stage, "local:", hf_sync.local_steps(ROOT / "runs", stage),
          "remote:", hf_sync.remote_steps(hf_sync.repo_files(hf_sync.resolve_repo()), stage))
