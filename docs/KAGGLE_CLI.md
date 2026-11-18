# Kaggle without a browser: CLI workflow actually used

`KAGGLE_SKILL.md` describes the Benchmarks surface; this file records the concrete commands that worked on this
account and the mistakes hit along the way. Credentials come from `KAGGLE_API_TOKEN` in `.env`; never print it.

## Credentials and quota

All Kaggle CLI/API traffic from this machine must go through the local SOCKS proxy:

```powershell
$env:HTTP_PROXY = 'socks5h://localhost:10808'
$env:HTTPS_PROXY = 'socks5h://localhost:10808'
# PySocks is installed in .venv; without it, requests raises "Missing dependencies for SOCKS support"
```

```powershell
$env:KAGGLE_API_TOKEN = (& .venv/Scripts/python.exe -c "from pathlib import Path; print([l.split('=',1)[1].strip() for l in Path('.env').read_text().splitlines() if l.startswith('KAGGLE_API_TOKEN=')][0])")
.venv/Scripts/kaggle.exe benchmarks quota --format json
.venv/Scripts/kaggle.exe benchmarks init -y --env-file data/kaggle_proxy.env
```

The proxy key lasts about an hour; `thermo_agent.teacher.cli_runner` refreshes it automatically.

## Teacher pass (hosted model, local driver)

```powershell
python -m thermo_agent.teacher.cli_runner probe
python -m thermo_agent.teacher.cli_runner run --model google/gemini-3.1-flash-lite-preview \
    --out data/teacher_runs/completions.jsonl --wait
python -m thermo_agent.teacher.transfer ingest data/frontier_export data/teacher_runs/completions.jsonl \
    data/frontier --memory-dir data/memory --model google/gemini-3.1-flash-lite-preview
```

The runner journals every success and error with token cost (nanodollars from Kaggle's `usage`), checks quota before
each batch, pauses with `--wait`, and mirrors the journal to `teacher/progress.jsonl` in the private HF repo. To
evaluate frontier arms on the *test* rows, first run `scripts/make_eval_export.py`, which exports exactly the rows
`model_eval` samples; answering an export built from the training split produces ids replay cannot find.

## Datasets and kernels

```powershell
pwsh scripts/upload_corpus_dataset.ps1
pwsh -File scripts/make_kaggle_bundle.ps1
# dataset slugs used by the kernel:
#   rezanadimi/thermo-agent-code, rezanadimi/thermo-agent-corpus, rezanadimi/thermo-teacher-prompts
#   rezanadimi/thermo-kernel-wheels (prebuilt fused kernels),
#   rezanadimi/thermo-agent-resume (LoRA resume state),
#   rezanadimi/thermo-agent-eval-replays (saved teacher answers)
$kdir = (Resolve-Path 'kaggle/train_kernel_smoke').Path
.venv/Scripts/kaggle.exe kernels push -p "$kdir" --accelerator NvidiaTeslaT4
.venv/Scripts/kaggle.exe kernels status rezanadimi/thermo-train-smoke
.venv/Scripts/kaggle.exe kernels logs rezanadimi/thermo-train-smoke
```

Gotchas that cost time:

- The metadata file is `dataset-metadata.json` (singular `dataset`), and it needs a `licenses` array.
- `kaggle datasets create` fails on relative `-p` paths on Windows; pass an absolute path.
- `Resolve-Path` returns a `PathInfo` that stringifies to a path but has no `.FullName`; cast to `[string]`.
- `kernels push` **runs** the kernel; use `--no-run` to only save a version.
- The GPU accelerator is selected on the push command (`--accelerator NvidiaTeslaT4`), not in metadata.
- Kernel title determines the slug; keep it identical to the id's last component.
- `PYTHONIOENCODING=utf-8` is required locally when the `kaggle_benchmarks` SDK prints emoji.
- Kaggle Secrets are unavailable in hosted benchmark tasks; use the shared HF repo for credentials and progress.

## Fused kernels: build once, never recompile

Qwen3.5-4B's linear attention uses the slow PyTorch reference kernels unless `fla` and `causal-conv1d` are
installed. Without them a T4 eval forward OOMs at 4096 tokens and a training step takes ~50 s. The exact pinned
environment is `kaggle/environment.lock.json`.

- Build the wheels **once** by pushing `kaggle/wheel_kernel/` (`rezanadimi/thermo-kernel-wheels`, source
  `kaggle/wheel_runner.py`) on a T4. It compiles `causal-conv1d` against Kaggle's own torch and verifies that
  CUDA torch survives.
- **Kaggle itself transfers the wheels**: the build kernel's output is attached to the train, smoke and eval
  kernels through `kernel_sources: ["rezanadimi/thermo-kernel-wheels"]` in their `kernel-metadata.json`. No local
  download or dataset upload is involved.
- `kaggle_notebook.setup()` locates `/kaggle/input/thermo-kernel-wheels/wheels/*.whl`, checks its `manifest.json`
  against the worker's torch/CUDA version, installs with `pip --no-deps`, and imports the kernels. A mismatch or
  missing wheel **fails loudly** — there is no source-build fallback in any worker. The only rebuild trigger is a
  change to Kaggle's torch/CUDA/Python image.

## Cross-session resume without the HF token

Kaggle's Secrets service was unreachable in the training worker, so the token cannot be relied on. Between
sessions, relay the newest checkpoint from the kernel output into a private dataset (no credentials in it):

```powershell
pwsh scripts/resume_kaggle_checkpoint.ps1 -Stage sft -Kernel rezanadimi/thermo-train-run
# ... then start the next session with `kernels push` on kaggle/train_kernel/
```

`restore_checkpoints()` in `thermo_agent/kaggle_notebook.py` copies the newest *complete* checkpoint into
`runs/<stage>/` at the start of a session; the trainer's existing `latest()` then resumes normally. GRPO uses
the locally merged SFT weights when no token is present. No HF token is copied into any notebook, bundle or
dataset.
