# Kaggle without a browser: CLI workflow actually used

`KAGGLE_SKILL.md` describes the Benchmarks surface; this file records the concrete commands that worked on this
account and the mistakes hit along the way. Credentials come from `KAGGLE_API_TOKEN` in `.env`; never print it.

## Credentials and quota

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
