# Generic Kaggle model runner

`kaggle_model_runner.py` is the single source of truth; `kaggle_model_runner.ipynb` is generated from it. Push the notebook once, then run any allowed model against any shard/mode without editing the task again.

## Hard rules baked in

- Only `gpt-6-luna` and `claude-haiku-5-5` are allowed by default. `gemini-3.8-flash` and `gemini-3.5-flash-lite` are **hard-blocked** even if someone adds them to a config.
- A Kaggle task push auto-runs the session default model; the runner **skips** any non-allowed model before making a model call, so no quota is spent. Always push, then explicitly `tasks run -m ...`.
- `HF_TOKEN` / `KAGGLE_API_TOKEN` are never read here — this notebook only needs the attached code/data datasets. All credentials stay in the repo-root `.env` locally.
- No local data is uploaded to Kaggle: the attached dataset is fed from the private HF corpus repo by `kaggle/hf_to_kaggle_kernel`.

## Modes

| Mode | Harness | Tools | Use |
|---|---|---|---|
| `harness` | yes | yes | multi-turn CoT tracegen |
| `eval_h` | yes | yes | harness/tool evaluation rollouts |
| `plain` | no | no | zero-shot generation |
| `eval_z` | no | no | zero-shot evaluation |
| `cohort_h` | yes | yes | tier-sharded harness tracegen |
| `cohort_z` | no | no | tier-sharded plain/zero-shot cohort |

Rounds default to 6 for tool modes and 1 for plain modes; override with config or `HARNESS_ROUNDS`.

## Point it anywhere

Put `harness_config.json` in the attached dataset (it is generated from HF, so edit there, not locally):

```json
{
  "mode": "harness",
  "shard": "prompts_audit_medium.jsonl",
  "limit": 100,
  "rounds": 6,
  "k_memory": 3,
  "journal": "audit_medium.jsonl",
  "manifest": "audit_medium_manifest.json",
  "memory_dir": "",
  "export_dir": "",
  "allowed_models": ["gpt-6-luna", "claude-haiku-5-5"]
}
```

Local environment variables override the file: `HARNESS_MODE`, `HARNESS_SHARD`, `HARNESS_LIMIT`, `HARNESS_ROUNDS`, `HARNESS_K_MEMORY`, `HARNESS_JOURNAL`, `HARNESS_MANIFEST`, `HARNESS_ALLOWED_MODELS`, `HARNESS_CONFIG`.

## Commands

```powershell
# push (auto-run default is skipped by the guard; no quota spent)
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks push thermo-kaggle-runner `
  --file notebooks/kaggle_model_runner.py `
  --kaggle-dataset rezanadimi/thermo-agent-data

# run explicitly, only budget models
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks run thermo-kaggle-runner `
  -m gpt-6-luna -m claude-haiku-5-5 --wait 1800 -v

# download journals
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks download thermo-kaggle-runner `
  -m gpt-6-luna -o data/teacher_runs/kaggle_hosted/budget_batches
```

Then gate/ingest locally:

```powershell
.venv/Scripts/python.exe scripts/frontier_cohort_tokengate.py journal gpt-6-luna <downloaded>.jsonl <gated>.jsonl <manifest>.json
.venv/Scripts/python.exe -m thermo_agent.teacher.transfer ingest data/processed <gated>.jsonl data/frontier_tools_cohort/gpt-6-luna --model gpt-6-luna --run-id <runid> --extra-export-dir data/processed_arch --extra-export-dir data/frontier_prov
.venv/Scripts/python.exe scripts/check_datagen_complete.py
```

`transfer.ingest` writes success/failure memory notes; the gate drops (never truncates) rows over 8,192 tokens.
