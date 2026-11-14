# Status and operating notes

Working copy on the second laptop: `C:\Users\rezax\Home\Code\thermo_agent\Repo` (git, branch `main`).

## Environment

| Item | Value |
|---|---|
| Machine | REZA_PC, Windows 11 Home, 16 cores, 15.2 GB RAM, RTX 4050 6 GB |
| Python | 3.12.3 in `.venv` (created with `uv`), pinned by `requirements-lock.txt` |
| Package | installed editable (`pip install -e . --no-deps`) |
| Tools | `uv`, `ripgrep`, `fd`, `jq` in `%USERPROFILE%\tools\bin` and `%USERPROFILE%\.local\bin` |
| Local model | LM Studio serving `qwen3.5-4b` on `http://127.0.0.1:1234/v1` |
| Remote access | SSH key auth, host alias `reza-pc` |

No WSL and no tmux: long jobs are detached Windows processes (`Wi32_Process.Create`), so they survive an SSH disconnect.

## Running long jobs

```powershell
pwsh scripts/long.ps1 start gen-data '& ./scripts/gen_data.ps1'   # start detached
pwsh scripts/long.ps1 status gen-data -Lines 40                    # state + log tail
pwsh scripts/long.ps1 list                                        # all jobs
pwsh scripts/long.ps1 kill gen-data
```

Logs: `logs/<name>.log`. A job writes `logs/<name>.pid`; `status` uses it to tell RUNNING from FINISHED.

## Pipeline

1. `scripts/gen_data.ps1` — repair (4000), audit (800), design (1500, max-evals 60), ledger (3000), then `reports/data_report.json`. CPU, seeded.
2. `scripts/baselines.ps1` — the non-model baselines from runbook section 3 plus reward diagnostics (3b).
3. Model evaluation against LM Studio: see `docs/LOCAL_MODEL.md`.

## State

- Test suite: 234 pass. The OpenAI backend adds 2 more (`tests/test_openai_backend.py`).
- Datasets are generated on this machine; they are git-ignored (`data/processed/`).
- Training has not been run: the runbook lists the 6 GB RTX 4050 as data generation, tests and non-model baselines only.
  GPU work is meant for a Kaggle dual-T4 later; nothing here depends on it.

## Next

1. Confirm `reports/data_report.json` has an empty `problems` list.
2. Run the baselines and record the numbers.
3. Write those numbers into `paper/outline.md` and `paper/methods_draft.md`, replacing the "measured on small samples" language with this run's values.
4. Evaluate `qwen3.5-4b` through LM Studio on small limits to get base-model numbers without a GPU box.
