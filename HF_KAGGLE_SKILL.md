---
name: hf-kaggle-dataset-sync
description: Keep this project's Hugging Face corpora dataset and its Kaggle datasets fresh, auditable and isolated. Use when refreshing training corpora, mirroring data to/from Kaggle, or pulling the same bytes inside a Kaggle notebook. Hard allow-lists only; never touch another dataset or model repo.
---

# HF + Kaggle dataset sync: project field guide

Short actionable companion to `KAGGLE_SKILL.md` (benchmarks) and `docs/KAGGLE_CLI.md` (raw commands).
Read those for model calls and quota. This file is only about **data movement**.

## 1. Two HF repos, never mixed

| Repo | Type | What lives there | Writer |
|---|---|---|---|
| `Reza-Nadimi/thermo-agent-corpora` | private **dataset** | SFT/GRPO tool corpora, mix corpora, A-slice, frontier sets, memory, held-out **prompts only** | `scripts/sync_corpora_hf.py` |
| `Reza-Nadimi/thermo-agent-checkpoints` | private **model** repo | LoRA/optimizer/state checkpoints + merged models | `thermo_agent/train/hf_sync.py` |

The checkpoint repo uses **distinct stage prefixes** so arms cannot overwrite each other:
`sft/checkpoint-N/`, `grpo_k4/checkpoint-N/`, `grpo_k8/checkpoint-N/`, `grpo_k16/checkpoint-N/`
(and legacy `grpo_<task>/`), with merged weights under `merged/`. Corpora sync never writes there.
Each script refuses any other repo id **before a network call**; the ids are constants, not env vars.

> **Never touch other datasets or model repos on HF or Kaggle.** No other id is accepted, no
> repository is created or deleted outside the two HF repos and the six Kaggle datasets below.

## 2. Exact sync commands (PowerShell, repo root)

```powershell
# 1) Corpora -> HF. ALWAYS dry-run first: prints the plan and writes a sha256 manifest, sends nothing.
python scripts/sync_corpora_hf.py --dry-run
# 2) Real refresh: explicit message + tag; additive only (never deletes remote files).
python scripts/sync_corpora_hf.py --message "v2 corpus rebuild" --tag 2026-11-19

# Kaggle dataset upload / download. Proxy is scoped to the kaggle command by the helper itself;
# if you run a raw kaggle command, scope it the same way:
$env:HTTP_PROXY='socks5h://localhost:10808'; $env:HTTPS_PROXY='socks5h://localhost:10808'
.venv/Scripts/kaggle.exe datasets status rezanadimi/thermo-agent-code
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY -ErrorAction SilentlyContinue

python scripts/sync_kaggle_datasets.py upload   --dataset thermo-agent-code --dry-run
python scripts/sync_kaggle_datasets.py download --dataset thermo-agent-corpus --dry-run
python scripts/sync_kaggle_datasets.py upload   --datasets thermo-agent-code thermo-agent-corpus `
    --message "code+corpus rebuild" --tag 2026-11-19
```

Every real HF sync writes an audit manifest locally under `reports/hf_corpora_manifests/` **and** uploads
it to `manifests/` in the dataset: repo id, UTC timestamp, message, tag, and per-file path/sha256/size.
Refreshes are additive versions; nothing remote is ever deleted.

## 3. Hard allow-lists (a typo or a foreign id is refused)

HF dataset dirs (`scripts/sync_corpora_hf.py`):
`data/sft_tools_v2`, `data/grpo_tools_v2`, `data/sft_tools_v2_h`, `data/grpo_tools_v2_h`,
`data/sft_mix_v2`, `data/grpo_mix_v2`, `data/a_slice`, `data/frontier_tools_clean_v2`,
`data/frontier_tools_pilot`, `data/frontier_tools_cohort`, `data/memory`, `data/heldout_v2`.

`data/heldout_v2` is **prompts only**: a per-field filter keeps `id`, `split`, `task`, `topology`,
`fault_type`, `tier`, `gen_seed`, `prompt` and drops completions, answers, reports, verdicts, ground
truth and every other solution-bearing field at any nesting depth. Never restore answers from this copy.

Kaggle dataset slugs (`scripts/sync_kaggle_datasets.py`, all under owner `rezanadimi`):
`thermo-agent-code`, `thermo-agent-corpus`, `thermo-agent-resume`, `thermo-teacher-prompts`,
`thermo-teacher-prompts-big`, `thermo-agent-eval-replays`. Local dirs live under
`data/kaggle_datasets/` (`thermo-teacher-prompts-big` → `teacher_big_prompts`). The adjacent
`thermo-kernel-wheels` dataset is **not** in the list and is always refused.

## 4. Tokens and logs

`HF_TOKEN` / `KAGGLE_API_TOKEN` come from the environment or `.env`; the helpers pass them only to the
HF/Kaggle clients and **never print or upload them**. No tokens in notebooks, committed files, dataset
metadata or logs. The Kaggle proxy (`socks5h://127.0.0.1:10808`) is set only in the kaggle subprocess
environment - non-Kaggle calls are never proxied.

**One repo per purpose.** DATA goes only to `Reza-Nadimi/thermo-agent-corpora`; MODELS/adapters only to
`Reza-Nadimi/thermo-agent-checkpoints`. No third repo, no project mirror, no mixing. On Kaggle keep project
data in ONE dataset (`thermo-agent-corpus`); attach the code/prompt datasets read-only, and reuse an existing
shard with a row limit instead of creating new proof shards, tasks or datasets.

**One Kaggle account only.** `.env` must contain exactly one `KAGGLE_API_TOKEN`: the project account that
owns the `rezanadimi` datasets and the AI quota we have already spent. A second token (a different
account) was removed; never add it back and never mix dataset access with AI billing accounts. Run every
Kaggle CLI command through `powershell -NoProfile -File scripts/kaggle_project.ps1 <kaggle args>` — it
reads the single token from `.env`, scopes the proxy to that one command, and never prints the value.
Verify access with `... datasets status rezanadimi/thermo-teacher-prompts-big` (`ready`); if the SOCKS
proxy aborts the first attempt, retry the same read-only command rather than switching accounts.

In a Kaggle notebook, read the token from a Kaggle
Secret or an attached private `env` dataset; do not paste it into a cell.

## 5. Refresh cadence

- **Corpora:** after datagen is frozen and the v2 compact-harness corpora pass `token_audit`/`verify`
  (H1-H3), then on every corpus rebuild. Dry-run, review the manifest, then push with a new tag.
- **Checkpoints:** automatic after every Trainer save via `hf_sync` (12 h Kaggle sessions); each
  session pulls the newest step before training (`python -m thermo_agent.train.hf_sync status --stage sft`).
- **Kaggle datasets:** version `thermo-agent-code` when the bundle changes, corpora/replays when data
  is regenerated; check with `kaggle datasets status <id>` after each version.
- Periodic means *additive*: same repo/dataset id, new commit/version + tag + manifest, nothing removed.

## 6. Pull the same data inside Kaggle

```python
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Reza-Nadimi/thermo-agent-corpora", repo_type="dataset",
    token=None,  # inject from Kaggle Secret / attached private env dataset; never hard-code
    allow_patterns=["data/sft_tools_v2/*", "data/grpo_tools_v2/*", "data/heldout_v2/*"],
)
```

For Kaggle-hosted copies, in the notebook set the proxy only around the download and use the helper
from this repo (or the raw CLI), then point `DATA_DIRS` at the mounted `data/` tree:

```powershell
$env:HTTP_PROXY='socks5h://localhost:10808'; $env:HTTPS_PROXY='socks5h://localhost:10808'
python scripts/sync_kaggle_datasets.py download --dataset thermo-agent-corpus
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY -ErrorAction SilentlyContinue
```

Never pull a checkpoint from someone else's repo: `hf_sync.resolve_repo` accepts only
`Reza-Nadimi/thermo-agent-checkpoints`.
