# Budget frontier batch report (overnight)

Rule: only `gpt-6-luna` and `claude-haiku-5-5` on hosted Kaggle runs. No premium model without Reza's explicit go. Every accepted trace is token-gated at 8,192, ingested with model/run provenance, and success/failure recaps go to local memory via `transfer.ingest`.

## Baseline at start of overnight work

- Gate: `passed=false`, **84** accepted frontier traces, **24 budget cells** model×task×tier cells missing.
- Completed proof: luna run `7233597` (5 rows, 5 accepted, $0.003358), haiku run `7233598` (5 rows, 5 accepted, $0.012746).
- Missing budget cells: luna/audit easy+medium+hard; haiku/audit easy+medium, haiku/ledger easy+medium.

## Batch log

| Batch | Task version | Models | Rows/model | Accepted after gate | Cost | Gate cells still missing | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| proof | 8 | luna, haiku | 5 | 5 + 5 | $0.016104 | 24 budget cells | Haiku sanitizer recovered over-gate wrappers. |
| batch 1 | 9 | luna, haiku | 100 | pending | pending | pending | Generic runner pushed into the existing task; per-model shards from unique-diagram plans. |

## Reusable code

- `notebooks/kaggle_model_runner.py` / `.ipynb` — one generic Kaggle task for tracegen/eval/zero-shot/cohort, with per-model shards and a hard block on gemini 3.8/3.5.
- `scripts/build_next_budget_batch.py` — builds the next 100-row per-model shards from the missing gate cells.
- `scripts/process_budget_batch.ps1` — gate → ingest → audit one downloaded journal.
- `scripts/kaggle_project.ps1` — the only Kaggle CLI wrapper; reads credentials from root `.env` and scopes the SOCKS proxy to one command.

## Current tally

- Accepted tool traces: **256**.
- Budget cells still empty: **7** (luna/audit easy+medium+hard; haiku/audit easy+medium; haiku/ledger easy+medium).
- Goal: thousands of traces, no cap, equal thirds across easy/medium/hard.
