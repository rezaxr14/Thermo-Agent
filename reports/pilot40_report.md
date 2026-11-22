# 40-row reasoning pilot - luna + haiku (2026-11-21/22)

## 2026-11-22 - native-protocol re-pilot: shards ready, hosted run BLOCKED

`scripts/build_next_budget_batch.py --size 40 --name native_pilot` produced two
40-row shards (10 per task, 14/13/13 tiers). Local preflight selected 40/40 for each
model; `scripts/rehearse_budget_batch.py --rows 40` gated/ingested 39/39 per model
with no loss, proving the post-processing path. Shards were synced to the HF corpora
dataset. Kaggle connectivity was later restored and the bridge kernel was pushed
(version 16, COMPLETE), but the Kaggle dataset copy is stale: dataset preflight returns
404 for the new shard files. No hosted luna/haiku calls were made. No re-pilot
metrics exist; the reasoning-share gate (>=0.85 per model) and the per-task accepted
row requirement remain unverified. No bulk run was started.

Setup: `scripts/build_next_budget_batch.py --size 40 --name pilot40` produced two shards of exactly 40 rows
(10 per task, tiers rotated 3/3/4 so the shard-wide counts are 14/13/13 and the hosted `select_rows` keeps all
40). `harness_config.json`: rounds 6, `dynamic_tools: true`, 16 tools default, limit 40, journal `pilot40_completions.jsonl`.
Local preflight: 40/40 selected per model. HF sync → bridge kernel → Kaggle dataset preflight: 40/40 selected.
Task version 22 ran both models to completion (no runner errors after the stale-bundle fix). Runs:
luna 7302405, haiku 7302406.

## Raw journal metrics (before any gate)

From `reports/pilot40_raw_journal_metrics.json`. "Reasoned" = a tool turn whose `reasoning` or visible `assistant`
prose is >= 30 chars. Every row has a non-empty completion and `error = null`.

| Model | Task | Rows | Tool turns | Reasoned share | Mean chars | Raw cost |
|---|---|---:|---:|---:|---:|---:|
| gpt-6-luna | repair | 10 | 46 | 0.652 | 207.2 | $0.0143 |
| gpt-6-luna | design | 10 | 55 | 0.618 | 237.0 | $0.0178 |
| gpt-6-luna | audit | 10 | 47 | 0.851 | 265.2 | $0.0234 |
| gpt-6-luna | ledger | 10 | 28 | 0.607 | 260.2 | $0.0117 |
| **gpt-6-luna** | **ALL** | **40** | **176** | **0.688** | — | **$0.0671** |
| claude-haiku-5-5 | repair | 10 | 32 | 0.719 | 196.9 | $0.0306 |
| claude-haiku-5-5 | design | 10 | 41 | 0.780 | 175.4 | $0.0326 |
| claude-haiku-5-5 | audit | 10 | 35 | 1.000 | 245.3 | $0.0469 |
| claude-haiku-5-5 | ledger | 10 | 3 | 1.000 | 206.3 | $0.0205 |
| **claude-haiku-5-5** | **ALL** | **40** | **111** | **0.838** | — | **$0.1305** |

Pilot reasoning rule (DECISIONS §2026-11-20 item 8): >= 75% of tool turns with reasoning.
**luna FAILS (0.688); haiku PASSES (0.838).**

## Gate / ingest outcome

`scripts/process_budget_batch.ps1` (real tokenizer gate + `--require-tools` ingest), counted from the raw journals
so this is the first-pass result (no double-ingest):

| Model | Journal rows | Tool+verifier accepted | Survived 8k | Dropped over 8k | Ingest accepted | `rejected_no_reasoning` | `rejected_no_tool_use` |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-luna | 40 | 22 | 16 | 6 | 6 | 9 (repair 2, design 2, ledger 5) | 0 |
| claude-haiku-5-5 | 40 | 15 | 15 | 0 | 6 | 7 (repair 5, design 2) | 1 (ledger) |

Final cohort after id-dedupe: luna 6 unique rows (repair 3, ledger 3), haiku 6 unique rows (repair 4,
ledger 2). **No audit and no design rows were accepted by either model.** Cost per accepted trace (full pilot
cost / accepted): luna $0.0671 / 6 = **$0.0112**, haiku $0.1305 / 6 = **$0.0218**.
(`usage` over the gated attempts only: luna $0.0203, haiku $0.0460.)

## Yield diagnosis from the raw journals (before bulk)

Local verifier re-scored each journal completion against `data/frontier_export` by id:

| Task | luna verifier-ok | haiku verifier-ok | Failure class |
|---|---:|---:|---|
| repair | 10/10 | 10/10 | — |
| ledger | 8/10 | 9/10 | `exact` (off-by-one ledger fix), `closure` |
| design | 4/10 | 3/10 | `plant` (design does not solve), `grounding` (values not in site) |
| audit | 0/10 | 0/10 | `format: unknown component ids ['DC']` (9/8 rows), `grounding` (1/2) |

- **audit (0/10 both):** the export's `ed_json` allowlist is `CC, GT, CH, AC` — `DC` (the data-centre load) is
  deliberately excluded from exergy-destruction ranking. Both models rank `DC` first/second (it appears large in
  the plant), so `parse_audit` rejects it as an unknown id before any other check. This is a harness/contract
  inconsistency, not a model typo: nothing in the served contract tells the model which ids are rankable. Fix
  before bulk by stating the rankable id set in the audit prompt (from `ed_json` keys), NOT by loosening the
  verifier to accept bad ids.
- **design (3-4/10):** models emit plausible configurations that either fail the solver or cite values outside
  the site; expected for a 40-row pilot, but must be watched in bulk.
- **ledger (8-9/10):** high yield; a few `exact`/`closure` misses.
- haiku produced only **3 tool turns across 10 ledger rows** (it often answered without tools); `--require-tools`
  correctly rejected the 8 no-tool rows.

## Verdict

The pilot does not yet clear the DECISIONS gate: luna is below the 0.75 reasoning share, both models have zero
audit/design accepted rows, and acceptance is 6/40 per model. No bulk run is launched. Required follow-ups before
bulk: (a) expose rankable audit component ids in the served contract, (b) raise luna's reasoning compliance
(the contract already requires it — inspect the non-reasoned turns), (c) re-pilot after those fixes. These are
recorded in `docs/STATUS.md`.
