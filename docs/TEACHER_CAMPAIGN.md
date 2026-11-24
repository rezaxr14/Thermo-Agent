# TEACHER_CAMPAIGN - hosted frontier rows

## Current truth (2026-11-24)

Only budget models are used: `gpt-6-luna` and `claude-haiku-5-5`. The hosted runner
exposes all 16 tools by default and requires a `reason` on every teacher tool call.
The full conversation is replayed into one persistent chat, tools execute once, and the
native structured trajectory is persisted for SFT.

## Accepted frontier evidence

`data/frontier_tools_accepted` holds 30 verifier-accepted native rows from the
`native_pilot` journals: repairs 17, designs 6, ledgers 7, audits 0. Two models solving
the same plant are two distinct traces and both are kept: rows dedupe on the
model-tagged `id`, never on the shared `record_id`.

The cohort was rebuilt from the two journals with
`scripts/process_budget_batch.ps1`; the previous hand-normalized 18-row snapshot is
archived at `data/archive/frontier_tools_cohort_normalized_2026-11-24/`. Re-ingest
costs nothing; the paid pilot itself cost $0.07797 (luna) and $0.14529 (haiku).

| model | journal rows | gate survived | cohort rows | tool turns | reasoning share |
|---|---|---|---|---|---|
| gpt-6-luna | 40 | 18/21 tool rows | 17 | 197 | 100% |
| claude-haiku-5-5 | 40 | 15/15 tool rows | 13 | 110 | 100% |

Reasoning share counts tool turns whose `reason` argument (or provider reasoning) is
non-empty after normalization. Haiku's ledger answers are zero-shot prose: no tool
turn exists, so they cannot enter a tool cohort.

The mix selects 17 of the 30 accepted rows. Thirteen are dropped by the exact
8,192-token gate (repair 8, ledger 5). One gated repair row per model
(`arch-107-00019#v16`) resolves only through `data/frontier_export`, which ingest does
not read; both are skipped as `skipped_unknown_or_empty` (step 3 fixes this).

## Reasoning gate

A tool-using trace is accepted only if the first tool turn carries reasoning, at least
75% of tool turns carry >=30 characters, and the reasoning is not one repeated sentence.
Provider reasoning is preferred; visible per-turn prose is the fallback. The teacher
`reason` argument fills only turns with no other reasoning.

## Audit yield

Zero of the 20 audit traces reached the cohort. The verifier rejects every audit answer
for format (9 per model) plus change/prediction. Step 4 owns this; do not promote audit
rows until the validator and the prompt agree on the required `<analysis>` /
`<ranking>` / `<change>` / `<prediction>` contract.

## Next hosted action

The 40-row pilot is paid and ingested; no new Kaggle run is authorized. Next work is
local: fix row loss and the exact gate in step 3, fix audit yield in step 4, review real
rows in step 5, and restore the docs in step 6.
