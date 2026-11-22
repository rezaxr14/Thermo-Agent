# TEACHER_CAMPAIGN - hosted frontier rows

## Current truth (2026-11-22)

Only budget models are used: `gpt-6-luna` and `claude-haiku-5-5`. The hosted runner
exposes all 16 tools by default and requires a `reason` on every teacher tool call.
The full conversation is replayed into one persistent chat, tools execute once, and the
native structured trajectory is persisted for SFT.

## Accepted frontier evidence

`data/frontier_tools_accepted` holds 11 unique verifier-accepted rows: 7 repair and 4
ledger from the `pilot40` cohort. Duplicate plants attempted by both models were
deduplicated by `record_id`. These rows are migrated to native structured messages at
mix time by `scripts/mix_tool_corpora.py`.

## Reasoning gate

A tool-using trace is accepted only if the first tool turn carries reasoning, at least
75% of tool turns carry >=30 characters, and the reasoning is not one repeated sentence.
Provider reasoning is preferred; visible per-turn prose is the fallback. The teacher
`reason` argument fills only turns with no other reasoning.

## Re-pilot status (2026-11-22)

`native_pilot` shards: 40 rows per model, 10 per task, 14/13/13 tiers. Local preflight
selected 40/40 per model; rehearsal gated/ingested 39/39 with zero loss. HF sync is
done. Kaggle connectivity is working again and the bridge kernel was pushed (version
16, COMPLETE), but the Kaggle dataset copy is stale: dataset preflight returns 404 for
the new shard files. Therefore no hosted luna/haiku calls were made and the >=0.85
reasoning-share gate is unverified. No bulk run was launched.

## Next hosted action

Refresh the Kaggle dataset, re-run dataset preflight, run only luna and haiku on the
existing shards, apply the gate, ingest accepted native rows, rebuild the mix and re-run
the provenance audit. Never promote a legacy XML trace.
