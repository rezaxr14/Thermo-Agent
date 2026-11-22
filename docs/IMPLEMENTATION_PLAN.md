# IMPLEMENTATION_PLAN - remaining work

## Current truth (2026-11-22)

Steps 1-3, 5 and 6 are complete. Step 4 is blocked on a stale Kaggle dataset copy, not
on local plumbing or credentials. No training or bulk run is authorised.

## Done

1. Native structured tool protocol everywhere; legacy XML rejected; real-tokenizer
   round-trip and 3-step parse smoke in `tests/test_tool_protocol.py`. Corpora rebuilt
   under the exact 8,192 gate. Commit `2906d2b`.
2. Teacher-only required `reason` schema; ingest moves it into `<think>` and strips it
   from student call arguments. Commit `9307a46`.
3. Audit rankable ids + DC rejection test; design final answer forced to the last
   validity-check-passed candidate; ledger compact table. Commit `046bcfe`.
5. One-row-at-a-time repair/design reviews in `reports/trace_review.jsonl`; the four old
   markdown reviews deleted.
6. Docs curated to 10 files / 31 KB, each opening with a current-truth section.

## Step 4 - what remains

1. Refresh `rezanadimi/thermo-agent-data` from the synced HF corpus. The bridge kernel
   (version 16) reports COMPLETE, but the dataset's lastUpdated is still 2026-11-21 and
   `preflight_budget_batch --dataset` returns 404 for the new `native_pilot` shards.
2. Re-run the dataset preflight. Do not rebuild a 100-row batch; reuse the existing
   40-row `native_pilot` shards (10 rows/task, 14/13/13 tiers per model).
3. Push the generic runner and run only `gpt-6-luna` and `claude-haiku-5-5`.
4. Gate each model: reasoning share >=0.85 overall; at least one accepted row in every
   task for at least one model. Record raw metrics first.
5. Append the dated result to `reports/pilot40_report.md`, ingest only accepted native
   rows, rebuild the mix, and re-run the provenance audit.

## Explicitly not authorised

- Any bulk Kaggle run beyond the 40-row luna/haiku pilot.
- Training, SFT, GRPO, reward tuning or a full evaluation sweep.
- Mixing `data/archive/` into any live corpus.
