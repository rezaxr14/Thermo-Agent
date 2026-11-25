# IMPLEMENTATION_PLAN - remaining work

## Current truth (2026-11-25)

Hard-fix steps 1-3 are complete: one native protocol (legacy traces archived, never
migrated), the native pilot re-ingested as 30 accepted rows, and the exact 8,192 gate now
measures the training render with retention 4,780/4,800 deterministic rows (99.58%).
Step 4 (audit format + haiku ledger tool flow) is next. No premium model, bulk run,
training, SFT, GRPO or eval launch is authorised.

## Done

1. Step 1 - one native protocol. Legacy `<tool_call name=...>` rows archived under
   `data/archive/`, converter/normalizer, 9-check protocol audit. Commit `d81ca1f`.
2. Step 2 - native pilot re-ingested: 30 accepted rows (luna 17, haiku 13; repair 17,
   design 6, ledger 7, audit 0). Pre-reingest cohort archived. Evidence in
   `reports/native_pilot_reingest.md`. Commit `bda80ab`.
3. Step 3 - row loss fixed at the exact gate:
   - the gate measures the training text (harness prompt + completion, one template
     render with the 16 serving schemas, +96 reserve), not the teacher prompt;
   - tool results are compact tables (`columns`/`rows`, `line_table`), skill bodies are
     budgeted, descriptions are one sentence, reasoning cites only the worst lines;
   - stored frontier results re-served through the same compaction;
   - retention 99.58% deterministic (audit 99.08%, ledger 99.25%, repair/design 100%;
     every tier >=97.25%); accepted frontier 30/30 under 8,192, max 7,369;
   - ingest reads `data/frontier_export`. Evidence in
     `reports/step3_row_retention.md`, `reports/gate_retention_step3.json`,
     `reports/harness_corpus_audit_step3_after.json`.

## Step 4 - audit format and the haiku ledger tool flow

1. Print three raw audit final answers per model next to the expected
   `<analysis>/<ranking>/<change>/<prediction>` contract.
2. Fix the contract/validator/parser mismatch with a failing-then-passing test.
3. Explain and fix haiku ledger traces with zero tool calls; a ledger row without at
   least one tool call must not be accepted.
4. Re-run only the audit + ledger pilot rows for luna and haiku if a code change is
   needed to confirm the fix; no other model.

## Step 5 - reasoning quality gate

- Each `reason` >= 40 characters, distinct within a trace.
- Non-first tool turns: at least half must cite an identifier or number from an earlier
  tool result.
- Report per model x task; add a filler-reason failure test.

## Step 6 - whole-row reviews

- Read three whole rows per task for deterministic and frontier rows.
- Explain the 20 deterministic `reasoning_ok=false` rows and the 6 frontier
  `accepted=false` rows; fix the gate or the data.
- Replace `reports/trace_review.jsonl` with real whole-row reviews.

## Step 7 - docs and test hygiene

- Restore the training/decoding parameter table from `config.py` and
  `tests/test_decisions_applied.py`; add the anti-litter rule.
- Make transformer/trl/untracked-data tests skip clearly; make the provenance audit skip
  a missing `data/memory`.
- Keep `docs/` at exactly 10 files and under 150 KB.

## Step 8 - re-pilot and stop

- Audit + ledger only, luna + haiku, 10 rows each (40 attempts).
- Report accepted/rejections per model x task, then stop and wait.

## Explicitly not authorised

- Any model beyond `gpt-6-luna` and `claude-haiku-5-5`.
- Any bulk run beyond the stated pilot shards.
- Training, SFT, GRPO, reward tuning or a full evaluation sweep.
- Mixing `data/archive/` into any live corpus.
