# AUDITS - verification evidence

## Current truth (2026-11-22)

All audits below were run locally. No model call, training run or paid API call was part
of an audit. Commands are reproducible with `.venv/Scripts/python.exe`.

## Native protocol and 8,192 gate (2026-11-22, final)

Command:
`python -m scripts.audit_harness_corpus --dirs data/sft_tools_v3_h data/grpo_tools_v3_h data/sft_mix_v3 data/grpo_mix_v3 --out reports/harness_corpus_audit_native.json`

Result: **10,404 rows, 0 over 8,192 (rendered), 0 over 8,192 (pipeline), 0 anomalies**
(legacy call syntax, malformed JSON, missing prompt variant, render failure).

| Task | Rows | rendered p50/p95/max | pipeline p50/p95/max |
| --- | ---: | --- | --- |
| audit | 2,092 | 5,709 / 8,073 / 8,096 | 2,912 / 4,907 / 4,940 |
| design | 1,992 | 4,819 / 8,070 / 8,096 | 2,022 / 4,744 / 4,784 |
| ledger | 1,110 | 5,586 / 8,055 / 8,096 | 2,798 / 5,085 / 5,139 |
| repair | 5,210 | 5,320 / 7,131 / 7,451 | 2,523 / 4,063 / 4,398 |

The 96-point difference between rendered and pipeline maxima is the chat reserve. The
earlier plain-text gate produced 3,270 over-gate rows at the same nominal 8,192 limit,
so it was not a valid measurement and was replaced.

## Provenance (2026-11-22)

`python scripts/audit_provenance_chain.py --strict`:
**PROVENANCE OK, 8/8 checks** - cohort provenance fields, frontier source mapping,
recomputed tiers, prompt-shard leakage, harness leakage, memory outcomes/serving, mix
propagation and mix answer-field leakage all pass. 0 `ref_spec` leaks in 7,086 mix rows.

## Prompt separation (2026-11-22)

`python scripts/audit_prompt_separation.py`: all arms pass. The H arm is detected by
the `MULTI-TURN TOOL ENVIRONMENT` header and the absence of answer oracles; the retired
`<tool_call ...>` marker is deliberately not used as a contract detector.

## Trace reviews (2026-11-22)

`reports/trace_review.jsonl` contains 27 one-row-at-a-time repair/design reviews (21
verifier-accepted). Reviews use the verifier only for the final answer; scoring columns
are never quoted as review evidence. The four superseded `trace_review_2026-11-20_*.md`
files were deleted.

## Tests (2026-11-22)

- Full suite with `LLM_DEFAULT=gemini-2.5-flash` (needed only for kbench test
  collection): **1,172 passed, 9 skipped**.
- 9 skips are intentional: 8 "v2 mix not built yet" corpus guards, 1 legacy pre-CoT
  corpus guard.
- Two kbench files cannot be collected without `LLM_DEFAULT`; this is an environment
  requirement, not a test failure. No other test was not runnable.
