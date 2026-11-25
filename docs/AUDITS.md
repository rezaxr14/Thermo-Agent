# AUDITS - verification evidence

## Current truth (2026-11-25)

All audits below were run locally. No model call, training run or paid API call was part
of an audit. Commands are reproducible with `.venv/Scripts/python.exe`.

## Native protocol and 8,192 gate (2026-11-25, step 3)

Command:

`python scripts/audit_harness_corpus.py --dirs data/sft_tools_v3_h data/grpo_tools_v3_h data/sft_mix_v3 data/grpo_mix_v3 --out reports/harness_corpus_audit_step3_after.json`

Result: **20,780 rows, 0 over 8,192 (rendered), 0 over 8,192 (pipeline), 0 anomalies**
(legacy call syntax, malformed JSON, missing prompt variant, render failure).

| Task | Rows | rendered p50/p95/max | pipeline p50/p95/max |
| --- | ---: | --- | --- |
| audit | 5,156 | 4,201 / 7,038 / 8,049 | 3,031 / 5,468 / 6,457 |
| design | 5,212 | 3,172 / 5,958 / 6,281 | 2,003 / 4,198 / 4,401 |
| ledger | 5,178 | 4,841 / 7,129 / 8,089 | 3,673 / 5,739 / 6,726 |
| repair | 5,234 | 3,623 / 5,429 / 5,751 | 2,451 / 3,988 / 4,325 |

The rendered measure is the training text (`prompt + completion`, one template render
with the 16 serving schemas); pipeline adds the 96-token chat reserve. The builder's own
gate reports max 8,185 with the reserve.

## Row retention (2026-11-25, step 3)

`python scripts/gate_retention_report.py --corpus data/sft_tools_v3 --out reports/gate_retention_step3.json`

- Deterministic: **4,780 / 4,800 kept (99.58%)**.
- Per task: repair 100%, design 100%, audit 99.08%, ledger 99.25%.
- Worst tier: audit/hard 97.25%, ledger/hard 97.75%; every tier >=97.25%.
- All 20 drops are hard-tier, 8.2k-8.8k training tokens.
- Accepted frontier: **30 / 30 under the gate**, max 7,369 tokens.

Full table and dropped ids: `reports/step3_row_retention.md`.

## Provenance (2026-11-25, step 3)

`python scripts/audit_provenance_chain.py --strict`:
**PROVENANCE OK, 9/9 checks** - cohort provenance fields, frontier source mapping,
recomputed tiers, prompt-shard leakage, harness leakage, memory outcomes/serving, mix
propagation, mix answer-field leakage, protocol markers. The marker check scans 7 corpus
directories and 30,410 rows for `call_id`, `ToolInvocationResult` and
`<tool_call name=`; 0 leaks. Mix leakage: 11,220 rows, 0 `ref_spec` leaks.

## Prompt separation (2026-11-25)

`python scripts/audit_prompt_separation.py`: all arms pass. The H arm is detected by
the `MULTI-TURN TOOL ENVIRONMENT` header and the absence of answer oracles; the retired
`<tool_call ...>` marker is deliberately not used as a contract detector.

## Trace reviews (2026-11-25)

`reports/trace_review.jsonl` still contains the earlier one-row-at-a-time reviews.
Step 6 replaces it with real whole-row reviews (3 rows/task for deterministic and
frontier) and explains the `reasoning_ok=false` / `accepted=false` rows.

## Tests (2026-11-25, step 3)

- Full suite: **1,183 passed, 9 skipped** in 1,178.7 s
  (`$env:LLM_DEFAULT="gpt-6-luna"; .venv/Scripts/python.exe -m pytest -q`).
- The 9 skips are data-gated, with explicit reasons:
  - 8 x `tests/test_final_mix_tiers_provenance.py`: "v2 mix not built yet" (the v2 mix is
    not part of the v3 pipeline).
  - 1 x `tests/test_mix_tool_corpora.py`: "needs the local corpora (pre-CoT tool corpora
    are archived under `data/archive/`)".
- No test failed to run. kbench collection needs `LLM_DEFAULT=gpt-6-luna`; that is an
  import sentinel, not a model call.
