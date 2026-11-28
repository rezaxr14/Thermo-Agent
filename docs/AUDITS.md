# AUDITS - verification evidence

## Current truth (2026-11-27)

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

## Audit format and ledger tool flow (2026-11-27, step 4)

Diagnosis from the native pilot journal (10 audit + 10 ledger attempts per model), no new
model call:

- All 20 audit answers fail `validate.validate("audit", ...)`: 18 with
  `format: unknown component ids ['DC']` (9 per model); the other 2 (`aud-0-00078`) with
  `change: change not allowed` and `prediction: no usable prediction`.
- Cause: the harness built the advertised rankable ids as plant components minus
  combustor/gas engine, which includes `dc_load`. `audit.truth.ranking` excludes
  `dc_load` from `ed_json` (the IT load is not part of the plant) and
  `parse_audit(text, set(ed))` rejects it, so following the prompt guaranteed a format
  failure. Three raw answers per model and the full explanation:
  `reports/step4_audit_format.md`.
- Fix: the advertised set is exactly `set(ed_json) - {combustor, gas_engine, dc_load}`.
  `tests/test_audit_ranking_prompt.py` asserts the advertised set equals the validator
  set and that `DC` is absent; it failed before the fix.
- Haiku ledger: all 10 attempts have one assistant turn and zero tool calls. The ledger
  verifier accepted the text (reward 1.0), so only the reasoning gate caught them, by
  accident. Fix: `transfer.ingest` now rejects a ledger row with no tool call
  unconditionally (`rejected_no_tool_use:ledger`, `reason:no_tool_use`), and the ledger
  contract states the rule. Test:
  `tests/test_transfer.py::test_ingest_rejects_ledger_without_tool_calls`.
- Re-ingesting the stored journals with `--tasks ledger` and no `--require-tools`:
  luna accepts 10/10 (`accepted_tool_using=10`), haiku rejects 10/10
  (`rejected_no_tool_use:ledger=10`). No model call.

## Tool results erased by the cleaner (2026-11-27, step 4)

- Symptom: in the 30 accepted frontier rows, 114 of 158 stored tool messages were the
  literal string `None`; the harness mixes rendered the same results as empty strings.
  Training on those rows would have taught the model that tool calls return nothing.
- Cause: `teacher.toolkit.compact_result` reused its `text` parameter as the loop
  variable for the solver hint, so the unchanged path returned `payload.get("message")`
  (None) or the bare hint instead of the original JSON.
- Fix: the loop uses `hint`; an unchanged payload is returned verbatim. Tests:
  `tests/test_toolkit.py::test_compact_result_returns_an_unchanged_payload_verbatim`
  and `tests/test_tool_protocol.py::test_normalize_tool_turns_keeps_the_served_output`
  (both fail before the fix).
- Re-ingest (no model call): both journals were re-gated (luna 18 survived / 3 over,
  haiku 15 / 0) and re-ingested with `--require-tools`. Result: **32 accepted rows**
  (luna 18: repair 9, design 2, ledger 7; haiku 14: repair 10, design 4), **0 `None` or
  empty tool results**, max 7,306 tokens under the gate. `data/frontier_tools_cohort`
  and `data/frontier_tools_accepted` were rebuilt from the corrected rows.

## Row retention after the step-4 prompt fix (2026-11-27)

`python scripts/gate_retention_report.py --corpus data/sft_tools_v3 --out reports/gate_retention_step4.json`

- Deterministic: **4,778 / 4,800 kept (99.54%)**.
- Per task: repair 100%, design 100%, audit 99.08%, ledger 99.08%.
- Worst tiers: audit/hard and ledger/hard 97.25%; every tier >=97.25%.
- All 22 drops are hard-tier, 8.2k-8.8k training tokens. The ledger prompt gained one
  sentence, which added two hard ledger drops versus step 3.
- Accepted frontier: 32 / 32 under the gate, max 7,306 tokens (re-ingested with real
  tool results; the previous 30-row set measured 7,369 but had empty results).

The corpora were rebuilt from `data/processed` with the fixed prompt and the mixes
re-derived:

`python -m scripts.audit_harness_corpus --dirs data/sft_tools_v3_h data/grpo_tools_v3_h data/sft_mix_v3 data/grpo_mix_v3 --out reports/harness_corpus_audit_step4.json`

Result: **20,776 rows, 0 over 8,192 (rendered), 0 over 8,192 (pipeline), 0 anomalies**
(legacy call syntax, malformed JSON, missing prompt variant, render failure).

| Task | Rows | rendered p50/p95/max | pipeline p50/p95/max |
| --- | ---: | --- | --- |
| audit | 5,156 | 4,202 / 7,039 / 8,050 | 3,033 / 5,470 / 6,459 |
| design | 5,212 | 3,171 / 5,957 / 6,280 | 2,003 / 4,198 / 4,401 |
| ledger | 5,170 | 4,868 / 7,151 / 8,068 | 3,756 / 5,765 / 6,696 |
| repair | 5,238 | 3,622 / 5,448 / 7,139 | 2,453 / 4,012 / 5,458 |

## Provenance (2026-11-27, step 4)

`python scripts/audit_provenance_chain.py --strict`: **PROVENANCE OK, 9/9 checks** -
30,408 rows scanned for protocol markers, 11,220 mix rows inspected, 0 leaks.

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

## Tests (2026-11-27, step 4)

- Full suite: **1,188 passed, 9 skipped** in 1,072.0 s
  (`$env:LLM_DEFAULT="gpt-6-luna"; .venv/Scripts/python.exe -m pytest -q`).
- The 9 skips are data-gated, with explicit reasons:
  - 8 x `tests/test_final_mix_tiers_provenance.py`: "v2 mix not built yet" (the v2 mix is
    not part of the v3 pipeline).
  - 1 x `tests/test_mix_tool_corpora.py`: "needs the local corpora (pre-CoT tool corpora
    are archived under `data/archive/`)".
- No test failed to run. kbench collection needs `LLM_DEFAULT=gpt-6-luna`; that is an
  import sentinel, not a model call.
- Step-4 additions: `tests/test_audit_ranking_prompt.py` (3 tests: advertised ids,
  equality with `ed_json`, no `DC`), `tests/test_transfer.py` (zero-tool ledger
  rejection), `tests/test_toolkit.py` (unchanged compact payload), and
  `tests/test_tool_protocol.py` (cleaned turn keeps the served output).
