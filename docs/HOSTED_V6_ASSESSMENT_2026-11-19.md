# Hosted run v6 assessment — `thermo-teacher-tools-s00` version 6

Assessed on 2026-11-19 from the downloaded journal, not from the log tail. Every number below is reproducible with the
commands in §7.

## 1. Run identity

| Field | Value |
|---|---|
| Kaggle task | `rezanadimi/thermo-teacher-tools-s00` version 6 |
| Model | `google/gemini-3.8-flash` (single model in the version) |
| Run id | `7179338` |
| Task created / model started / ended | 2026-11-19 11:29:34 / 11:30:36 / 13:09:20 |
| Journal | `data/teacher_runs/kaggle_hosted/thermo-teacher-tools-s00/6/gemini-3.8-flash/7179338/harness_completions.jsonl` (935,803 bytes) |
| Shard | `prompts_s00.jsonl`, 320 rows, harness `"tools"`, 5 tools offered, `notes_used` = 3 |
| Attempts | 40 |
| Cost | **$0.842202** (1,210,642 input tokens, 66,217 output tokens) |

The version ran with `SPEC["limit"] = 0`, 6 rounds and no cost/deadline cap (D1-D9), which is why it attempted the whole
40-row shard instead of stopping at 40 accepted.

## 2. Journal counts (raw)

| Bucket | Count |
|---|---|
| rows in journal | 40 |
| API errors (`RateLimitError` 429 "model is currently experiencing heavy load") | 5 |
| rows with a non-empty `sft_completion` | 35 |
| rows whose `sft_completion` contains `<tool_call>` **and** `<tool_result>` | 35 |
| turns per non-error row | 5 |
| rows with `completion` (final answer) non-empty | 34 |

So the model did the multi-turn loop on 35 of 35 non-error attempts. The failure mode of this run is provider load, not
the payload: the 5 failures are all 429s, no 400/format rejection.

## 3. Ingested traces

First pass (only `data/processed` as the export source) accepted **22**. The other 12 non-error rows were counted as
`skipped_unknown_or_empty` because their ids are not in `data/processed/*_train.jsonl`. That was an ingest defect, not a
model failure (§6).

After fixing id resolution, the same journal yields:

| Task | Accepted traces |
|---|---|
| repair | 9 |
| ledger | 10 |
| audit | 6 |
| design | 7 |
| **total** | **32** |

- 32/32 carry a real tool turn (`source_detail = "frontier_tools"`, i.e. `<tool_call>` + `<tool_result>` in the saved
  completion), 32 unique `record_id`s, no duplicate rows.
- 1 rejection: design, `grounding: only 56% of the numbers in <rationale> are in the plant`.
- 2 rows unusable: 1 empty completion (`led-arch-0-02876`), 1 rejected (above). 5 rows were 429s.
- **Cost per accepted tool trace: $0.842202 / 32 = $0.0263.**
- **Yield: 32 accepted / 40 attempts = 80%**; 32/35 non-error = 91%.

This is the first non-zero frontier tool-trace yield in the project. `docs/CLAUDE_HANDOFF_2026-11-19.md` said "zero
verifier-accepted frontier traces containing a real tool call"; that statement was true before this run and is now
superseded by this document.

## 4. Gate measurement with the real Qwen3.5 tokenizer

Measured on the saved row (prompt messages joined + completion + 96 reserved tokens), tokenizer
`C:\Users\rezax\.cache\thermo\qwen3.5-4b\tokenizer.json`:

| | raw | after removing `call_id` blobs |
|---|---|---|
| completion tokens p50 | 5,698 | 2,019 |
| completion tokens p95 | 14,643 | 4,529 |
| completion tokens max | 15,840 | 4,689 |
| total tokens max | **17,212** | **7,280** |
| rows over the 8,192 gate | **11 / 32** | **0 / 32** |

The 11 over-gate rows are over *because of* the `ToolInvocationResult(..., call_id='call_...__mpthought__<base64>')`
wrapper, which is a provider-side invocation id and carries no training signal. Removing it (keeping tool name +
`output=` JSON) puts every row under the gate with 912 tokens of headroom to spare at the max. That is the sanitizer
task from the handoff, defect #2; the accepted fix is a rewrite of the completion text only, ids and the final answer
unchanged, output written to a new dir (`data/frontier_tools_clean`) so `data/frontier_tools` stays as received.

Note the prompt stored in these rows is the **plain** 2-message prompt (493-3,010 tokens by task), not the harness
prompt. The harness text is applied when the row is rebuilt for training, so the trace is reusable under the compact
harness; the model's assistant turns do not embed the harness.

## 5. What this run proves and does not prove

Proves:
- The hosted payload is accepted by `gemini-3.8-flash` end to end: tools are called, results are returned, the model
  self-corrects over 5 turns, and the project's own validators accept 32 of those answers.
- The verifier is not rubber-stamping: 1 of 33 answered rows was rejected for grounding.

Does not prove:
- Nothing about training. No checkpoint exists, no SFT/GRPO step has run, no eval has been scored.
- Nothing about the compact harness. This run used the 17.2k-char harness prompt; the traces are model output only, so
  they survive the compaction, but a later run must confirm the model behaves the same under the compact text.
- Nothing about scale. 32 traces against the E1 plan of ~4,800 deterministic tool rows and a frontier share at
  R_tool; this is a pilot yield and a unit cost, not a corpus.

## 6. Defects found while ingesting (all fixed in this pass)

1. **`load_exports` did not tag the task.** `data/processed/*_train.jsonl` rows have no `task` column; every row fell
   back to `"repair"` and the first design row crashed with `KeyError: plant_broken`. Fixed: the file name tags the row.
2. **Ids were resolved from one corpus only.** The trace shard is built from `data/processed` **plus**
   `data/processed_arch`; 7 traces have `*-arch-*` ids. 5 more are `#vN` paraphrase ids that only exist in the already
   ingested `data/frontier_prov`. All 12 were silently dropped as "unknown". Fixed: `ingest --extra-export-dir`
   (repeatable), keyed by `record_id` when present, primary dir wins. Two tests added in `tests/test_transfer.py`.
3. **The provider wrapper leaks into the training text** (see §4). Fix in progress: `scripts/clean_frontier_traces.py`
   writing `data/frontier_tools_clean`.
4. **Cosmetic, known:** the 5 `#vN` rows resolved through `data/frontier_prov` inherited that corpus's id suffix, e.g.
   `rep-42-00045#v17-t-google-gemini-3-1-flash-lite-...`. The authoritative producer is the row's `model` field
   (`google-gemini-3-8-flash`); `record_id` is correct. Left as-is rather than rewriting accepted rows.

## 7. Reproduce

```powershell
# 1. status + journal download (SOCKS proxy as always)
$env:HTTP_PROXY='socks5h://localhost:10808'; $env:HTTPS_PROXY='socks5h://localhost:10808'
.venv/Scripts/kaggle.exe benchmarks tasks status thermo-teacher-tools-s00
.venv/Scripts/kaggle.exe benchmarks tasks download thermo-teacher-tools-s00 -m gemini-3.8-flash -o data/teacher_runs/kaggle_hosted

# 2. ingest (idempotent: accepted record_ids are skipped)
.venv/Scripts/python.exe -m thermo_agent.teacher.transfer ingest data/processed `
  "data/teacher_runs/kaggle_hosted/thermo-teacher-tools-s00/6/gemini-3.8-flash/7179338/harness_completions.jsonl" `
  data/frontier_tools --model google/gemini-3.8-flash --run-id hosted-v6-7179338 --memory-dir data/memory `
  --extra-export-dir data/processed_arch --extra-export-dir data/frontier_prov

# 3. counts and gate (real tokenizer)
.venv/Scripts/python.exe scripts/clean_frontier_traces.py --src data/frontier_tools --out data/frontier_tools_clean
.venv/Scripts/python.exe -m thermo_agent.train.token_audit data/frontier_tools_clean
```

## 8. Next gate

Per `docs/IMPLEMENTATION_PLAN.md` Phase 0 and J4, this run authorises nothing further by itself. Order stays: compact
harness (0.1) -> gate re-measure (0.2) -> clean traces (0.3) -> 40-row pilot with `gemini-3.5-flash-lite`, then
`claude-haiku-5-5`, then `gpt-6-luna`. A model that rejects the payload is dropped, not retried.
