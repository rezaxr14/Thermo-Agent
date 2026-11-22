# RUNBOOK - reproducible commands

## Current truth (2026-11-22)

All commands run from the repo root with `.venv/Scripts/python.exe`. Set
`$env:PYTHONIOENCODING='utf-8'` on Windows. For kbench test collection set
`$env:LLM_DEFAULT='gpt-6-luna'`; this is only an import sentinel for
`kaggle_benchmarks`, not a model call.

## Tests

```powershell
`$env:LLM_DEFAULT='gpt-6-luna'`
.venv/Scripts/python.exe -m pytest -q
```

Last result: **1,172 passed, 9 skipped**. Focused protocol checks:
`.venv/Scripts/python.exe -m pytest tests/test_tool_protocol.py -q`.

## Corpus rebuild

```powershell
.venv/Scripts/python.exe -m scripts.build_tool_corpus `
  --src data/processed --out data/sft_tools_v3 --grpo-out data/grpo_tools_v3 `
  --tasks repair,design,audit,ledger --split train --include-frontier `
  --max-tokens 8192 --max-tool-result-chars 4500 --heldout data/heldout `
  --prompt-variant harness_tools --plants-per-task 600 --plant-seed 0
```

Run the same command with `--prompt-variant plain` to rebuild the plain tool corpora.
Then mix:

```powershell
.venv/Scripts/python.exe -m scripts.mix_tool_corpora `
  --deterministic-sft data/sft_tools_v3_h --deterministic-grpo data/grpo_tools_v3_h `
  --frontier data/frontier_tools_accepted --a-slice data/a_slice `
  --out-sft data/sft_mix_v3 --out-grpo data/grpo_mix_v3 `
  --harness-prompt --seed 0 --force --manifest data/sft_mix_v3/MANIFEST.json
```

## Audits and stats

```powershell
.venv/Scripts/python.exe -m scripts.audit_harness_corpus `
  --dirs data/sft_tools_v3_h data/grpo_tools_v3_h data/sft_mix_v3 data/grpo_mix_v3 `
  --out reports/harness_corpus_audit_native.json
.venv/Scripts/python.exe scripts/audit_provenance_chain.py --strict
.venv/Scripts/python.exe scripts/audit_prompt_separation.py
.venv/Scripts/python.exe -m scripts.dataset_stats --dirs <all live dirs> `
  --out reports/dataset_stats.json --markdown reports/dataset_stats.md
```

`dataset_stats` is run over: `data/a_slice`, `data/frontier_export`,
`data/frontier_tools_cohort_prompts`, `data/grpo_all`, `data/grpo_mix_v3`,
`data/grpo_tools_v3`, `data/grpo_tools_v3_h`, `data/heldout_v2`, `data/processed`,
`data/sft_all`, `data/sft_mix_v3`, `data/sft_tools_v3`, `data/sft_tools_v3_h`.

## Trace reviews

```powershell
.venv/Scripts/python.exe scripts/review_traces.py --limit 10
```

Writes one record per line to `reports/trace_review.jsonl`.

## Hosted pilot (not currently connectable)

The 40-row `native_pilot` shards are already built. Preflight and rehearsal commands:

```powershell
.venv/Scripts/python.exe -m scripts.preflight_budget_batch --model gpt-6-luna
.venv/Scripts/python.exe -m scripts.rehearse_budget_batch --rows 40
```

The Kaggle bridge kernel push currently fails with `ConnectionAbortedError(10053)`.
Do not substitute a different model or launch a 100-row batch.
