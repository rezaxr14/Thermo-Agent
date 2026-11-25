# Step 3 - row retention at the exact 8,192 gate

Date: 2026-11-25. Scope: deterministic corpora (`data/sft_tools_v3*`) and the accepted
frontier cohort (`data/frontier_tools_accepted`). No training was run.

## What was wrong

Three separate losses stacked on deterministic rows:

1. The gate was measured on the teacher export prompt (`build_context`) while training
   renders the compact harness prompt (`build_harness_tool_messages`) through
   `apply_chat_template(tools=...)`. Two different sequences were being measured.
2. Tool schemas were the long teacher docstrings, and `load_skill` served the full
   `AGENTS.md` body; both are injected into every prompt and pushed rows over the gate.
3. Tool results repeated every column name on every row (`{"stream": ...}` lists), so
   `solve_plant` and the ledger residual tables were several hundred tokens larger than
   they needed to be.

## The gate, as implemented now

The gate measures exactly the text training tokenizes (`thermo_agent/train/render.py`
`render_pair`): `prompt + completion`, rendered in one `apply_chat_template` call with
the same 16 schemas that `row_tools()` injects, plus the 96-token chat reserve. Rows over
the gate are dropped; nothing is truncated.

Schemas are the serving contract (`tool_schemas(task)`, 16 tools, identical for all
tasks): **1,247 tokens** after shortening the docstrings, down from 1,418. The
row-aware long descriptions are not used by training, eval, or the gate.

## Deterministic retention per task and tier

Attempted = 600 plants x 2 variants (direct, recover) = 1,200 rows/task, 400/tier.

| Task | Attempted | Kept | Retention | Dropped tier(s) |
| --- | ---: | ---: | ---: | --- |
| repair | 1,200 | 1,200 | 100.0% | - |
| design | 1,200 | 1,200 | 100.0% | - |
| audit | 1,200 | 1,189 | 99.08% | hard 389/400 (97.25%) |
| ledger | 1,200 | 1,191 | 99.25% | hard 391/400 (97.75%) |
| **total** | **4,800** | **4,780** | **99.58%** | hard only |

Every task and every tier is at or above 97.25%; the >=90% target is met with margin.
All 20 dropped rows are hard-tier and are genuine over-gate rows (8.2k-8.8k training
tokens), not a gate mismatch. Tier labels here come from the full 1,200-row pool, before
drops, so the denominators are stable:

- audit (11, all hard): `aud-0-00283`, `00335`, `00567`, `00589`, `01138`, `01845`,
  `01966`, `02369`, `02448`, `02538`, `02774` (all `-dt-recover-c3c2`).
- ledger (9): `led-0-01930` (direct + recover), `03649` (direct + recover),
  `04718` (recover), `04932` (recover), `05661` (recover), `06790` (direct + recover).

## Accepted frontier rows at the gate

Measured through the same path the mix uses (`to_harness_prompt` -> training render with
serving schemas). Reserve included.

| Task | Accepted rows | Over 8,192 | Max tokens |
| --- | ---: | ---: | ---: |
| repair | 17 | 0 | 7,286 |
| design | 6 | 0 | 5,208 |
| ledger | 7 | 0 | 7,369 |
| audit | 0 | 0 | - |
| **total** | **30** | **0** | **7,369** |

Before the tool-output compaction and skill-budget change, `led-0-00362-t-gpt-6-luna`
measured 8,230 and was the only accepted row over the gate. It now measures 7,369.

## What changed to get there

- `thermo_agent/teacher/toolkit.py`: `solve_plant` returns a column/row table
  (`columns` + `rows`, 8 fields) instead of one dict per stream; ledger
  `inspect_plant`/`first_law_check` return `line_table` + `line_errors` instead of
  `per_line_residuals`; `compact_skill_body` serves the priority sections only.
- `thermo_agent/agent/tool_protocol.py`: tool descriptions truncate at the first
  sentence; stored results are re-served through the same compaction
  (`normalize_tool_turns` -> `compact_result`), and skill results are refreshed.
- `thermo_agent/synth/tool_traces.py`: reasoning cites only the five worst residual
  lines plus a "N more within tolerance" count instead of the full table.
- `scripts/mix_tool_corpora.py`, `scripts/build_tool_corpus.py`,
  `scripts/frontier_cohort_tokengate.py`: one gate measure (training render) and one
  schema source (`tool_schemas(task)`), each covered by a failure test.
- `scripts/process_budget_batch.ps1`: ingest now reads `data/frontier_export`, where
  `arch-107-00019#v16` lives, so those rows can resolve.

## Reproduction

```powershell
.venv/Scripts/python.exe -m scripts.build_tool_corpus `
  --src data/processed --out data/sft_tools_v3 --grpo-out data/grpo_tools_v3 `
  --tasks repair,design,audit,ledger --split train --include-frontier `
  --max-tokens 8192 --max-tool-result-chars 4500 --heldout data/heldout `
  --prompt-variant harness_tools --plants-per-task 600 --plant-seed 0
.venv/Scripts/python.exe scripts/gate_retention_report.py `
  --corpus data/sft_tools_v3 --out reports/gate_retention_step3.json
.venv/Scripts/python.exe scripts/audit_harness_corpus.py `
  --dirs data/sft_tools_v3_h data/grpo_tools_v3_h data/sft_mix_v3 data/grpo_mix_v3 `
  --out reports/harness_corpus_audit_step3_after.json
```

Exact per-row rows (including every dropped id and token count) are in
`reports/gate_retention_step3.json`. The audit in
`reports/harness_corpus_audit_step3_after.json` reports 0 rows over 8,192 and 0
anomalies on the written corpora.
