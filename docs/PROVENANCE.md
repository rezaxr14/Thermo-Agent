# PROVENANCE - row labels and chain checks

## Current truth (2026-11-24)

Every training row carries a machine-readable `provenance` dict and a human `source`
label. `source_family` is `deterministic` or `ai_generated`. AI rows always name the
exact model; `frontier` alone is never a model name.

## Required fields

- `source`: `deterministic_tools`, `frontier_tools` or a ground-truth source.
- `source_model`: exact slug for AI rows, `null` otherwise.
- `provenance.generator`, `provenance.source_family`, `provenance.model`,
  `provenance.provider`, and `provenance.variant`.
- `provenance.run_id` when the batch has one; it is copied from the journal and never
  read from credentials.
- For tool rows: `provenance.tool_calls`, `source_row_id` and `source_dir`.

## Legacy frontier traces are archived, not migrated

Pre-existing frontier traces used `<tool_call name=...>` / `<tool_result>` text. They are
not a compatibility format. The 11 legacy accepted rows and the 12 legacy source rows
were moved to `data/archive/frontier_tools_legacy_2026-11-24/` with a SHA-256 manifest;
`data/archive/` is outside every training corpus. `build_accepted_frontier.py` and
`mix_tool_corpora.py` now reject any row that still carries the retired syntax.

Hosted journals store tool results as Python `repr(ToolInvocationResult(...))`. The
converter in `tool_protocol.tool_result_output` extracts only `output=` JSON, and
`normalize_trajectory` drops `call_id` and the teacher `reason` argument while rebuilding
`completion` as assistant/tool messages plus a native `sft_completion`.

## Answer-field hygiene

Scoring-only columns (`ref_spec`, `ref_evals`, `ref_fuel_w`, `ed_json`, verdicts) stay in
row metadata for rewards. They are never rendered into a prompt, never quoted as review
evidence, and are scrubbed at mix write time. The provenance audit checks this.

## Trace ledger

Generated from 38218 ledger records. This is the same total as
`data/provenance/trace_ledger.jsonl`.

| Item | Count |
| --- | ---: |
| Traces | 38218 |

## Dataset cards

### Repair

Columns: `id` `split` `topology` `fault_type` `prompt` `completion` `plant_broken` `plant_original` `site` `deducible` `min_edits` `detected`.

### Design

Columns: `id` `split` `prompt` `completion` `site` `ref_fuel_w` `ref_spec` `n_valid` `ref_evals`.

### Audit

Columns: `id` `split` `topology` `prompt` `completion` `plant` `site` `ed_json` `fuel_w` `net_w` `best_gain_w` `n_candidates`.

### Ledger

Columns: `id` `split` `topology` `fault_types` `prompt` `completion` `ledger` `min_edits`.

## Chain audit

`python scripts/audit_provenance_chain.py --strict` reports 9 checks: cohort fields,
frontier source mapping, recomputed tiers, prompt-shard leakage, harness leakage,
memory outcomes/serving, mix propagation, mix answer-field leakage, and protocol
markers. The marker check fails if `call_id`, `ToolInvocationResult`, or
`<tool_call name=` appears in accepted, both mixes, or any `sft_tools*` /
`grpo_tools*` directory. Before the rebuild it failed with 19 leaks; current result:
**PROVENANCE OK, 9/9**. The two accepted-model slugs are `gpt-6-luna` and
`claude-haiku-5-5`; no other model may appear in a new AI row.
