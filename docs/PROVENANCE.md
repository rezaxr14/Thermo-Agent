# Per-trace provenance

This document describes the per-trace provenance ledger for thermo_agent. The
ledger is **per reasoning trace, not per corpus row**: a synthetic ground-truth
record, a hosted-model answer, and a tool-harness answer each get their own
record, and every downstream use is attached to that record.

## Regenerate

```bash
python scripts/build_provenance_ledger.py
```

The script streams the source files (it loads no model and imports no training
code) and rewrites three derived artifacts:

| Output | Contents |
| --- | --- |
| `data/provenance/trace_ledger.jsonl` | one JSON object per canonical trace |
| `data/provenance/stats.json` | exact counts computed from the ledger records |
| `docs/PROVENANCE.md` | this file, including the generated stats tables |

`scripts/build_corpus.ps1` calls the script automatically after the swapped
corpora have been verified. All three outputs are fully derived and can be
deleted and regenerated at any time; no source data is modified. The large
`trace_ledger.jsonl` is gitignored (like its source corpora); `stats.json` and
this document are committed so the exact numbers remain reviewable.

## Canonical id and correlation

`trace_id` is `<base id>` plus `@m=<model>` when the trace came from a hosted
model plus `@h=<harness>` when the answer was produced through a non-default
harness (`tools`, `skills`, `agents+skills`). The model suffix is always kept as
provenance. `#vN` paraphrase markers are **stripped**: they are listed in
`paraphrase_variants` on the trace they rewrite, so 20 paraphrases of one repair
trace are one ledger record with 20 variants, not 20 traces. Correlation with
SFT, GRPO, paper and eval rows is done exclusively through these canonical ids
and the recorded `base_id`, never by counting rows.

When the same answer is exported to several journals (for example a probe and a
full run), it is still one trace: `journal` holds the first path + line and
`journal_occurrences` holds every path + line with its own hash, token count and
cost. Cost and token fields on the record come from the first occurrence, so
re-exports never double-count spend.

## Ledger record schema

| Field | Meaning |
| --- | --- |
| `trace_id` | canonical trace id (see above) |
| `base_id` | source record id with `#vN`, model and harness suffixes removed; this is the correlation key shared with `memory.record_id`, paper examples and eval replay rows |
| `kind` | `synthetic` (datagen), `frontier` (accepted/hosted answer), `harness` (tool/skill run), `adversarial` (rejected memory attempt) |
| `task`, `split` | task and split, inferred from the source path/row and id prefix |
| `topology`, `site_ref`, `topology_site_ref` | plant topology and/or a stable `sha256:` reference to the site payload |
| `origin` | generator + seed from `meta_*.json`, or the exact model id (with slug and journal path) |
| `journal` | `{"path", "line"}` of the first journal occurrence (null for synthetic traces) |
| `journal_occurrences` | every journal hit: path, line, timestamp, completion hash + length, finish/error, tokens, cost, harness |
| `harness` | `none`, `tools`, `skills`, `agents+skills`, … |
| `paraphrase_variants` | sorted `#vN` tokens observed for this trace |
| `cost_nanodollars`, `input_tokens`, `output_tokens` | usage from the first journal occurrence |
| `completion_sha256`, `completion_length`, `finish_reason`, `error`, `tool_limit_exhausted` | completion fingerprint and outcome |
| `acceptance` | `status` (`accepted`/`rejected`/`unknown`), verdict `reason`, `reward`, `reasons`, `source`; rejection reasons where present |
| `downstream` | structured list of `{artifact, kind, count, row_ids, harness_modes, paraphrase_variants}` plus `arm`/`replay_file` for eval arms |
| `first_seen` | `{source, path, line, timestamp}` where the trace was first observed |
| `notes`, `attributes` | human-readable provenance notes and auxiliary source fields |

`acceptance` records an actual verdict: frontier validation verdicts (accepted),
memory-note rejections (rejected), and no verdict at all (unknown). Synthetic
ground-truth traces have no teacher verdict, so they are `unknown` even though
they are valid training targets; do not read `unknown` as a failure.

## Adding a new model journal

No code edit is needed. Drop the journal into `data/teacher_runs/` as JSONL with
one answer per line (`id`, `model`, `harness`, `usage`, `completion`, optional
`ts`), then re-run `python scripts/build_provenance_ledger.py`. The script
auto-discovers every `*.jsonl` in that directory, slugifies the exact `model`
id, and matches it to the `-t-<model>` suffix used in `data/frontier`,
`data/sft` and `data/grpo`. A new harness mode automatically becomes a new
`@h=<mode>` trace id. Eval arms are attributed through each report's
`meta.replay_file`, so a replay journal for a future model also needs no code
change.

Old Gemini (`google/gemini-3.1-flash-lite-preview`) and GPT-nano
(`openai/gpt-5.4-nano-2026-03-17`) traces remain in the ledger unchanged.
Future bigger-model traces are **appended additively**: re-running the builder
adds their records and downstream links without rewriting the existing ones
(other than refreshing the regenerated tables below).

## Current exact stats (generated from `data/provenance/stats.json`)

Generated from 37945 ledger records
(`sha256:23b644365ecb7152`, 60253023 bytes).

### Totals

| Metric | Value |
|---|---|
| Traces | 37945 |
| Downstream entries | 57488 |
| Journal occurrences | 5473 |
| Accepted / rejected / unknown | 635 / 339 / 36971 |
| Acceptance rate of known traces | 65.1951% |
| Total cost (nanodollars) | 2540551550 |
| Total cost (USD) | 2.540552 |
| Nanodollars per accepted trace | 4000868.58 |

### Coverage by kind

| Kind | Traces | With downstream | Coverage |
|---|---|---|---|
| adversarial | 339 | 0 | 0.0% |
| frontier | 3339 | 1350 | 40.4313% |
| harness | 1111 | 120 | 10.8011% |
| synthetic | 33156 | 27767 | 83.7465% |

### Coverage by first-seen source

| Source | Traces | With downstream | Coverage |
|---|---|---|---|
| frontier | 201 | 112 | 55.7214% |
| memory | 339 | 0 | 0.0% |
| processed | 27418 | 23540 | 85.856% |
| processed_arch | 5544 | 4227 | 76.2446% |
| processed_big | 194 | 0 | 0.0% |
| teacher_runs | 4249 | 1358 | 31.9605% |

### By model

| Model | Traces | Accepted | Rejected | Acc. rate (known) | Cost (n$) | Input tok. | Output tok. | Cost/accepted (n$) |
|---|---|---|---|---|---|---|---|---|
| (adversarial/memory) | 339 | 0 | 339 | 0.0% | 0 | 0 | 0 | n/a |
| (synthetic) | 33156 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| anthropic/claude-sonnet-5-5@default | 5 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| google/gemini-3.1-flash-lite-preview | 2460 | 513 | 0 | 100.0% | 1751449500 | 4501674 | 417354 | 3414131.58 |
| openai/gpt-5.4-nano-2026-03-17 | 1985 | 122 | 0 | 100.0% | 789102050 | 2586850 | 299961 | 6468049.59 |

### By task

| Task | Traces | Accepted | Rejected | Cost (n$) |
|---|---|---|---|---|
| audit | 5356 | 0 | 43 | 561961110 |
| design | 4983 | 140 | 177 | 509263030 |
| ledger | 12268 | 160 | 51 | 740150190 |
| repair | 15338 | 335 | 68 | 729177220 |

### By split

| Split | Traces |
|---|---|
| (none) | 3434 |
| test_id | 2375 |
| test_ood_climate | 675 |
| test_ood_fault | 7827 |
| test_ood_load | 471 |
| test_ood_scale | 520 |
| test_ood_topology | 2505 |
| train | 17933 |
| val | 2205 |

### Harness-mode shares

| Harness | Traces | Share |
|---|---|---|
| agents+skills | 652 | 1.7183% |
| none | 36834 | 97.0721% |
| skills | 135 | 0.3558% |
| tools | 324 | 0.8539% |

### Paraphrase expansion

| Metric | Value |
|---|---|
| Traces with variants | 11644 |
| Variant tokens total | 134178 |
| Mean variants per trace | 11.5234 |
| Downstream rows with a variant | 117276 |

By task: {'repair': 117276}

### Downstream artifacts

| Artifact | Kind | Unique traces | Rows |
|---|---|---|---|
| data/grpo/audit_test_id.jsonl | grpo_row | 321 | 321 |
| data/grpo/audit_test_ood_topology.jsonl | grpo_row | 238 | 238 |
| data/grpo/audit_train.jsonl | grpo_row | 2564 | 2564 |
| data/grpo/audit_val.jsonl | grpo_row | 338 | 338 |
| data/grpo/design_test_id.jsonl | grpo_row | 204 | 204 |
| data/grpo/design_test_ood_climate.jsonl | grpo_row | 615 | 615 |
| data/grpo/design_test_ood_load.jsonl | grpo_row | 411 | 411 |
| data/grpo/design_test_ood_scale.jsonl | grpo_row | 460 | 460 |
| data/grpo/design_train.jsonl | grpo_row | 1894 | 1894 |
| data/grpo/design_val.jsonl | grpo_row | 235 | 235 |
| data/grpo/ledger_test_id.jsonl | grpo_row | 656 | 656 |
| data/grpo/ledger_test_ood_fault.jsonl | grpo_row | 2360 | 2360 |
| data/grpo/ledger_test_ood_topology.jsonl | grpo_row | 854 | 854 |
| data/grpo/ledger_train.jsonl | grpo_row | 5165 | 5165 |
| data/grpo/ledger_val.jsonl | grpo_row | 666 | 666 |
| data/grpo/repair_test_id.jsonl | grpo_row | 563 | 563 |
| data/grpo/repair_test_ood_fault.jsonl | grpo_row | 3966 | 3966 |
| data/grpo/repair_test_ood_topology.jsonl | grpo_row | 864 | 864 |
| data/grpo/repair_train.jsonl | grpo_row | 5226 | 5226 |
| data/grpo/repair_val.jsonl | grpo_row | 614 | 614 |
| data/memory/adversarial_repair.jsonl | memory_note | 19 | 24 |
| data/memory/frontier_notes.jsonl | memory_note | 169 | 169 |
| data/sft/audit_test_id.jsonl | sft_row | 321 | 321 |
| data/sft/audit_test_ood_topology.jsonl | sft_row | 238 | 238 |
| data/sft/audit_train.jsonl | sft_row | 2564 | 2564 |
| data/sft/audit_val.jsonl | sft_row | 338 | 338 |
| data/sft/design_test_id.jsonl | sft_row | 204 | 204 |
| data/sft/design_test_ood_climate.jsonl | sft_row | 615 | 615 |
| data/sft/design_test_ood_load.jsonl | sft_row | 411 | 411 |
| data/sft/design_test_ood_scale.jsonl | sft_row | 460 | 460 |
| data/sft/design_train.jsonl | sft_row | 1894 | 1894 |
| data/sft/design_val.jsonl | sft_row | 235 | 235 |
| data/sft/ledger_test_id.jsonl | sft_row | 656 | 656 |
| data/sft/ledger_test_ood_fault.jsonl | sft_row | 2360 | 2360 |
| data/sft/ledger_test_ood_topology.jsonl | sft_row | 854 | 854 |
| data/sft/ledger_train.jsonl | sft_row | 5165 | 5165 |
| data/sft/ledger_val.jsonl | sft_row | 666 | 666 |
| data/sft/repair_test_id.jsonl | sft_row | 563 | 563 |
| data/sft/repair_test_ood_fault.jsonl | sft_row | 3966 | 3966 |
| data/sft/repair_test_ood_topology.jsonl | sft_row | 864 | 864 |
| data/sft/repair_train.jsonl | sft_row | 5226 | 100036 |
| data/sft/repair_val.jsonl | sft_row | 614 | 614 |
| paper/assets/examples.json | paper_example | 32 | 33 |
| reports/eval_frontier_tools.json | eval_arm | 120 | 120 |
| reports/eval_frontier_zero.json | eval_arm | 360 | 360 |
| reports/eval_other_model.json | eval_arm | 360 | 360 |

### Aggregate memory rejection reasons

`data/memory/failure_reasons.json` has no per-trace ids, so it is reported here
and in `stats.json` only (it is never turned into synthetic ledger records).

Total recorded failures: **0**.
By task: {}.
By fault type: {}.

| Top lesson | Count |
|---|---|
