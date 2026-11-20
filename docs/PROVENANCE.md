# Per-trace provenance

This document describes the per-trace provenance ledger for thermo_agent. The
ledger is **per reasoning trace, not per corpus row**: a synthetic ground-truth
record, a hosted-model answer, and a tool-harness answer each get their own
record, and every downstream use is attached to that record.

## Per-row provenance (2026-11-19) — hand-maintained section

> This section is **maintained by hand** and is not part of the generated tables
> below (those are rewritten wholesale by `scripts/build_provenance_ledger.py`).
> It records the provenance carried *inside each corpus row*.

Every datagen/training row states in the row itself whether it is deterministic
or AI-generated and, if AI, which model produced it. The fields are:

- `source` — `ground_truth` (deterministic ground truth), `deterministic_tools`
  (deterministic tool trace), or a frontier value. Stored frontier clean rows
  carry `source: "frontier:<model>"` together with
  `source_detail: "frontier_tools"`.
- `provenance` — for deterministic rows a dict
  `{generator, variant, source_row_id, source_dir}`, e.g. generator
  `deterministic_tools`, variant `direct` / `recover`, `source_row_id` = the
  ground-truth id, `source_dir = "data/processed"`. **Frontier rows do not carry
  this dict**; they carry `model` (plus `provider`, `run_id`, `source_detail`).
- `model` — present on AI-generated rows only; the exact hosted model id. This
  is the authoritative "which AI model" field.
- `paraphrase` — A-slice rows carry `{variant, n}` and `prompt_variant: plain`.
  The A-slice is a **copy** of deterministic ground-truth rows, so it is not AI.
- Multi-turn tool traces are not single-turn answers: in `data/sft_tools` the
  whole `tool_call` / `tool_result` round sequence is stored in one SFT
  `completion` (a JSON list of assistant turns); in `data/grpo_tools` the same
  rounds are in `trajectory.turns` (with `trajectory.sft_completion`).

### Source-family table (counts where known)

| Source family | Producer | Model (AI rows) | Where stored | Rows |
| --- | --- | --- | --- | ---: |
| Deterministic ground truth | synth generators + simulator/verifiers | none (deterministic) | `data/processed` (plus `processed_arch`, `processed_big`) | **27,418** across all splits; train split: repair 5,458 / ledger 4,846 / audit 2,192 / design 1,768 |
| Plain A-slice (COPY of ground truth) | `build_a_slice.py`, seeded, 1 paraphrase per plant | none (deterministic copy) | `data/a_slice` | **1,600** (400/task) = round(4,800 × 0.3333) |
| Deterministic tool traces | `synth.tool_traces`, 600 plants × 2 variants (`direct`, `recover`) × 4 tasks | none (deterministic) | `data/sft_tools_v2*`, `data/grpo_tools_v2*` (and `*_h`, `sft_mix_v2`) | **DONE: 4,800 per task family**, 100% verifier-accepted, real tool results, equal-thirds tiers |
| Frontier harness CoT (AI) | generic Kaggle runner, verifier-accepted | `gpt-6-luna`, `claude-haiku-5-5` | per-model frontier trace dirs fed from the HF corpus repo | **IN PROGRESS**: each model's 5-row proof (5/5 accepted) and one 100-row batch completed; batch journals at gate/ingest/audit. Target **thousands**, no cap, equal thirds. |

**Frontier CoT provenance (current campaign, DECISIONS §6.1).** Only `gpt-6-luna` and
`claude-haiku-5-5` are authorized. Each AI row carries the exact `model` id,
`provider`, `run_id`, tool turns and the per-row deterministic-vs-AI tag. The old
72-cell six-model premium target is **discarded**. Premium models
(`gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default`, `gemini-4-argon`)
and `gemini-3.8-flash` / `gemini-3.5-flash-lite` are **not to be run**; no
traces, costs or scores are claimed for them. The small set of legacy pilot rows
recorded in the generated ledger below comes from a superseded early campaign and
is not part of the current authorized frontier campaign.

Nothing in `data/processed` was deleted or overwritten; the A-slice is an
additional copy. Deterministic rows are never labelled as AI; AI rows are never
labelled as deterministic.

**Data flow and memory (DECISIONS §7.8–§7.11).** Every file that touches Kaggle goes
through the private HF corpus repo first: **local → `Reza-Nadimi/thermo-agent-corpora`
→ Kaggle kernel → ONE Kaggle dataset (`thermo-agent-corpus`)**; no local data uploads
to Kaggle; credentials only in the repository-root `.env`; one generic reusable
Kaggle notebook for all workflows. RAG memory records **short success/failure
recaps** (what failed → what works); model ids, timestamps, cost and raw checker
logs are stripped before a note is served to a prompt.

### 2026-11-20 — provenance audit OK (8/8) and current cohort counts

`scripts/audit_provenance_chain.py --strict` reports **PROVENANCE OK: 8/8 checks pass**.
Findings:

- **237 accepted frontier cohort rows** carry `model`, `provider`, `run_id` and `source`
  (`source_detail: frontier_tools`).
- Prompt shards are clean — no answers or scoring reference leaked into tracegen prompts.
- No harness bookkeeping (ids, manifests, run ids) leaks into model-visible text.
- Memory has both accepted and rejected notes (short success/failure recaps only).
- **320 frontier mix rows traced**: 258 cohort rows + 62 documented legacy gemini rows.
- **0 `ref_spec` leaks in 11,520 mix rows** — the scoring-only `ref_*` fields (`ref_fuel_w`, `ref_spec`,
  `ref_evals`) were stripped from `sft_mix_v2` / `grpo_mix_v2` training rows and are read at scoring time from
  `data/frontier_export/design_train.jsonl` for GRPO design rewards (**1,216/1,216 design rows resolved**).

This audit is read-only: all already captured CoTs and deterministic data are kept unchanged (DECISIONS §9);
no trace was bulk-rewritten. Per-trace audits are allowed only for a specific diagnosed defect.

> **Generated tables below are a historical ledger snapshot.** They are rewritten by
> `scripts/build_provenance_ledger.py` from the journals currently on disk and reflect
> the superseded early pilot; they are not a statement of the current authorized
> luna/haiku campaign, and no model arm has been trained or evaluated.

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

> **Regeneration caveat:** `render_docs()` currently rewrites this whole file
> from its template, so re-running the builder drops the hand-maintained
> "Per-row provenance" section above. That section is mirrored in
> `docs/dataset_cards.md`; restore it after any regeneration until the builder
> template is updated.

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

Generated from 38218 ledger records
(`sha256:0841cfbd1bddd4c3`, 60693168 bytes).

### Totals

| Metric | Value |
|---|---|
| Traces | 38218 |
| Downstream entries | 57514 |
| Journal occurrences | 5677 |
| Accepted / rejected / unknown | 635 / 429 / 37154 |
| Acceptance rate of known traces | 59.6805% |
| Total cost (nanodollars) | 3152867300 |
| Total cost (USD) | 3.152867 |
| Nanodollars per accepted trace | 4965145.35 |

### Coverage by kind

| Kind | Traces | With downstream | Coverage |
|---|---|---|---|
| adversarial | 402 | 0 | 0.0% |
| frontier | 3512 | 1376 | 39.18% |
| harness | 1148 | 120 | 10.453% |
| synthetic | 33156 | 27767 | 83.7465% |

### Coverage by first-seen source

| Source | Traces | With downstream | Coverage |
|---|---|---|---|
| frontier | 209 | 112 | 53.5885% |
| memory | 402 | 0 | 0.0% |
| processed | 27418 | 23540 | 85.856% |
| processed_arch | 5544 | 4227 | 76.2446% |
| processed_big | 194 | 0 | 0.0% |
| teacher_runs | 4451 | 1384 | 31.0941% |

### By model

| Model | Traces | Accepted | Rejected | Acc. rate (known) | Cost (n$) | Input tok. | Output tok. | Cost/accepted (n$) |
|---|---|---|---|---|---|---|---|---|
| (adversarial/memory) | 402 | 0 | 396 | 0.0% | 0 | 0 | 0 | n/a |
| (synthetic) | 33156 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| anthropic/claude-sonnet-5-5@default | 5 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| google/gemini-3.1-flash-lite-preview | 2460 | 513 | 0 | 100.0% | 1751449500 | 4501674 | 417354 | 3414131.58 |
| google/gemini-3.8-flash | 208 | 0 | 33 | 0.0% | 612315750 | 371051 | 89074 | n/a |
| gpt-6-luna | 2 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| openai/gpt-5.4-nano-2026-03-17 | 1985 | 122 | 0 | 100.0% | 789102050 | 2586850 | 299961 | 6468049.59 |

### By task

| Task | Traces | Accepted | Rejected | Cost (n$) |
|---|---|---|---|---|
| audit | 5616 | 0 | 127 | 1174276860 |
| design | 4987 | 140 | 179 | 509263030 |
| ledger | 12271 | 160 | 53 | 740150190 |
| repair | 15344 | 335 | 70 | 729177220 |

### By split

| Split | Traces |
|---|---|
| (none) | 3674 |
| test_id | 2375 |
| test_ood_climate | 675 |
| test_ood_fault | 7827 |
| test_ood_load | 471 |
| test_ood_scale | 520 |
| test_ood_topology | 2505 |
| train | 17966 |
| val | 2205 |

### Harness-mode shares

| Harness | Traces | Share |
|---|---|---|
| agents+skills | 652 | 1.706% |
| none | 37070 | 96.9962% |
| skills | 170 | 0.4448% |
| tools | 326 | 0.853% |

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
| data/memory/frontier_notes.jsonl | memory_note | 195 | 195 |
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

## 2026-11-19 — provenance ledger re-run

Auditor re-ran the builder exactly as documented; no code/data was hand-edited:
`$env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe scripts/build_provenance_ledger.py 2>&1 | Select-Object -Last 40` → exit code 0.
Script takes no CLI args (no argparse in `main()`); docstring documents three derived outputs.
Outputs rewritten: `data/provenance/trace_ledger.jsonl`, `data/provenance/stats.json`, `docs/PROVENANCE.md`.
Builder stdout ended with: `wrote 38218 provenance records` (plus stats + docs lines).

Counts (regenerated `stats.json`, cross-checked against the ledger):
| Metric | Count |
|---|---|
| Total records | 38218 |
| by_kind | synthetic 33156; frontier 3512; harness 1148; adversarial 402 |
| by_first_seen_source | processed 27418; processed_arch 5544; teacher_runs 4451; memory 402; frontier 209; processed_big 194 |
| from `data/memory/*.jsonl` | 402 (agent_patterns 14; adversarial_repair 24; frontier_notes 364) |
| from `data/frontier/*.jsonl` | 209 (audit 8; design 43; ledger 93; repair 65) |

Seed notes: all 14 `agent_patterns.jsonl` rows appear as `kind=adversarial`, with `acceptance.source=seed:method` and `origin.source=seed:method` (14/14) and `record_id` empty (14/14); status is rejected 8 / unknown 6 / accepted 0. The 635 verifier-accepted rows all come from `data/frontier/*.jsonl`; 0 carry a `seed:method` source. Seed notes are therefore labelled as non-verifier evidence, not as verifier-accepted rows.

Secret scan of all 3 generated files: `KAGGLE_API_TOKEN` 0, `HF_TOKEN` 0, `sk-` 0, `C:\Users` 0, known secret-prefix tokens 0. Generic high-entropy regex: 10 matches in 5 `error` fields, all Anthropic API `request_id` values inside 400 BadRequestError messages (values redacted at inspection; provider request IDs, not credentials). Verdict: no secret or absolute local path leaked.
