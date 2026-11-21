# Data: provenance, protocol, corpora, dataset cards

Everything about what the data is and where it came from.

## Per-trace provenance

*(merged from `docs/PROVENANCE.md` on 2026-11-20; content kept, headings demoted one level)*

This document describes the per-trace provenance ledger for thermo_agent. The
ledger is **per reasoning trace, not per corpus row**: a synthetic ground-truth
record, a hosted-model answer, and a tool-harness answer each get their own
record, and every downstream use is attached to that record.

### Per-row provenance (2026-11-19) — hand-maintained section

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

#### Source-family table (counts where known)

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

#### 2026-11-20 — provenance audit OK (8/8) and current cohort counts

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

### Regenerate

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
> `docs/PROVENANCE.md`; restore it after any regeneration until the builder
> template is updated.

### Canonical id and correlation

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

### Ledger record schema

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

### Adding a new model journal

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

### Current exact stats (generated from `data/provenance/stats.json`)

Generated from 38218 ledger records
(`sha256:0841cfbd1bddd4c3`, 60693168 bytes).

#### Totals

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

#### Coverage by kind

| Kind | Traces | With downstream | Coverage |
|---|---|---|---|
| adversarial | 402 | 0 | 0.0% |
| frontier | 3512 | 1376 | 39.18% |
| harness | 1148 | 120 | 10.453% |
| synthetic | 33156 | 27767 | 83.7465% |

#### Coverage by first-seen source

| Source | Traces | With downstream | Coverage |
|---|---|---|---|
| frontier | 209 | 112 | 53.5885% |
| memory | 402 | 0 | 0.0% |
| processed | 27418 | 23540 | 85.856% |
| processed_arch | 5544 | 4227 | 76.2446% |
| processed_big | 194 | 0 | 0.0% |
| teacher_runs | 4451 | 1384 | 31.0941% |

#### By model

| Model | Traces | Accepted | Rejected | Acc. rate (known) | Cost (n$) | Input tok. | Output tok. | Cost/accepted (n$) |
|---|---|---|---|---|---|---|---|---|
| (adversarial/memory) | 402 | 0 | 396 | 0.0% | 0 | 0 | 0 | n/a |
| (synthetic) | 33156 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| anthropic/claude-sonnet-5-5@default | 5 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| google/gemini-3.1-flash-lite-preview | 2460 | 513 | 0 | 100.0% | 1751449500 | 4501674 | 417354 | 3414131.58 |
| google/gemini-3.8-flash | 208 | 0 | 33 | 0.0% | 612315750 | 371051 | 89074 | n/a |
| gpt-6-luna | 2 | 0 | 0 | 0.0% | 0 | 0 | 0 | n/a |
| openai/gpt-5.4-nano-2026-03-17 | 1985 | 122 | 0 | 100.0% | 789102050 | 2586850 | 299961 | 6468049.59 |

#### By task

| Task | Traces | Accepted | Rejected | Cost (n$) |
|---|---|---|---|---|
| audit | 5616 | 0 | 127 | 1174276860 |
| design | 4987 | 140 | 179 | 509263030 |
| ledger | 12271 | 160 | 53 | 740150190 |
| repair | 15344 | 335 | 70 | 729177220 |

#### By split

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

#### Harness-mode shares

| Harness | Traces | Share |
|---|---|---|
| agents+skills | 652 | 1.706% |
| none | 37070 | 96.9962% |
| skills | 170 | 0.4448% |
| tools | 326 | 0.853% |

#### Paraphrase expansion

| Metric | Value |
|---|---|
| Traces with variants | 11644 |
| Variant tokens total | 134178 |
| Mean variants per trace | 11.5234 |
| Downstream rows with a variant | 117276 |

By task: {'repair': 117276}

#### Downstream artifacts

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

#### Aggregate memory rejection reasons

`data/memory/failure_reasons.json` has no per-trace ids, so it is reported here
and in `stats.json` only (it is never turned into synthetic ledger records).

Total recorded failures: **0**.
By task: {}.
By fault type: {}.

| Top lesson | Count |
|---|---|

### 2026-11-19 — provenance ledger re-run

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

## Training-data protocol (user decision, 2026-11-18)

*(merged from `docs/PROVENANCE.md` on 2026-11-20; content kept, headings demoted one level)*

This is a binding decision for the remaining campaign. It supersedes the earlier zero-shot-only teacher pass for
*training data*. All old data is kept; everything below is additive.

### Why (measured gap)

- Accepted frontier CoTs to date: **635** (Gemini 3.1 Flash Lite 513, GPT-5.4-nano 122). All **635** survive the
  8,000-token prompt gate (`max_length` 8,192) and are present as frontier train rows in **both** merged corpora,
  `data/sft_all` and `data/grpo_all` (verified: 635 `source=frontier` rows in `data/sft_all/*_train.jsonl`). The old
  **460** figure was a legacy count: it applied only to the superseded 4,000-token-gated `data/sft` / `data/grpo`
  corpora, not to the merged ones. Additionally **33** from `gemini-3.8-flash` were ingested. Every one of these was
  produced with `harness="none"`, `tools="none"` — they are **zero-shot** answers, not tool-using reasoning.
- A separate tool arm (`frontier_tools`) recorded 120 answers / 242 generic answer-check calls, but those rows are an
  **evaluation** artifact (`reports/eval_frontier_tools.json`), not training data, and the only tool exposed was a
  generic answer checker — there were no first-law, second-law or validity tools.
- Therefore the model currently has **no verifier-tool-using traces to imitate**. This decision closes that gap.

### The decision, in order

1. **Frontier zero-shot baseline on a dedicated held-out test set.** Freeze a new test set that no cohort has seen:
   `data/heldout`, generated with the fresh ground-truth seed **987654**, excluded from every `data/processed*`,
   `data/sft*`, `data/grpo*` corpus and from every teacher prompt shard. Score frontier models zero-shot on it first;
   this is the *frontier baseline*. All comparison arms below are scored on this same `data/heldout` set — frontier
   zero-shot, frontier+harness+tools+skills+memory, other thermo LLMs, our SFT, and our SFT+GRPO with k=4, 8 and 16
   (see `docs/DECISIONS.md`, D8).
2. **Frontier models inside our harness produce CoT.** Re-run the frontier models with our physics tools exposed
   (`inspect_plant`, `first_law_check`, `second_law_check`, `validity_check`, `solve_plant`), keeping the reasoning trace
   and every tool call. These are the **tool-using CoT traces**.
3. **Train on those CoTs plus the deterministic synthetic generation.** The merged SFT corpus = synthetic ground
   truth (unchanged) + verifier-accepted zero-shot frontier rows (kept) + the new verifier-accepted tool-using CoT
   rows. GRPO prompts come from the same held-out ground truth. Nothing old is deleted.
4. **Evaluate on the same held-out test set and record our score.** This makes the baseline (step 1) and our model
   (step 4) directly comparable on identical, previously unseen rows.
5. **Compare against other thermo models on that set and record the result** (`other_model` arm and any domain LLM
   ids supplied later).

### Tool contract (what "uses tools" means)

The tools must be the real physics checks, exposed to the model during generation, with feedback returned to the
model so it can self-correct. From `thermo_agent/verify/verifiers.py`:

The exposed tool names are those in `thermo_agent/teacher/toolkit.py`:

| Tool | One-line contract (what it reports) |
|---|---|
| `inspect_plant` | solve / verify the plant or ledger AS GIVEN — no candidate argument |
| `first_law_check` | mass/energy residuals (kg/s, W), convergence, tear residual, unit/range flags |
| `second_law_check` | exergy destruction (MW), closure, non-negativity, Carnot/COP limits |
| `validity_check` | validity, failed checks, pinch approach, ranges |
| `solve_plant` | full stream table (T, P, h, s, exergy) |

Grading/acceptance (the full accept/reward/reasons verdict) is deliberately **not** exposed as a tool (P0-8): the model
receives only physics feedback and must decide for itself when its answer is acceptable.

A trace is "tool-using" only if it contains at least one of these calls **and** a subsequent revised final answer;
both are stored on the journal line (`tool_calls`, `reasoning`, `completion`).

### Generation invariants (P0-9)

- **One rounds constant.** `DEFAULT_MAX_ROUNDS = 4` in `thermo_agent/teacher/agent_graph.py` is the single source
  of truth. The "up to N rounds" sentence in the system prompt, the `run()` default, the `--rounds` default in
  `scripts/kaggle_harness_cli.py` and the `ROUNDS` environment default in `kaggle/teacher_harness.py` all use it.
  When a driver overrides the limit, it also passes `max_rounds` into `build_context`, so the prompt always
  states the same number the adapter actually enforces.
- **Memory retrieved once per episode.** `build_context` calls `mem.retrieve(...)` exactly once, with
  `exclude_record` set to the current row id, and uses that single result for both the served (compact) failure
  block and the journal's `notes_used`. A note mined from the current record is never shown to the model, and
  `notes_used` is exactly the set of notes rendered into the prompt — no duplicate serving, no phantom notes.
- **The native tool loop is the driver.** The graph is the state machine / verifier: it builds the context and
  records the verifier verdict and SFT trajectory. When the adapter returns a full 4-tuple trajectory the graph
  jumps straight to `finalize`; tools are executed exactly once, never re-run by the graph.

### Model selection for this round (user decision)

- Cheap models now, in order: `gpt-6-luna` then `claude-haiku-5-5`, reasoning ON, run as the hosted task
  `thermo-teacher-tools-s00` via `scripts/run_teacher_hosted.ps1 -m <slug>` (not the local SDK driver);
  `gemini-3.5-flash-lite` is excluded and must not consume quota. The expensive tier (`gpt-6.2-sol`,
  `claude-sonnet-5-5`, `claude-opus-5-5`, `gemini-4-argon`) follows only after the budget tier reports accepted
  reasoning traces, and `claude-sonnet-5-5` additionally needs Reza's explicit go-ahead.
- Reasoning **ON** (`reasoning="low"`). Capture reasoning where the provider exposes it; leaving it on where it cannot
  be captured is acceptable.
- **REMINDER:** revisit `claude-sonnet-5-5` later — deferred as too expensive; decide after the cheap models report
  how many accepted reasoning traces they yield.
- Add other models only after seeing the accepted-CoT yield of these three.

### Where this is implemented

- Hosted zero-shot / reasoning journals: `kaggle/teacher_big_task_s00.py`, `scripts/run_big_teacher_v2.ps1`.
- Tool-using CoT journals: `kaggle/teacher_harness.py` (physics tools + reasoning capture).
- Ingest is additive through `thermo_agent.teacher.transfer.ingest` (tool use is detected from the record itself;
  there is no `--harness` flag) — e.g. `python -m thermo_agent.teacher.ingest data/frontier_export
  harness_completions.jsonl data/frontier --model gpt-6-luna`. Provenance is recorded per trace in
  `data/provenance/trace_ledger.jsonl` with origin model, harness mode, tool calls, cost and acceptance.

### 2026-11-19 — tool-output masking coverage (sft_tools)

Audited all 960 rows of `data/sft_tools/*.jsonl` (4 tasks x 240). Counted `<tool_call` / `<tool_result` tags and
masked `<tool_result>...</tool_result>` spans via `thermo_agent.train.sft._tool_result_spans`. Must-be-zero checks
pass only when the count is 0.

| Task | n | `<tool_call` min/med/max | `<tool_result` min/med/max | span chars | compl chars | tool-output frac | <2 calls or <2 results | zero-unmasked rows | ends in tool block | `<tool_response>` rows |
|---|---|---|---|---|---|---|---|---|---|---|
| audit | 240 | 3/3.5/4 | 3/3.5/4 | 1,023,802 | 1,508,729 | 0.679 | 0 | 0 | 0 | 0 |
| design | 240 | 2/2.5/3 | 2/2.5/3 | 1,038,096 | 1,711,570 | 0.607 | 0 | 0 | 0 | 0 |
| ledger | 240 | 2/2.5/3 | 2/2.5/3 | 833,012 | 1,140,102 | 0.731 | 0 | 0 | 0 | 0 |
| repair | 240 | 3/3/3 | 3/3/3 | 604,696 | 917,647 | 0.659 | 0 | 0 | 0 | 0 |
| **ALL** | **960** | — | — | **3,499,606** | **5,278,048** | **0.663** | **0** | **0** | **0** | **0** |

Must-be-zero checks: `<2 calls or <2 results` **PASS (0)**, `zero-unmasked rows` **PASS (0)**,
`ends in tool block` **PASS (0)**. Corpus is XML `<tool_result>` only (0 `<tool_response>`). All 960 completions
start with `<think>`; 0 place a tool call before any reasoning. Prompt ground-truth key counts (`plant_original`,
`ref_spec`, `ed_json`, `fuel_w`, `net_w`) are all **0 (PASS)**. Verdict: training targets are only model-generated
reasoning, calls and answers; tool output is fully masked and every row is a genuine >=2-call multi-turn trace.

## Dataset cards

*(merged from `docs/PROVENANCE.md` on 2026-11-20; content kept, headings demoted one level)*

All four datasets (repair, ledger, audit, design) are synthetic: plants and sites are sampled by the code in this repository and labelled by the simulator and verifiers. They are not measurements. (The unrelated "Frontier energy dataset" can set site specifications through `thermo_agent.datasets.frontier`; no measured data is inside these files, and it is distinct from the *frontier tool traces* described below.) Files are written as `<task>_<split>.jsonl` plus `meta_<task>.json` (seed, settings, CoolProp and schema versions). The data files are not in the repository; generate them with the commands in `docs/RUNBOOK.md`. Licence: not yet chosen by the authors.

### Evaluation set and dev set (F1, 2026-11-19)

Per DECISIONS F1 (2026-11-19), which supersedes the earlier frozen protocol:
the **test set is `data/heldout_v2`** (new seed; manifest must be frozen before any model is scored on it),
and **`data/heldout` (214 `test_*` ids, seed 987654) is now the dev/tuning set** — it is never reported as the
headline test result. The older frozen protocol `reports/heldout_dev_protocol.md` was written while
`data/heldout` was still the evaluation set; its role as *test* is superseded by F1, while its cell counts below
remain an accurate description of the dev set. `data/heldout_v2` is **generated and FROZEN** (3,476 scored rows in
12 cells, equal thirds per task, contamination scan PASS with overlap 0; test hash `3cc51773…` recorded in
`docs/EVAL_PROTOCOL.md`). No arm has been scored on it.

Counts below (dev set `data/heldout`) were measured by counting non-empty lines in each `*_test_*.jsonl`, not copied
from prose; they agree with the `counts` in `meta_<task>.json`. **214 dev ids in 12 cells.**

| Task | `test_id` | `test_ood_fault` | `test_ood_topology` | `test_ood_climate` | `test_ood_load` | `test_ood_scale` | task total |
|---|---:|---:|---:|---:|---:|---:|---:|
| repair | 6 | 70 | 15 | — | — | — | 91 |
| audit | 16 | — | 6 | — | — | — | 22 |
| ledger | 11 | 32 | 7 | — | — | — | 50 |
| design | 4 | — | — | 20 | 13 | 14 | 51 |

Measured topology composition of the held-out `test_ood_topology` cells: repair 15 rows = `dc_engine_absorption-ext` 5
+ `dc_engine_electric-ext` 10; audit 6 rows = 1 + 5; ledger 7 rows = 2 + 5. This corrects the generic card wording
below ("steam-ORC plant and gas-engine plants"): the frozen held-out topology cells contain **only the two extended
gas-engine topologies**, no steam-ORC. Design rows have no `topology` field. Audit has no `test_ood_fault` cell.
The repair/ledger `test_ood_fault` cells mix the in-training topologies (repair: cgam_orc 4, dc_absorption 28,
dc_electric 28, dc_engine_absorption 6, dc_engine_electric 4; ledger: 3, 9, 16, 2, 2).

Every dev cell is below 85 rows (largest: repair `test_ood_fault`, n = 70), so no cell supports a 15-point pairwise
claim at 80% power; see `reports/heldout_dev_protocol.md` §6 for the computed minimum detectable differences and
the only defensible merges. This is precisely why F1 mandates `data/heldout_v2` with >= 85 rows per reported cell
where a generator can reach it, equal thirds per task (easy/medium/hard), and a frozen manifest before scoring.

### Data provenance and corpus status (2026-11-19)

This section is the stable mirror of the hand-maintained section in
`docs/PROVENANCE.md`. See that file for the per-row field definitions. Every row
states whether it is deterministic or AI-generated and, if AI, which model
produced it: `source` (`ground_truth` / `deterministic_tools` / frontier value
with `source_detail: frontier_tools`), `provenance` (deterministic rows:
`{generator, variant, source_row_id, source_dir}`), `model` (AI rows only), and
`paraphrase` (A-slice rows). Frontier rows do not carry a `provenance` dict;
they carry `model`, `provider`, `run_id` and `source_detail`.

| Source family | Producer | Model (AI rows) | Where stored | Rows |
| --- | --- | --- | --- | ---: |
| Deterministic ground truth | synth generators + simulator/verifiers; **nothing deleted or overwritten** | none (deterministic) | `data/processed` | **27,418** across all splits; train split repair 5,458 / ledger 4,846 / audit 2,192 / design 1,768 |
| Plain A-slice (COPY of ground truth) | `build_a_slice.py`, seeded, one paraphrase per plant, plain prompt, no tool markers | none (deterministic copy) | `data/a_slice` | **1,600** (400/task) = round(4,800 × 0.3333) |
| Deterministic tool traces | `synth.tool_traces`, 600 plants × 2 variants (`direct`, `recover`) × 4 tasks | none (deterministic) | `data/sft_tools_v2*`, `data/grpo_tools_v2*` (and `*_h`, `sft_mix_v2`) | **DONE: 4,800 per task family**, 100% verifier-accepted, real tool results, equal-thirds tiers |
| Frontier harness CoT (AI) | generic Kaggle runner, verifier-accepted | `gpt-6-luna`, `claude-haiku-5-5` | per-model frontier trace dirs fed from the HF corpus repo | **IN PROGRESS**: each model's 5-row proof (5/5 accepted) and one 100-row batch completed; journals at gate/ingest/audit. Target **thousands**, no cap, equal thirds. |

Notes on the two trace families:

- **Deterministic tool traces are multi-turn and complete.** In `data/sft_tools_v2*`
  the full `tool_call` / `tool_result` round sequence is stored in one SFT
  `completion` (a JSON list of assistant turns ending in the verified answer);
  in `data/grpo_tools_v2*` the same rounds are in `trajectory.turns`. The corpus is
  built at the decided 4,800 rows per task family = 600 plants/task × 2 variants ×
  4 tasks; tiers are equal thirds per task from `thermo_agent/synth/tiers.py`.
- **Frontier harness-CoT traces are being collected now.** Only `gpt-6-luna` and
  `claude-haiku-5-5` are authorized. Each accepted trace carries its exact `model`
  id, tool calls and verifier verdict; rows are gated at 8,192 tokens (over-gate
  rows dropped, never truncated). A small set of legacy pilot rows recorded in the
  provenance ledger comes from a superseded early campaign and is not part of the
  current authorized campaign.

**Frontier models NOT to be run:** `gpt-6.2-sol`, `claude-sonnet-5-5`,
`claude-opus-5-5-default`, `gemini-4-argon`, `gemini-3.8-flash` and
`gemini-3.5-flash-lite`. None was run; no traces, costs, acceptance counts or
scores exist for them. The old 72-cell premium target is discarded. The A-slice
and all deterministic corpora are AI-free. Data flows local → private HF repo
`Reza-Nadimi/thermo-agent-corpora` → Kaggle kernel → one Kaggle dataset; no local
uploads to Kaggle, credentials only in the repository-root `.env`, one generic
reusable Kaggle notebook for all workflows.

### Ledger
Purpose: repair an equation-level fault in the exergy ledger of a solved plant (stream exergy lines, component destruction lines, the ambient constant). These faults leave a plant that solves and passes parameter-range checks; only the ledger's own checks (closure, negative destruction) can reveal them, and some are invisible even to those. Generator: `thermo_agent.ledger.data`; 13 fault types in three layers (stream lines, component lines, the T0 constant); one fault (70 percent) or two. No code is executed: lines are arithmetic expressions checked by a restricted evaluator.
Columns: `id`, `split`, `topology`, `fault_types`, `prompt`, `completion`, `ledger`, `min_edits`.
Splits: `test_ood_topology` (steam-ORC plant and gas-engine plants), `test_ood_fault` (wrong_stream, dropped_work, wrong_ref), then `train`, `val`, `test_id` by hash.
Notes: `ledger` is a JSON string with the tables, topology, broken lines, truth and the numbers shown in the prompt, so rewards need no other column. A program that rebuilds every line from the topology solves every row by construction (`thermo_agent.ledger.run_eval` reports this); the benchmark measures whether a model applies that rule and localises the fault from text.

### Repair
Purpose: fix a plant that fails the solver or the verifiers. Generator: `thermo_agent.synth.build_repair_data`; one fault of five types (unit slip, sign flip, efficiency above 1, swapped ports, wrong fluid) injected into a sampled valid plant of one of five topologies. Only faults the solver or the verifiers detect are kept.
Columns: `id`, `split`, `topology`, `fault_type`, `prompt`, `completion`, `plant_broken`, `plant_original`, `site`, `deducible`, `min_edits`, `detected`.
Splits: `test_ood_topology` (steam-ORC plant and gas-engine plants), `test_ood_fault` (sign flip, swapped ports), then `train`, `val`, `test_id` by hash of the id.
Notes: `deducible` is false when the original value cannot be recovered from the broken plant (an efficiency above 1 that is not a percentage); exact-match reward applies only when it is true. `completion` is a programmatically written expert answer with a number-grounded diagnosis. Turbine plants are drawn three times as often as the held-out topologies.

### Design
Purpose: choose a plant for a site. Generator: `thermo_agent.design.data`; sites are sampled (IT load 5 to 60 MW log-uniform, chilled fraction 0.3 to 0.95, ambient 283 to 310 K, humidity 0.2 to 0.9, optional heat demand and water limit). The reference is the best of enumeration plus a Nelder-Mead polish; use `--max-evals 60` or more.
Columns: `id`, `split`, `prompt`, `completion`, `site`, `ref_fuel_w`, `ref_spec`, `n_valid`, `ref_evals`.
Splits: `test_ood_scale` (IT load >= 45 MW), `test_ood_climate` (ambient >= 305 K), `test_ood_load` (chilled fraction >= 0.85), then `train`, `val`, `test_id` by hash.
Notes: the reference fuel is the best design found by this search, not a proven optimum. Outcomes depend on assumed technology constants (see `paper/methods_draft.md`, last section).

### Audit
Purpose: rank exergy destroyers, propose one allowed change, predict its fuel effect. Generator: `thermo_agent.audit.data`; sampled valid plants (some with ORC bottoming and district heat). Only plants where at least one allowed change saves fuel are kept.
Columns: `id`, `split`, `topology`, `prompt`, `completion`, `plant`, `site`, `ed_json`, `fuel_w`, `net_w`, `best_gain_w`, `n_candidates`.
Splits: `test_ood_topology` (steam-ORC plant and gas-engine plants), then `train`, `val`, `test_id` by hash.
Notes: `ed_json` holds exergy destruction per component (the IT load excluded) and is hidden from the prompt. The ranking reward ignores the combustor or gas engine, which was the largest destroyer in every plant generated so far. `best_gain_w` is the largest fuel saving among the allowed changes after re-sizing the prime mover to the same net electricity.

## Corpus inventory — 2026-11-19

*(merged from `docs/PROVENANCE.md` on 2026-11-20; content kept, headings demoted one level)*

**Status:** dated, factual, read-only measurement. Only this file was written; no corpus, model, config or other
doc was modified. All counts below were measured this date by streaming each JSONL line-by-line (no corpus was
loaded into memory). "not measured" means the field is absent and no value was inferred.

Environment: `C:\Users\rezax\Home\Code\thermo_agent\Repo`, Python `.venv/Scripts/python.exe`.
Script and exact commands in the [Appendix](#appendix--script-and-commands).

---

### 1. Directory-level inventory (rows / harness / tool tags)

`completion` tool-tag counts = rows whose serialized `completion` contains the literal tag.
`traj` = `trajectory.sft_completion` (GRPO stores the multi-turn SFT target there, not in `completion`).

| dir (data/…) | files | rows | prompt present | completion present | `harness` values (rows) | `source` values (rows) | `<tool_call>` rows (completion) | `<tool_result>` rows (completion) | `<tool_response>` rows | traj tool-call rows |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| `processed` | 20 | 27,418 | 27,418 | 27,418 | field absent | field absent | 0 | 0 | 0 | 0 |
| `processed_arch` | 14 | 5,544 | 5,544 | 5,544 | field absent | field absent | 0 | 0 | 0 | 0 |
| `processed_big` | 8 | 194 | 194 | 194 | field absent | field absent | 0 | 0 | 0 | 0 |
| `sft_tools` | 4 | 960 | 960 | 960 | `none`: 960 | `deterministic_tools`: 960 | **960** | **960** | 0 | 0 |
| `grpo_tools` | 4 | 960 | 960 | 960 | field absent | `deterministic_tools`: 960 | 0 | 0 | 0 | **960** (`trajectory`) |
| `sft_all` | 21 | 133,032 | 133,032 | 133,032 | `none`: 87,219; `agents+skills`: 44,743; `skills`: 1,070 | `ground_truth`: 132,397; `frontier`: 635 | 0 | 0 | 0 | 0 |
| `grpo_all` | 21 | 28,247 | 28,247 | 28,247 | `none`: 18,268; `agents+skills`: 8,909; `skills`: 1,070 | `ground_truth`: 27,612; `frontier`: 635 | 0 | 0 | 0 | 0 |
| `frontier` | 4 | 668 | 668 | 668 | field absent | `frontier`: 668 | 0 | 0 | 0 | 0 |
| `frontier_prov` | 4 | 668 | 668 | 668 | field absent | `frontier:<model>` (see §3) | 0 | 0 | 0 | 0 |
| `frontier_export` | 4 | 1,600 | 1,600 | **0** (prompt-only export) | `none`: 1,213; `agents+skills`: 321; `skills`: 66 | `ground_truth`: 1,600 | 0 | 0 | 0 | 0 |
| `frontier_eval_export` | 24 | 360 | 360 | **0** (prompt-only export) | field absent | field absent | 0 | 0 | 0 | 0 |
| `heldout` | 20 | 542 | 542 | 542 | field absent | field absent | 0 | 0 | 0 | 0 |

Task field: present only in `sft_tools`, `grpo_tools`, `sft_all`, `grpo_all`, all `frontier*` task files and
`frontier_eval_export`; absent in `processed*`/`heldout` (task is taken from the filename there). Where present the
task histogram is given in the per-task tables below.

#### 1a. Rows per task × split

`data/processed` (seed 0, MIX):

| task | train | val | test_id | test_ood_fault | test_ood_topology | test_ood_climate | test_ood_load | test_ood_scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 5,458 | 663 | 649 | 4,203 | 837 | 0 | 0 | 0 | 11,810 |
| design | 1,768 | 235 | 204 | 0 | 0 | 615 | 411 | 460 | 3,693 |
| audit | 2,192 | 290 | 299 | 0 | 135 | 0 | 0 | 0 | 2,916 |
| ledger | 4,846 | 672 | 612 | 2,229 | 640 | 0 | 0 | 0 | 8,999 |
| **TOTAL** | **14,264** | **1,860** | **1,764** | **6,432** | **1,612** | **615** | **411** | **460** | **27,418** |

`data/processed_arch` (65-diagram catalog; no design task):

| task | train | val | test_id | test_ood_fault | test_ood_topology | total |
|---|---:|---:|---:|---:|---:|---:|
| repair | 818 | 110 | 95 | 761 | 279 | 2,063 |
| audit | 996 | 113 | 116 | 0 | 141 | 1,366 |
| ledger | 1,097 | 110 | 152 | 464 | 292 | 2,115 |
| **TOTAL** | **2,911** | **333** | **363** | **1,225** | **712** | **5,544** |

`data/processed_big` (big cycle; only repair/audit):

| task | train | val | test_id | test_ood_fault | test_ood_topology | total |
|---|---:|---:|---:|---:|---:|---:|
| repair | 57 | 6 | 5 | 50 | 0 | 118 |
| audit | 66 | 6 | 3 | 0 | 1 | 76 |
| **TOTAL** | **123** | **12** | **8** | **50** | **1** | **194** |

`data/sft_all` (21 files; `audit_test_ood_fault.jsonl` is 0 bytes):

| task | train | val | test_id | test_ood_fault | test_ood_topology | climate | load | scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 110,635 | 669 | 654 | 4,253 | 837 | 0 | 0 | 0 | 117,048 |
| design | 1,908 | 235 | 204 | 0 | 0 | 615 | 411 | 460 | 3,833 |
| audit | 2,258 | 296 | 302 | 0 | 136 | 0 | 0 | 0 | 2,992 |
| ledger | 5,006 | 672 | 612 | 2,229 | 640 | 0 | 0 | 0 | 9,159 |
| **TOTAL** | **119,807** | **1,872** | **1,772** | **6,482** | **1,613** | **615** | **411** | **460** | **133,032** |

`data/grpo_all` (21 files; same non-train rows as `sft_all`, smaller train):

| task | train | val | test_id | test_ood_fault | test_ood_topology | climate | load | scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 5,850 | 669 | 654 | 4,253 | 837 | 0 | 0 | 0 | 12,263 |
| design | 1,908 | 235 | 204 | 0 | 0 | 615 | 411 | 460 | 3,833 |
| audit | 2,258 | 296 | 302 | 0 | 136 | 0 | 0 | 0 | 2,992 |
| ledger | 5,006 | 672 | 612 | 2,229 | 640 | 0 | 0 | 0 | 9,159 |
| **TOTAL** | **15,022** | **1,872** | **1,772** | **6,482** | **1,613** | **615** | **411** | **460** | **28,247** |

`data/sft_tools` and `data/grpo_tools` (4 files each, `*_train.jsonl` only):

| task | train | total |
|---|---:|---:|
| repair | 240 | 240 |
| design | 240 | 240 |
| audit | 240 | 240 |
| ledger | 240 | 240 |
| **TOTAL (each corpus)** | **960** | **960** |

`data/frontier*` task files — all `split=train`, 4 tasks:

| corpus | repair | design | audit | ledger | total |
|---|---:|---:|---:|---:|---:|
| `frontier` | 335 | 140 | 33 | 160 | 668 |
| `frontier_prov` | 335 | 140 | 33 | 160 | 668 |
| `frontier_export` (prompt-only) | 400 | 400 | 400 | 400 | 1,600 |
| `frontier_eval_export` (prompt-only) | 90 | 120 | 60 | 90 | 360 |

`frontier_eval_export` splits: `test_id` 120, `test_ood_fault` 60, `test_ood_topology` 90,
`test_ood_climate` 30, `test_ood_load` 30, `test_ood_scale` 30 (total 360).

`data/heldout` (20 JSONL):

| task | train | val | test_id | test_ood_fault | test_ood_topology | climate | load | scale | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repair | 94 | 13 | 6 | 70 | 15 | 0 | 0 | 0 | 198 |
| design | 54 | 4 | 4 | 0 | 0 | 20 | 13 | 14 | 109 |
| audit | 84 | 9 | 16 | 0 | 6 | 0 | 0 | 0 | 115 |
| ledger | 60 | 10 | 11 | 32 | 7 | 0 | 0 | 0 | 120 |
| **TOTAL** | **292** | **36** | **37** | **102** | **28** | **20** | **13** | **14** | **542** |

---

### 2. `sft_tools` / `grpo_tools` deep metrics

#### 2a. Multi-turn call-pair counts (per row, min / median / max)

For `sft_tools` the tags live in `completion[0].content`; for `grpo_tools` the `completion` is only the final answer
(**0** tool tags) and the multi-turn SFT target lives in `trajectory.sft_completion`. `<tool_call>` and
`<tool_result>` counts are equal per row (one result per call); rows with `<2` calls/results = **0** (measured).

| task | sft_tools calls (min/med/max) | sft_tools results | grpo_tools traj calls | grpo_tools traj results |
|---|---|---|---|---|
| repair | 3 / 3.0 / 3 | 3 / 3.0 / 3 | 3 / 3.0 / 3 | 3 / 3.0 / 3 |
| design | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 |
| audit | 3 / 3.5 / 4 | 3 / 3.5 / 4 | 3 / 3.5 / 4 | 3 / 3.5 / 4 |
| ledger | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 | 2 / 2.5 / 3 |

`<tool_response>` rows = **0** in both corpora; all tool blocks are XML `<tool_result>`.

#### 2b. System-prompt char length (constant across rows of each task)

| task | system-prompt chars (sft_tools = grpo_tools) |
|---|---:|
| repair | 740 |
| design | 1,226 |
| audit | 914 |
| ledger | 838 |

#### 2c. Token counts

- `sft_tools` has a stored per-row `tokens` field. Measured over all 960 stored values:
  **min 1,913 / median 3,778 / p95 4,645 / max 5,742** (matches `MANIFEST.json` token_length p50 3,778,
  max 5,742; manifest p95 4,620 uses a different quantile convention).
- `grpo_tools`: **not measured** — no per-row `tokens` field is stored. The real-tokenizer audit
  (`python -m thermo_agent.train.token_audit`, which needs the cached Qwen3.5 tokenizer) was **not** re-run here;
  it is not a cheap per-row read. The `sft_tools/MANIFEST.json` token_length block describes `sft_tools`, not
  `grpo_tools`.

#### 2d. Stored `source` / `provenance.variant` mix

| corpus | `source` | `provenance.variant` (rows, per corpus) |
|---|---|---|
| `sft_tools` | `deterministic_tools`: 960 | `direct`: 480, `recover`: 480 |
| `grpo_tools` | `deterministic_tools`: 960 | `direct`: 480, `recover`: 480 |

Per task in each corpus: 240 rows = 120 `direct` + 120 `recover`. **`frontier_tools`: 0 rows** in both corpora
(corroborated by `data/sft_tools/MANIFEST.json`: counts `frontier_tools = 0` for all four tasks, `frontier.kept = 0`).
`harness`: `sft_tools` = `none` for all 960; `grpo_tools` has no `harness` field.

`grpo_tools` `validation.reward`: 718 rows at exactly `1.0`, 2 rows at `0.45`, and 240 rows at values in
`(0.9996, 1.0)` (non-unit floats) — i.e. all 960 rows are present as GRPO training rows regardless of reward.

---

### 3. Frontier (`data/frontier*`)

#### 3a. Totals and acceptance

| corpus | rows | accepted (`validation.reward == 1.0`) | rejected (`reward != 1.0`) | rows with tool calls (`<tool_call>`/`<tool_result>`, completion or trajectory) |
|---|---:|---:|---:|---:|
| `frontier` | 668 | 635 | 33 | **0** |
| `frontier_prov` | 668 | 635 | 33 | **0** |
| `frontier_export` | 1,600 | not measured (no `validation`/`completion` field) | not measured | 0 (prompt-only) |
| `frontier_eval_export` | 360 | not measured (no `validation`/`completion` field) | not measured | 0 (prompt-only) |

All 33 rejected rows are the `google-gemini-3-8-flash` audit rows; every design/ledger/repair row has reward `1.0`.

#### 3b. Rows by model and task (`frontier`; `frontier_prov` identical counts)

| task | rows | `google-gemini-3-1-flash-lite-preview` | `openai-gpt-5-4-nano-2026-03-17` | `google-gemini-3-8-flash` | accepted / rejected |
|---|---:|---:|---:|---:|---|
| repair | 335 | 311 | 24 | 0 | 335 / 0 |
| design | 140 | 114 | 26 | 0 | 140 / 0 |
| ledger | 160 | 88 | 72 | 0 | 160 / 0 |
| audit | 33 | 0 | 0 | 33 | 0 / 33 |
| **TOTAL** | **668** | **513** | **122** | **33** | **635 / 33** |

Model source: `frontier` carries no `model` field, so the model slug was parsed from the `id` suffix after `-t-`;
`frontier_prov` carries explicit `model`/`provider` fields and a `source` of `frontier:<model>`. Both give the same
per-task distribution. `frontier_prov/provenance_manifest.json` records the same histogram (513 / 122 / 33;
1, total 668) and SHA-256 per file.

#### 3c. Prompts / responses present

| corpus | `prompt` | `completion` (response) | `validation` | note |
|---|---|---|---|---|
| `frontier` | yes, 668/668 | yes, 668/668 | yes, 668/668 | zero-shot answers, no tool turns |
| `frontier_prov` | yes, 668/668 | yes, 668/668 | yes, 668/668 | same rows + `model`/`provider` |
| `frontier_export` | yes, 1,600/1,600 | **no**, 0/1,600 | no | Kaggle teacher-prompt export (400/task) |
| `frontier_eval_export` | yes, 360/360 | **no**, 0/360 | no | held-out prompt export (30/row-cell, limit 30) |

#### 3d. Usable frontier tool traces — explicit statement

**0 usable frontier tool traces exist on disk.** Measured: 0 of 668 `frontier` rows and 0 of 668 `frontier_prov`
rows contain any `<tool_call>` / `<tool_result>` tag in `completion` or `trajectory`; 0 rows in either tool corpus
have `source=frontier_tools`. The tool-trace builder's own manifest (`data/sft_tools/MANIFEST.json`,
generated 2026-11-19T09:27:59Z) records `frontier.scanned = 1,336` (= 668 + 668), `with_tool_turns = 0`,
`kept = 0`, `rejected = 0`, `over_budget = 0`, `deduplicated = 0`. The 635 accepted frontier rows that do exist are
zero-shot (`harness=none`, `tools=none`), with 0 tool calls, so they cannot serve as tool traces.

---

### 4. Held-out (`data/heldout`, seed 987654)

- 20 JSONL files / **542 distinct ids** total; of these, **12 frozen `*_test_*.jsonl` files / 214 ids / 4 tasks**
  are the scored set. Per-file counts (all ids unique per file; each row `split` matches its filename):

| file | split | task | rows |
|---|---|---|---:|
| `audit_test_id.jsonl` | test_id | audit | 16 |
| `audit_test_ood_topology.jsonl` | test_ood_topology | audit | 6 |
| `design_test_id.jsonl` | test_id | design | 4 |
| `design_test_ood_climate.jsonl` | test_ood_climate | design | 20 |
| `design_test_ood_load.jsonl` | test_ood_load | design | 13 |
| `design_test_ood_scale.jsonl` | test_ood_scale | design | 14 |
| `ledger_test_id.jsonl` | test_id | ledger | 11 |
| `ledger_test_ood_fault.jsonl` | test_ood_fault | ledger | 32 |
| `ledger_test_ood_topology.jsonl` | test_ood_topology | ledger | 7 |
| `repair_test_id.jsonl` | test_id | repair | 6 |
| `repair_test_ood_fault.jsonl` | test_ood_fault | repair | 70 |
| `repair_test_ood_topology.jsonl` | test_ood_topology | repair | 15 |
| **TOTAL** | — | **4 tasks** | **214** |

Task totals: repair 91, audit 22, ledger 50, design 51. Measured bytes of the 12 test files: **1,531,809** (this
matches the byte total quoted in the docs below).

**Frozen eval-set hash (quoted, not recomputed).** `docs/EVAL_PROTOCOL.md` §"Eval-set hash" records, and
`docs/EVAL_PROTOCOL.md` §1 / Appendix A repeat, that the SHA-256 over sorted test-file names then each file's bytes
for the 12 `*_test_*.jsonl` files is:

```
8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e   (12 files, 1,531,809 bytes)
```

This report did **not** recompute the hash or guess the recipe; the value above is quoted from
`docs/EVAL_PROTOCOL.md`, which states it matches the current on-disk data. (For reference only, the 20-file variant
quoted in the same doc is `12cb731a062486e51e39582e3c1de708e7d8276295cba6f3994051314b12be3e`; it is **not** the
protocol hash.) The `meta_<task>.json` files record seed 987654 and per-split `counts` that agree with the tables.

---

### 5. GAPS — what is missing for the Gate 4 mix (deterministic + accepted frontier traces)

Gate 4 (`IMPLEMENTATION_PLAN.md` §"Gate 4") requires `data/sft_tools` and `data/grpo_tools` to mix **deterministic
tool traces + accepted frontier tool traces** (single-turn rows optional as a documented ablation). Measured state:

| ingredient | sft_tools | grpo_tools | status |
|---|---:|---:|---|
| deterministic tool traces | 960 (240/task; 480 direct + 480 recover) | 960 (240/task; 480 direct + 480 recover) | present |
| accepted frontier tool traces | **0** | **0** | **missing** |
| frontier rows scanned by tool builder | 1,336 | 1,336 | scanned; 0 with tool turns |
| accepted frontier zero-shot rows (not tool traces) | 0 inside these corpora; 635 exist in `sft_all`/`grpo_all` | same | unusable as tool traces (0 tool calls) |

**Exact counts still needed:**

1. **Frontier tool traces: 960 per tool corpus (240 per task × 4) for a 1:1 frontier:deterministic mix, or a hard
   minimum of 1 per task × 4 = 4** so the "accepted frontier tool traces" ingredient is non-empty at all. Current
   measured count is **0 in both corpora**. To go from 960 → 1,920 rows in each corpus at 1:1 would require
   **+960 accepted frontier tool traces per corpus** (i.e. 960 for `sft_tools` and 960 for `grpo_tools`; if the same
   traces are reused for both, 960 unique accepted traces are needed).
2. **Frontier pool with tool turns: currently 0 of 1,336 scanned rows.** Need at least 1 accepted trace with a tool
   call per task; to match the deterministic tier, need 240 accepted tool-using traces per task (960 total). The
   build manifest shows `with_tool_turns=0`, so new harness generations (physics tools `inspect_plant`,
   `first_law_check`, `second_law_check`, `validity_check`, `solve_plant`) must be produced and ingested — none of
   the existing zero-shot rows qualify.
3. **Audit frontier: 0 accepted rows exist** (all 33 `google-gemini-3-8-flash` audit rows have reward < 1.0). A
   nonzero accepted audit yield is a prerequisite for any audit frontier tool trace; per `IMPLEMENTATION_PLAN.md`
   the audit prompt/tool fix (P0-5) should precede an audit pilot. Need **≥1 accepted audit frontier row**, and
   **240** to match the audit deterministic tier.
4. **Tokens for `grpo_tools`: not measured** (no stored `tokens` field). Need a real-tokenizer `token_audit` run
   (or a stored token field) to confirm the 8,000-token gate; not required to fix the frontier-trace gap but listed
   as an outstanding measurement.

Deterministic side is complete for Gate 4's mix; every remaining gap is on the accepted-frontier-tool-trace side.

---

### Appendix — script and commands

All reads were streaming `for line in open(path, encoding="utf-8")` over each `*.jsonl`; no corpus was loaded whole.
Scripts were written only to `%TEMP%/opencode`, never into the repo.

```
# from C:\Users\rezax\Home\Code\thermo_agent\Repo
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe C:\Users\rezax\AppData\Local\Temp\opencode\corpus_inventory_measure.py   > ...\corpus_inventory_out.json
.venv\Scripts\python.exe C:\Users\rezax\AppData\Local\Temp\opencode\corpus_inventory_summary.py  > ...\corpus_inventory_summary.txt
.venv\Scripts\python.exe C:\Users\rezax\AppData\Local\Temp\opencode\corpus_inventory_crosstab.py > ...\corpus_inventory_crosstab.txt
```

`corpus_inventory_measure.py` computed, per file: row count; `split` histogram; `task`/`harness`/`source`/
`provenance.variant` histograms; `validation.reward` histogram; model (from `model` field or `id` suffix after
`-t-`); prompt/completion presence; counts of rows containing `<tool_call>` / `<tool_result>` / `<tool_response>`
in `completion` and in `trajectory.sft_completion`; per-row tag-count and system-prompt-char-length distributions;
and stored `tokens`. Companion scripts aggregated these into the cross-tabs and the frontier per-task/model/acceptance
tables. Quantiles use `xs[min(n-1, int(f*n))]` (p50/p95); medians in §2a use the mean of the two central values.
Held-out hash was quoted from `docs/EVAL_PROTOCOL.md` ( corroborated in `docs/EVAL_PROTOCOL.md` §1/Appendix A ); not
recomputed here. No data, code, config or model was changed.
