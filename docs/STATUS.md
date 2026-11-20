# Status and operating notes

Working copy: `C:\Users\rezax\Home\Code\thermo_agent\Repo` on REZA_PC (git, branch `main`). No SSH, no WSL: every
step below runs in this checkout.

## Truth snapshot — 2026-11-20 (LATEST; supersedes every section below)

Authoritative decisions: `docs/DECISIONS_2026-11-19.md` (esp. §4–§7, §7.9–§7.11). Plan: `docs/IMPLEMENTATION_PLAN.md`.
Source-of-truth reads for this snapshot: `docs/BUDGET_BATCH_REPORT.md`, `notebooks/README.md`,
`notebooks/kaggle_model_runner.py`, `docs/TRACE_QUALITY_AUDIT_2026-11-19.md`,
`docs/HOSTED_V6_ASSESSMENT_2026-11-19.md`, `docs/EVAL_PROTOCOL.md`. Where any older section below calls a
corpus, eval set, model order or smoke version "current", read this section instead.

### Local verification and current facts (2026-11-20)

- **Commit `4228347`** is in place and `docs/USER_PROMPTS_VERBATIM.md` carries Reza's prompts **1-9** verbatim
  (prompt 9 added 2026-11-20). That file's wording is not to be edited.
- **Provenance audit OK:** `scripts/audit_provenance_chain.py --strict` → **PROVENANCE OK, 8/8 checks pass**
  (237 cohort rows carry model/provider/run_id/source; prompt shards clean; no harness bookkeeping leakage; memory
  has both accepted and rejected notes; 320 frontier mix rows traced — 258 cohort + 62 documented legacy gemini;
  0 `ref_spec` leaks in 11,520 mix rows). Detail in `docs/PROVENANCE.md`.
- **Preflight OK:** `scripts/preflight_budget_batch.py` replays the hosted runner's `Config.select_rows` locally
  against the Kaggle dataset bytes; wired into `scripts/overnight_budget_loop.ps1` before any HF/Kaggle/model call.
- **Rehearsal OK:** `scripts/rehearse_budget_batch.py` builds runner-shaped fake journals, runs the real token gate
  and real `transfer.ingest` into a temp dir, and checks the empty/`__error__` guards — both budget models 4/4
  accepted. Zero quota spent.
- **Runner selection fix:** partial-tier shards no longer balance against empty tiers (the cause of the 0-byte
  journals); `notebooks/kaggle_model_runner.ipynb` regenerated.
- **Datagen gate honesty:** `scripts/check_datagen_complete.py` counts only the 24 budget cells and dedupes by base
  diagram id (was inflated by legacy copies).
- **Reasoning capture:** per-turn assistant prose IS the reasoning trace for chat-completions models; captured,
  replayed into later turns, never served as raw text through memory. `reasoning_effort` stays off for models that
  reject it (`reasoning_for_model`).
- **Skills + dynamic tools:** 27 `SKILL.md` files cover every solver component, variants, tools, pinch, exergy,
  design-selection and repair diagnosis. `thermo_agent/teacher/thermo_tools.py` adds 9 continuous-input,
  CoolProp-backed dynamic tools (fluid state, saturation, turbomachine, HX, combustor, cycles, flow network,
  psychrometrics, exergy) with 29 tests. Skills load on demand (short description, full skill on demand). Dynamic
  tools are opt-in (`include_dynamic`); old 5-tool traces stay valid (DECISIONS §9).
- **GRPO scoring join + expanded scoring-only export DONE:** `ref_fuel_w`/`ref_spec`/`ref_evals` were stripped from the
  training mix rows; the scoring-time join reads them from `data/frontier_export/design_train.jsonl` so GRPO design
  rewards still work — **1,216/1,216 design rows resolved** (`tests/test_design_scoring_refs.py`).
- **Test suite:** 10 failures from verbose-harness skill bloat are being fixed by another agent (no AI/Kaggle).
  Final counts land in the next commit. Re-run `scripts/audit_provenance_chain.py --strict` after those integrations
  (it was already green before them).
- **No SFT step, GRPO step, model evaluation or checkpoint exists.** Training waits for Reza's "datagen complete"
  and "go".
- **Kaggle runs v19/v20 (luna + haiku) reported COMPLETED and are being processed** per Reza's prompt 9 —
  download → gate → ingest → memory/mix refresh. All older 0-byte `batch*_completions.jsonl` files are known failed
  attempts and must be refused by `process_budget_batch.ps1`. No other Kaggle command beyond these runs has been issued.
- **Rules unchanged:** only `gpt-6-luna`/`claude-haiku-5-5` may be called; gemini 3.8/3.5 hard-banned; premium needs
  explicit permission; Kaggle traffic via `socks5h://127.0.0.1:10808`; credentials only from repo-root `.env`; never
  print tokens. One hosted task `thermo-teacher-cotproof-s00`; one Kaggle dataset (`thermo-agent-corpus`, per
  DECISIONS §7.5/§7.9); local → HF → bridge kernel → dataset only. New docs are governed by the anti-litter rule
  (DECISIONS §8); orientation samples live in `data/samples/`.

**What has NOT happened:** no SFT step, no GRPO step, no model evaluation and no checkpoint exists anywhere.
No arm has been scored, and no results table claims measured model performance. The only hosted compute is
teacher datagen, not training.

**Paper artifact caveat:** `paper/generated/table_results.tex`, `table_comparison.tex`, `table_training.tex` and
`values.tex` were updated by hand to pending/decided content. Do NOT rerun
`paper/tools/generate_paper_assets.py` until its generator is updated, or it will overwrite these with stale
values. The build path is `paper/build_paper.ps1` (48-page `main.pdf`, no undefined refs); `verify_paper.py`
exits 1 here only because MiKTeX's `latexmk` needs Perl, which is not installed.

**Ordering (Reza directive, DECISIONS §4).** ALL datagen first, then **SFT → GRPO (k=4/8/16) → every eval arm**.
Tiers are **equal thirds per task** (`easy`/`medium`/`hard`) in both training and test data. Training starts only
when every datagen item is built, audited and committed. The remaining datagen item is the frontier CoT campaign.

### Datagen state (truthful, 2026-11-20)

| Item | State | Evidence |
| --- | --- | --- |
| Deterministic tool corpora | **DONE** — 4,800 rows per task family (600 plants × 2 variants `direct`/`recover` × 4 tasks), real physics tool results, verifier accepted | `data/sft_tools_v2*`, `data/grpo_tools_v2*`, manifests |
| Equal-thirds tiers | **DONE** — `easy`/`medium`/`hard` per task from `thermo_agent/synth/tiers.py`, remainder to hard, exact counts in manifests | `docs/EVAL_PROTOCOL.md` §2, trace audit §1 |
| Plain A-slice | **DONE** — 1,600 rows (400/task) = round(4,800 × 0.3333), plain prompt, no tool markers | `data/a_slice` |
| `heldout_v2` (test) | **DONE and FROZEN** — 3,476 scored rows in 12 cells, equal thirds per task, contamination scan PASS (overlap 0) | `docs/EVAL_PROTOCOL.md` (test hash `3cc51773…`) |
| Compact harness + gate re-measure | **DONE** — one ≤2,500-token text for tracegen/training/eval; build_trace re-measures on the rendered row; gate 8,192, over-gate rows dropped not truncated | DECISIONS J2/J3, §1 |
| `*_h` corpora / `sft_mix_h` | **DONE** — every tool row uses the compact harness; plain prompt only for the A-slice (E4) | `data/sft_tools_v2_h`, `data/sft_mix_v2` |
| Training-path fixes | **DONE** — task+tier loading, one harness rendering, `<tool_result>`/`<tool_response>` label masking, collator, checkpoint/resume, eval/save cadence, red smoke kernel; each with a test | `tests/test_decisions_applied.py`, `tests/test_tool_loss_masking.py` |
| Provenance | **DONE** — every row states deterministic vs AI and, if AI, the exact `model` id | `docs/PROVENANCE.md`, `docs/dataset_cards.md` |
| Token efficiency | **DONE** — token gate measured with the real Qwen3.5 tokenizer, over-gate rows dropped and counted, never truncated | `docs/TRACE_QUALITY_AUDIT_2026-11-19.md` §2 |
| Frontier harness CoT | **IN PROGRESS** — see below | `docs/BUDGET_BATCH_REPORT.md` |

### Frontier CoT campaign (IN PROGRESS)

- **Only `gpt-6-luna` and `claude-haiku-5-5` are authorized.** Completed so far: each model's **5-row proof**
  (luna run `7233597`, 5/5 accepted, \$0.003358; haiku run `7233598`, 5/5 accepted, \$0.012746) and **one
  100-row batch per model** on task version 9. The 100-row journals are at the local token-gate → ingest →
  audit stage; no accepted-row count is claimed here.
- **Target: THOUSANDS of verifier-accepted harness-CoT traces with NO CAP** on traces, per-model counts, cost or
  rows (DECISIONS §6.1, D1/D2/D6/D7/D9). Coverage is equal thirds across `easy`/`medium`/`hard` in all four
  tasks; this is a floor, not a stopping number. **The old 72-cell (six-model) premium target is DISCARDED.**
- **Not to be run, and no traces/costs/scores are implied for them:** `gpt-6.2-sol`, `claude-sonnet-5-5`,
  `claude-opus-5-5-default`, `gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`. The generic runner
  hard-blocks gemini 3.8/3.5 and skips non-allowed models before any model call
  (`notebooks/kaggle_model_runner.py`).
- Data flow: **local → private HF repo `Reza-Nadimi/thermo-agent-corpora` → Kaggle kernel → ONE Kaggle dataset**
  (`thermo-agent-corpus`), with code/prompt datasets attached read-only. **No local data uploads to Kaggle.**
  All credentials live only in the repository-root `.env`. One generic reusable notebook
  (`notebooks/kaggle_model_runner.py` / `.ipynb`) serves every workflow (tracegen, eval, zero-shot, cohorts,
  with/without tools and harness); no per-task variants.
- RAG memory stores **short success/failure recaps** (what failed → what works), with model ids, timestamps,
  cost and raw checker logs stripped before prompting.

Test = `data/heldout_v2`; `data/heldout` (214 ids) is **dev** only. The older "Done — held-out set",
"final comparison scored with `DATA_DIRS=heldout`" and "Next" item 5 below are superseded. The
"Truth snapshot (evening)" and every older section below are historical.

## Truth snapshot (2026-11-19 evening; historical — superseded by the section above)

Authoritative values: `docs/DECISIONS_2026-11-19.md`. Plan: `docs/IMPLEMENTATION_PLAN.md` (traces first, then train,
then score). Hard rule still standing: no `kaggle` command without Reza's "go".

- **Done - first real frontier tool traces.** `thermo-teacher-tools-s00` version 6 (`google/gemini-3.8-flash`, run
  `7179338`) produced **32 verifier-accepted tool-using CoT traces** out of 40 attempts (repair 9, ledger 10, audit 6,
  design 7), 5 turns each, 1 rejected for grounding, 5 rows lost to provider 429s. Cost $0.842202 = **$0.0263 per
  accepted trace**. Rows in `data/frontier_tools/frontier_*_train.jsonl` (32 unique `record_id`s). Assessment:
  `docs/HOSTED_V6_ASSESSMENT_2026-11-19.md`.
- **Done - ingest id resolution.** `ingest --extra-export-dir` (repeatable) + `record_id` keying; without it 12 of the
  32 traces were silently dropped because their ids live in `data/processed_arch` or `data/frontier_prov`, not in
  `data/processed`. Tests in `tests/test_transfer.py` (6 pass).
- **Open - trace text is not yet gate-clean.** 11 of 32 rows exceed the 8,192 gate because of the provider's
  `call_id='call_...__mpthought__<base64>'` wrapper. Removing only that wrapper gives 0/32 over, max 7,280 tokens
  (real Qwen3.5 tokenizer). Sanitizer targets `data/frontier_tools_clean`; `data/frontier_tools` stays as received.
  *Superseded:* the sanitizer has since run to `data/frontier_tools_clean_v2` (32 rows, 0/32 over gate); see the
  latest snapshot above.
- **Open - compact harness (0.1) and gate re-measure (0.2).** The harness prompt is still the 17.2k-char version and
  `build_tool_corpus.py` still measures the plain prompt, so the 157 over-gate rows in `data/sft_tools_h` are still
  there and no `*_h` rebuild has happened.
- **Open - corpus composition (0.4-0.6).** No A-slice, no `sft_mix_h`, no `heldout_v2` yet; 32 traces is a pilot
  yield, not the E1 corpus (deterministic target ~4,800 tool rows).
  *Superseded:* `data/a_slice` now exists (1,600 plain rows, 400/task); `sft_mix_h` and `heldout_v2` still open.
  See the latest snapshot above.
- **Not done - no trained weights, no scores.** `runs/` has only smoke directories; no SFT adapter, no GRPO stage, no
  held-out report. Training stays closed until the smoke kernel is green (G1).
- **Model order (J4).** `gemini-3.5-flash-lite` -> `claude-haiku-5-5` -> `gpt-6-luna` (tools, reasoning off,
  documented) -> premium tier only after bulk shows non-zero tool yield and a 40-row pilot cost is recorded. The older
  "budget first: luna" order in the snapshot below is superseded.

## Previous snapshot (2026-11-19 morning; superseded by the section above)

This is the truth snapshot for the current plan. The longer historical sections below are kept for provenance;
where they call an older smoke version or an older teacher pass "current", read this section instead.

- **Done — self-correcting teacher harness.** `thermo_agent/teacher/agent_graph.py` is a LangGraph `StateGraph`
  (`build_context -> propose -> {call_tools -> reflect -> propose}* -> finalize -> END`, up to `max_rounds=4`).
  `thermo_agent/teacher/toolkit.py` exposes five real tools — `inspect_plant`, `first_law_check`,
  `second_law_check`, `validity_check`, `solve_plant` — each built from the model's own candidate (or the given
  plant for `inspect_plant`) and run through the real solver/verifiers. The old `check_answer` oracle was removed
  from the tool list; acceptance stays at grading time as the GRPO reward.
  `thermo_agent/teacher/compact_memory.py` implements the D10 guidance and D11 compact memory (what failed ->
  what works; model ids, timestamps, cost and logs stripped) and is wired into `build_context`.
- **Done — held-out set.** `data/heldout/` is frozen with seed **987654** (present in each `meta_*.json`). It is
  the single set on which all arms are scored and is excluded from every training corpus and teacher prompt shard.
- **Done — evaluation arms.** `kaggle/eval_runner.py` defines the arms, including `ours_sft_grpo_k4`,
  `ours_sft_grpo_k8` and `ours_sft_grpo_k16`. Each k pulls its **own** checkpoint dir
  (`runs/grpo_<task>_k4`, `_k8`, `_k16`), so the three scores never share weights. The final comparison is scored
  with `DATA_DIRS=heldout`.
- **Done — smoke environment.** The smoke kernel (`rezanadimi/thermo-train-smoke`, labelled v23 in the current
  campaign record; the Kaggle CLI does not expose a version number in this checkout) was pushed with the
  PyPI-pinned numpy/scipy shim in `thermo_agent/kaggle_notebook.py::setup()`: `numpy==2.4.4`, `scipy==1.17.1`,
  installed from `https://pypi.org/simple` and verified in a clean subprocess before the worker continues.
- **Pending — teacher campaign.** The frontier tool-CoT campaign has **not** been executed. Run
  `scripts/run_teacher_campaign.ps1` with the budget models first, then the expensive tier (see `docs/RUNBOOK.md`
  and `docs/TEACHER_MODEL_CHOICES.md`).
- **Not done — no trained weights.** No SFT or GRPO adapter / merged weights exist yet (`runs/` contains only
  smoke directories), so every trained-model arm is still unscored.
- **Model order.** Budget first: `gpt-6-luna`, then `claude-haiku-5-5`, reasoning ON; `gemini-3.5-flash-lite` is
  excluded and must not consume quota. These are hosted-only slugs run as the hosted task `thermo-teacher-tools-s00`
  via `scripts/run_teacher_hosted.ps1 -m <slug>`, not the local SDK driver. Then the opt-in expensive tier:
  `gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default` (`claude-opus`), `gemini-4-argon` (`gemini-4`) —
  only after the budget tier reports verifier-accepted reasoning-trace yield, and `claude-sonnet-5-5` additionally
  needs Reza's explicit go-ahead. **REMINDER: revisit `claude-sonnet-5-5`** (deferred as too expensive).

## Environment

| Item | Value |
|---|---|
| Machine | REZA_PC, Windows 11 Home, 16 cores, 15.2 GB RAM, RTX 4050 6 GB |
| Python | 3.12.3 in `.venv` (created with `uv`), pinned by `requirements-lock.txt` |
| Package | installed editable (`pip install -e . --no-deps`) |
| Tools | `uv`, `ripgrep`, `fd`, `jq` in `%USERPROFILE%\tools\bin` and `%USERPROFILE%\.local\bin` |
| Local model | LM Studio serving `qwen3.5-4b` on `http://127.0.0.1:1234/v1` (format smoke checks only) |
| Tokenizer | `Qwen/Qwen3.5-4B` `tokenizer.json` cached in `~\.cache\thermo\qwen3.5-4b\` (248,070 tokens) |
| Training target | Kaggle `GPU T4 x2` via `scripts/train_dual_t4.sh` |
| Frontier teacher | Kaggle Benchmarks (`kaggle_benchmarks`) via `kaggle/teacher_benchmark.py` |
| Generated notebooks | `kaggle/teacher.ipynb`, `train.ipynb`, `teacher_harness.ipynb`, `eval.ipynb` (all from `kaggle/make_notebook.py`) |
| Paper readiness | `paper/Q1_CHECKLIST.md` separates measured evidence from pending model runs |

No WSL and no tmux: long jobs are detached Windows processes (`Win32_Process.Create`), so they survive an SSH
disconnect. `Start-Process` children do not. A machine sleep still kills them.

## Running long jobs

```powershell
pwsh scripts/long.ps1 start gen-data '& ./scripts/gen_data.ps1'   # start detached
pwsh scripts/long.ps1 status gen-data -Lines 40                    # state + log tail
pwsh scripts/long.ps1 list                                        # all jobs
pwsh scripts/long.ps1 kill gen-data
```

Logs: `logs/<name>.log`; the job writes `logs/<name>.pid`, which `status` uses to tell RUNNING from FINISHED.
Never give a job the same name as the script it runs.

## Data pipeline

The corpus has four sources, all deterministic or verifier-grounded (no free-form CoT labels):

> **Historical (superseded).** This describes the legacy zero-shot corpus. The current training mix adds
> verifier-accepted frontier **harness-CoT** traces (luna/haiku campaign, in progress) on top of the complete
> 4,800-row deterministic tool corpora; see the 2026-11-20 snapshot above.

1. **Ground-truth paraphrases** (`thermo_agent/synth/paraphrase.py`): every ground-truth repair row is re-emitted
   with `--variants 20` truthful phrasings built from the fault facts. A row picks its variants by a hash of its id,
   so the corpus as a whole covers the phrasing space, not one fixed wording: the current train split has
   **100,036 repair rows** (teacher rows included; 20 paraphrases per ground-truth row). The pool is 20 to 80 renderings per fault class
   (`unit_slip` 20, `efficiency_gt_1` 25, `wrong_fluid` 80), so 20 is always available and exhausts the sparsest
   class. Paraphrases are a train-split device only; the eval splits keep one row per plant so a model evaluation
   never counts the same plant twice.
2. **Architecture corpus** (`thermo_agent/synth/architectures.py`): 65 plant diagrams (turbine/engine prime mover,
   absorption/electric cooling, wet/dry/extended rejection, with/without ORC, plus `cgam_steam_orc`), each with ORC
   pressures chosen per working fluid. **Diagram combinations** are held out for the OOD splits: `cgam_steam_orc` and `engine-*-ext*` go to `test_ood_topology`. ORC, engines, turbines, district heat all train.
3. **Frontier teacher traces** (`thermo_agent/teacher/transfer.py` + `validate.py`): prompts are exported without
   their answers, answered on Kaggle Benchmarks, then every answer is re-run through the tool suite here. Only
   accepted answers become rows; rejections become hindsight memory.
4. **Hindsight failure memory** (`thermo_agent/teacher/memory.py`): lessons mined from rejected attempts are
   retrieved per prompt (k=3) and injected as "known failure modes". There are **342** stored notes
   (318 from rejected frontier answers across all four tasks + 24 from the adversarial repair miner); the
   graded rejection summary counts 965 attempts before deduplication. See `docs/API_USAGE_AND_PROGRESS.md`.

`scripts/build_corpus.ps1` stages, audits, verifies and swaps two corpora:

- `data/sft/{task}_{split}.jsonl` — harness on 35 percent of rows, memory where it retrieves;
- `data/grpo/{task}_{split}.jsonl` — **same prompts as SFT** (with harness/memory, `-GrpoVariants 1`), so the model sees at GRPO time exactly the input format it learned during SFT. The trainer (`train/grpo.py:load_rows`) keeps only the prompt and the reward columns; the reference completion stays in the file for auditing but is never read. GRPO samples G=4 completions per prompt, so 20 paraphrases of one plant would only repeat the same prompt.

### Token budget (the gate that matches the trainer)

Every candidate row is measured with the real Qwen3.5 tokenizer before it is written; rows that do not fit are
dropped and counted, never truncated. The legacy 4,000-token gate described here (11,500-character cheap pre-filter;
prompt + completion + a 192-token reserve for the chat markers) has been superseded in the merged corpora
`data/sft_all` / `data/grpo_all` by an **8,000-token prompt gate under `max_length` 8,192**.

Measured harness cost per task (200 train rows, tokens, p50 / max):

| task | none | agents | skills | agents+skills | mode used |
|---|---|---|---|---|---|
| repair | 964 / 1129 | 2295 / 2460 | 1360 / 2630 | 2689 / 3959 | agents+skills |
| audit | 1362 / 2202 | 2692 / 3532 | 2854 / 4803 | 4183 / 6132 | skills |
| ledger | 1748 / 2983 | 3078 / 4313 | 2328 / 3563 | 3657 / 4892 | agents+skills |
| design | 502 / 519 | 1832 / 1849 | 944 / 961 | 2273 / 2290 | agents+skills |

Audit is the outlier: an audit plant can carry many components, so AGENTS.md plus its skills does not fit; audit
therefore trains with the skill block only. Ledger is the task whose harness rows hit the token ceiling: 985 ledger
candidates were dropped by the token gate (all with the agents+skills harness), and 28,808 repair candidates by the
character pre-filter.

### Current corpus (variants 20, harness 0.35, memory 3)

| corpus | rows | train rows | max row | rows over 8192 |
|---|---|---|---|---|
| `data/sft` (legacy 4k gate) | 123,024 | 109,659 | 4,000 tokens | 0 |
| `data/grpo` (legacy 4k gate) | 28,214 | 14,849 | 4,000 tokens | 0 |

The current corpora are `data/sft_all` (119,807 train rows) and `data/grpo_all` (15,022 train rows), both gated at
8,000 prompt tokens under `max_length` 8,192 (max row 6,405 tokens, zero drops) and both containing all 635 accepted
frontier rows.
Train split by task (SFT): repair 100,036 (16,862 harness, memory on every repair row, 49 architectures), ledger 5,165 (1,417 harness), audit 2,564 (525 skills), design 1,894 (635 harness). `sft_report.json` reports
`problems: []` for both corpora, and `reports/token_audit_{sft,grpo}.json` report `fits: true`. After the swap,
`python -m thermo_agent.synth.verify data/sft` re-reads every written file, compares the row counts with the report,
and rejects a duplicate id or a file the report does not know about; `build_corpus.ps1` runs it on the staged and
on the swapped corpus.

At 32 sequences per optimizer step (2 per device x 8 accumulation x 2 ranks) one SFT epoch is ~3,650 steps. A 4B fp16
LoRA step at 1.5k average tokens costs roughly 30 s on two T4s, so one epoch is about 30 hours: plan resume across
2–3 Kaggle sessions (12 h each). Checkpoints every 200 steps via `save_strategy="steps"`.

## Baselines (small-sample reference values)

`scripts/baselines.ps1` over `data/processed` + `data/processed_arch`: at most 200 repair rows per split
(30–106 per fault), 30 design sites per OOD split, 100 audit plants, 400 ledger rows.

- repair, verifier-guided search with a 60-evaluation budget (mean solver calls in brackets): `test_id` repairs
  58/65 `efficiency_gt_1` (24), 58/58 `unit_slip` (47) and 77/77 `wrong_fluid` (272); `test_ood_fault` repairs all
  106 `sign_flip` (8) and all 94 `swapped_ports` (239); `test_ood_topology` repairs 39/41 `efficiency_gt_1` (8),
  48/48 `sign_flip` (9), 38/38 `swapped_ports` (263), 43/43 `unit_slip` (57) and 25/30 `wrong_fluid` (283). Exact
  restoration of the original plant is rarer: 4–69 of 58–77 on `test_id` (the budget usually settles on a different
  valid plant for an efficiency above 1), 2–48 of 30–48 on held-out diagrams.
- design: enumerating the defaults lands 3.1–3.4 % from the best known design at 38–44 evaluations; the polish
  reaches 0.0 % at 126–148; random search with 12 evaluations is 8.8–12.7 % worse and invalid in 1–6 of 30 sites.
- audit: on `test_id` (n=100) first-law nDCG@3 0.630 vs 0.695 for random and top-1 1 % vs 24 %, while on
  `test_ood_topology` (n=100) first-law nDCG@3 0.864 vs 0.65, top-1 0.63 vs 0.401; the fixed rule
  (raise turbine inlet temperature 10 %) is valid in 87/100 and captures a mean 47.1 % of the best saving; change
  search needs 11.6 evaluations vs 1 for a model.
- ledger, `test_ood_fault` (n=400): no-op solves none, the regenerate rule solves all; at least one second-law
  symptom appears in 90–100 % of rows by fault type (closure residual or negative destruction), while closure alone
  is broken in 32–100 %.

The paper's numbers are regenerated by `paper/tools/generate_paper_assets.py`; the audit ranking macros now read the
same 100+100 row run as the baseline table. The SVG and PDF of `fig_audit_first_vs_second_law`, `fig_training` and
`fig_corpus` were drawn by a tool outside this repository and still show their earlier pilot samples and split bar;
they must be rebuilt from `paper/figures/figure_manifest.json` and the current assets before the camera-ready PDF.

**Pending data regeneration.** The five-template MIX generators named engine plants without their rejection
boundary, so ideal-rejection engines from `data/processed` leaked into `train`/`test_id` while the catalog's
`engine-*-ext*` diagrams were held out. `synth.splits.topology_label` now recovers the boundary from the plant graph
and Repair, Audit and Ledger use it for both the `topology` field and the split. The checked-in datasets and all
paper numbers above are still from the pre-fix data: re-run `scripts/gen_data.ps1`, `build_corpus.ps1` and the paper
asset stages after the teacher pass, before freezing the final corpus. Prompts are unchanged (the label is metadata),
so `data/frontier_export` stays valid for the teacher.

## Qwen3.5-4B architecture (for LoRA)

`Qwen3_5ForConditionalGeneration` with separate text + vision configs:

**Text backbone** (32 layers, hybrid linear/full attention):
- `hidden_size=2560`, `intermediate_size=9216` (SwiGLU), `num_attention_heads=16`, `num_key_value_heads=4` (GQA)
- `head_dim=256`, `partial_rotary_factor=0.25`, `rope_theta=10_000_000`
- **Layer pattern** (every 4th layer is full attention, indices 3,7,11,15,19,23,27,31):
  ```
  linear_attention, linear_attention, linear_attention, full_attention,
  linear_attention, linear_attention, linear_attention, full_attention,
  ...
  ```
- Linear attention layers use **gated delta nets** (SSM-like): `linear_q_proj`, `linear_k_proj`, `linear_v_proj`, `linear_out_proj` with `linear_conv_kernel_dim=4`, `linear_num_key_heads=16`, `linear_num_value_heads=32`, `linear_key_head_dim=128`, `linear_value_head_dim=128`
- Full attention layers: standard `q_proj`, `k_proj`, `v_proj`, `o_proj`
- FFN (all 32 layers): `gate_proj`, `up_proj`, `down_proj`
- MTP (multi-token prediction): 1 extra layer `mtp_num_hidden_layers=1`

**Vision encoder** (24 layers, 1024 hidden, patch_size=16, out=2560):
- `qwen3_5_vision` config, `gelu_pytorch_tanh` act, axial RoPE
- We **do not train vision** — our data is text-only plant JSON.

**LoRA targets** (full fp16, rank 16 SFT / rank 8 GRPO, all-linear on text):
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `linear_q_proj`, `linear_k_proj`, `linear_v_proj`, `linear_out_proj`
- FFN: `gate_proj`, `up_proj`, `down_proj`
- (Skip `patch_embed.proj` and vision attention/FFN — not used)

## Training path (dual T4)

- `scripts/train_dual_t4.sh` defaults to `DATA=data/sft` for SFT and `GRPO_DATA=data/grpo` for GRPO, detects
  `NGPU` and the dataloader worker count, and disables the InfiniBand probe. `STEPS` / `ROWS` add smoke limits.
- two DDP ranks via `torchrun --nproc_per_node=$NGPU`;
- **full fp16 LoRA**, no 4-bit weights: the T4 is Turing and has no bf16 kernels, so `--dtype float16`, guarded by
  `thermo_agent/train/ddp.py`;
- DDP replicates the frozen 4B model; only LoRA gradients cross PCIe, once per optimizer step (`no_sync` on
  accumulation micro-steps, `gradient_as_bucket_view`, `ddp_bucket_cap_mb=64`);
- overflow handling: AMP `GradScaler`, `max_grad_norm=1.0`, `adamw_torch`, fp32 matmul reductions;
- batch arithmetic: SFT 2 x 8 x 2 = 32; GRPO G = 4 with 4 x 2 x 2 = 16 prompts per step. TRL requires the
  per-device batch to be a multiple of G, so 4 is the smallest legal choice;
- **Resume**: both trainers save every 200 steps (`save_strategy="steps", save_steps=200`, GRPO every 50) and
  auto-resume from `latest(out_dir)` via `transformers.trainer_utils.get_last_checkpoint`. One SFT epoch ≈ 3,650
  steps ≈ 30 h.
- SFT `max_length` 8192; rows are gated at 8,000 tokens when the corpus is built, so nothing is truncated.
- The chat template was checked against the real tokenizer: `prefix_matches: true`, and the prompt ends at
  `<|im_start|>assistant<think></think>`, so SFT, GRPO and evaluation render the same generation prompt.
- GRPO uses the **same prompts** as SFT (with harness/memory); its loader keeps only the prompt and the reward
  columns.

Trainings have not been run yet locally: the 4050 cannot hold a 4B model for training. No SFT or GRPO weights
exist anywhere yet (locally or on Kaggle); see the current-state section above.

### Measured Kaggle smoke (dual-T4 path)

- Smoke kernel v17 (`rezanadimi/thermo-train-smoke`, 20 steps, 64 rows, 1 rank, batch 1, 512x8192 loss tiles)
  **trained successfully**: `[memory] 4.21B params, weight dtype torch.float16`/0.00 GiB before placement,
  `[loss] chunked cross-entropy` active, loss 1.59 -> 0.61 over 20 steps, ~50 s/step. No OOM in training.
- It failed in the **step-20 evaluation forward**: `torch.OutOfMemoryError`, 488 MiB requested for
  `torch.linalg.solve_triangular` in Qwen3.5's linear attention with 246 MiB free. The 4096-token
  allocation is in the **reference PyTorch implementation of Qwen3.5's linear attention**
  (`torch_chunk_gated_delta_rule` -> `solve_triangular`), not in the LM-head logits; the worker warned that
  `flash-linear-attention` and `causal-conv1d` are absent, so the slow fallback kernels are in use.
- Fix under test (eval path only, no weights/truncation/training settings changed):
  `ChunkedSFTTrainerMixin.prediction_step` frees cached CUDA blocks before the forward, forces fused SDPA
  backends for the full-attention layers, and runs under `torch.inference_mode()` with a narrow fallback.
  If the eval still OOMs, the remaining cause is the linear-attention fallback kernel and the next step is to
  install the project's fused `flash-linear-attention` kernels in the worker, not to shorten the window.
  Local CUDA check: train loss 6.953, eval loss 6.948 (reference full-logits fp16 loss 6.948).
- Historical: smoke v18 with this eval-path fix was pushed. **Superseded** — the pushed smoke environment and its
  PyPI-pinned numpy/scipy shim are recorded in the current-state section above, not here.
- Kaggle's Secrets service was unreachable inside the worker (`Connection error trying to communicate with
  service`), so v17 had no `HF_TOKEN`: checkpoints stayed local to that session. Cross-session HF resume
  requires the secret to be reachable in the worker; no token was copied into code or datasets.

## Tests

- Full CPU suite: **370 passed** (latest run 304.55 s, `logs/pytest_round2.log`) plus 3 new resume-relay tests
  (11 passed for `test_kaggle_notebook.py` + `test_resume_relay.py`). `paper/methods_draft.md` exists, so its
  constant checks run instead of skipping at module level.
- New this round: `tests/test_run_eval.py` (the Repair per-fault breakdown must score the rows it names) and three
  cases in `tests/test_splits.py` (the MIX generators must put ideal-rejection engines in `test_ood_topology`);
  the round before added `tests/test_verify.py` (the corpus verifier must notice a missing file, a row-count mismatch,
  a leak across splits and a row over the token gate), plus the earlier `test_token_audit.py`, `test_transfer.py`,
  `test_checkpoint.py`, `test_splits.py` and the per-task harness, train-only paraphrase and token-gate cases in
  `test_combine.py`.
- Run `pytest -q` before any commit that touches `thermo_agent/`; `paper/tools/verify_paper.py` checks that every
  `\val{...}` used in the sections exists in the generated assets (the LaTeX build itself needs a TeX installation).

## Kaggle (legacy zero-shot teacher pass) — completed; superseded for training data

The teacher pass was executed with the Kaggle CLI (not the browser notebook): 1,600 prompts per model,
Gemini 513 accepted / $1.090957 and GPT-5.4-nano 122 accepted / $0.658021; 3,668 raw training answers,
635 verifier-accepted rows, 635 used in both merged corpora (`data/sft_all` / `data/grpo_all`; the legacy 460 was
the 4k-gated `data/sft` / `data/grpo` subset). Held-out answers: 360 per model. Full accounting and
provenance are in `docs/API_USAGE_AND_PROGRESS.md`; `docs/TEACHER_MODEL_CHOICES.md` carries the model decision.
This pass is the **legacy zero-shot baseline**; the current tool-CoT campaign is still pending (see the
current-state section above).

## Expert-scale cycles and per-trace provenance (new)

- `thermo_agent/synth/big_topologies.py::big_cchp_steam2p_reheat_hp` adds a 19-component / 34-stream
  two-pressure steam cycle with reheat, separate LP loop, chilled water, district heat and a wet tower.
  It solves and passes all hard verifiers; sampled draws reach 60–120 MW IT load and ~150–230 MW fuel.
  It is registered in `BIG_ARCHITECTURES` and is **additive** to the 65 small diagrams (`catalog()` unchanged).
- New ground truth lives in `data/processed_big` (seeds 100/101) and is merged with the old corpora via
  `combine --extra-processed data/processed_big` into `data/sft_all` / `data/grpo_all`; nothing in
  `data/processed`, `data/sft` or `data/grpo` is deleted.
- The token gate is now 8,000 prompt tokens under `max_length` 8,192 (was 4,000 / 4,096), so long
  expert-cycle traces are not dropped; `data/grpo_all` audited at max row 6,405 tokens with zero drops.
- `scripts/build_provenance_ledger.py` emits one record per unique trace in
  `data/provenance/trace_ledger.jsonl` (37,746 traces) plus `stats.json` and `docs/PROVENANCE.md`, with
  origin, journal line, cost, acceptance and downstream artifact links. It is auto-discovering: dropping a
  new big-model journal into `data/teacher_runs/` adds traces without a code change.

## Model campaign round 2 (user decision, 2026-11-18) — superseded; pending execution

- Hosted task `thermo-teacher-big-s00` is now at **version 3** with reasoning **ON** (`reasoning="low"`,
  1,536 output tokens) to capture reasoning traces where possible. Reasoning also fixes the Claude 400.
- User-selected cheap models for this round: `gpt-6-luna`, then `claude-haiku-5-5` (reasoning ON);
  `gemini-3.5-flash-lite` is excluded and must not consume quota. Superseded by the current campaign order in the
  current-state section above.
  Driver `scripts/run_big_teacher_v2.ps1` runs them sequentially and ingests additively.
- **REMINDER: revisit `claude-sonnet-5-5` later** — the user deferred it as too expensive and wants to decide
  after seeing how many accepted reasoning traces the cheap models produce.
- `gemini-3.8-flash` already ingested 33 accepted / 200 attempted; the ledger is at **38,202 traces**.
- **Stale:** this round-2 driver (`scripts/run_big_teacher_v2.ps1`) describes the earlier hosted pass. The current
  campaign is `scripts/run_teacher_campaign.ps1` and has not been executed yet; see the current-state section.

## Next (current plan; supersedes the older items below)

1. **Finish the frontier CoT campaign** (the only open datagen item): larger batches with `gpt-6-luna` and
   `claude-haiku-5-5` only, equal thirds across easy/medium/hard, until thousands of verifier-accepted traces
   exist with no cap. After each batch, refresh local + served memory (short success/failure recaps) via
   `transfer.ingest`, then gate/ingest/audit the downloaded journal locally. Use the one generic runner
   `notebooks/kaggle_model_runner.py` and the HF-only bridge (local → `Reza-Nadimi/thermo-agent-corpora` → Kaggle
   → one `thermo-agent-corpus` dataset). No premium or gemini 3.8/3.5 run.
2. Re-run `scripts/check_datagen_complete.py`; training starts only when it exits zero.
3. Run SFT, then one joint GRPO run per k ∈ {4,8,16} from the merged SFT checkpoint, then score every arm on
   `data/heldout_v2` (`LIMIT=0`). No checkpoint exists yet; nothing is scored until the gate is green.
4. Only after the first full score: regenerate `paper/assets/*.json`, `paper/generated/*.tex`, the paper PDF and
   the provenance/dataset-card docs, then fill the remaining Q1 items in `paper/Q1_CHECKLIST.md`.

## Addendum 2026-11-19 — verified facts (appended; does not supersede the sections above)

### Hosted teacher task `thermo-teacher-tools-s00`
- **v3 / v4 errored before any model call**: the benchmarks venv (`/benchmarks/.venv`, Python 3.12) lacked scipy,
  then CoolProp. No model was billed on those versions.
- **Fix (in the tree today):** scipy root-finding goes through `thermo_agent/core/_roots.py` (scipy `brentq` with a
  pure-Python fallback); `design/search.py` imports scipy lazily; `core/fluids.py` and `core/props.py` use optional
  CoolProp imports. The hosted task tries `pip install --target /kaggle/working/pylibs CoolProp==6.8.0` and
  degrades to gas-only plants when the network is unavailable — it never fails the whole task on a missing
  real-fluid backend.
- **`thermo-agent-code` dataset versions:** v30 "scipy-free physics core", v31 "optional CoolProp + lazy scipy".
- **v5** built and ran `gemini-3.8-flash`. The `gpt-6-luna` pilot is **pending task readiness**.

### Smoke kernel `rezanadimi/thermo-train-smoke` v26 / v27
- **v26 failed** with CUDA OOM at the **first backward on one T4**: **13.55 GiB in use, 1.10 GiB requested**.
- Hypothesis: eager/math attention on a full-attention layer at ~5.7k tokens materialises 16 heads × seq² × fp16
  ≈ 1 GiB.
- **v27 fixes pushed, result pending:** `attn_implementation="sdpa"` in `ddp.model_init_kwargs`;
  `gradient_checkpointing_kwargs={"use_reentrant": False}` for SFT and GRPO; and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `kaggle/train_runner.py`.

### Corpus inventory (`docs/CORPUS_INVENTORY_2026-11-19.md`)
| corpus | rows | note |
| --- | ---: | --- |
| `data/sft_tools` | 960 | 240/task; 480 direct + 480 recover; 2–4 tool pairs per row |
| `data/grpo_tools` | 960 | same mix; multi-turn trace in `trajectory.sft_completion` |
| `data/frontier` | 668 | 635 accepted / 33 rejected; **0 rows with tool calls** → 0 usable frontier tool traces |
| `data/heldout` | 12 test files | 214 ids; SHA-256 `8b409359e7a2dd441d0be99c89c00f0f3c61320f9d06e9695cc22569f8ef4c6e` |

### Prompt separation — OPEN
Eval and teacher tracegen already emit separate Z (zero-shot) and H (harness) prompts, but training rows still
carry only the plain `harness="none"` system prompt (740 chars for repair). A `--prompt-variant
{plain,harness_tools}` implementation is in progress to emit `*_h` corpora whose system prompt matches the
`ours_harness` eval arm. **Recorded as OPEN until the code lands and the corpora are built.**

### Kaggle quota snapshot (2026-11-19)
| Metric | Value |
| --- | --- |
| Monthly spend | $6.05 / $100 used |
| Daily spend | $0.02 / $10 used |
| GPU time | 3.74 h used / 26.26 h remaining |
| Refill | 2026-11-21 |
