# Implementation plan — traces first, then train, then score

Status: **decided.** Every value lives in `docs/DECISIONS.md` (IDs A1..J5). Do not change a value; to change one,
add a superseding entry there with the measurement that justifies it. Order is fixed by Reza: make the AI traces first, then train.
**HARD RULE stays until Reza lifts it:** no `kaggle` command (push, run, sync, upload) is executed from this plan. Everything
marked *[Kaggle]* below is prepared, dry-run and tested locally, then waits for "go".

Standing rules: fp16 LoRA only (B6); additive-only (no corpus/model/journal deleted); no token in notebooks/datasets/logs; Kaggle only
through the SOCKS proxy; free subagents only; no assistant-chosen caps or hyper-parameters; every gate ends with a `docs/DECISIONS.md`
entry, an updated `docs/STATUS.md`, regenerated dataset stats and one commit.

**Data-flow rule (DECISIONS §7.8–§7.11, binding).** Every file that touches Kaggle goes through the private HF
corpus repo first: **local → `Reza-Nadimi/thermo-agent-corpora` → Kaggle kernel → ONE Kaggle dataset
(`thermo-agent-corpus`)**. No local data upload to Kaggle; code/prompt datasets are attached read-only; no account
littering. All credentials come only from the repository-root `.env`. One generic reusable notebook
(`notebooks/kaggle_model_runner.py` / `.ipynb`) serves every workflow; no per-task variants. The runner hard-blocks
gemini 3.8/3.5 and skips non-allowed models before any model call.

**Local-only verification gates (2026-11-20, no quota).** `scripts/preflight_budget_batch.py` (replays
`Config.select_rows` against the Kaggle dataset bytes) runs in `scripts/overnight_budget_loop.ps1` before any
HF/Kaggle/model call, and `scripts/rehearse_budget_batch.py` exercises the real token gate and `transfer.ingest` on
fake journals — both green. The scoring-time join for design `ref_*` fields (1,216/1,216 rows) and the 9 opt-in
dynamic tools are wired locally; their final test counts land in the next commit. Kaggle runs v19/v20 are completed
and being processed per Reza's prompt 9. New docs are governed by the anti-litter rule (DECISIONS §8); orientation
samples are in `data/samples/`.

## Next actions (2026-11-20, from the Claude review — in this order, one commit each)

Data that exists after the patch: `heldout_v2` (test), `data/a_slice` (plain ground-truth rows), prompt shards. Everything else live must be rebuilt.

1. **Apply and verify.** `git am` the patch; `pytest` (fresh clone); `python kaggle/make_notebook.py`; `scripts/make_smoke_kernel.py` then the smoke-notebook test. Fix any dangling reference; never recreate a merged doc.
2. **Re-measure the prompt budget.** The default prompt now carries the reasoning rule and the 16-tool schema. Run `scripts/audit_harness_corpus.py` and `thermo_agent/train/token_audit` on a sample of rendered rows per task; if the harness block + tool schema exceeds the 2,500-token budget or rows exceed the 8,192 gate, cut the skill catalog descriptions / tool docstrings, not the reasoning rule. Record the numbers in `docs/DECISIONS.md`.
3. **Rebuild deterministic traces.** `python scripts/build_tool_corpus.py --src data/processed --out data/sft_tools_v3 --grpo-out data/grpo_tools_v3 --tasks repair,design,audit,ledger --split train` (and the `_h` harness-prompt variants), 600 plants per task x {direct, recover}, then `scripts/audit_harness_corpus.py`, `scripts/drop_over_gate.py`, `scripts/dataset_stats.py`. Have subagents review 30 rows per task one at a time (read the whole row; verdict line each).
4. **Reasoning pilot, then bulk.** Regenerate shards with the new contract (`scripts/frontier_cohort_shards.py`), run 40 rows (10 per task, equal tiers) through luna and haiku via `scripts/overnight_budget_loop.ps1`'s single-batch path. Report per model x task: share of tool turns with reasoning, mean reasoning chars, accepted vs `rejected_no_reasoning` vs verifier-rejected, cost per accepted trace. Only then run bulk. Diagnose audit/ledger/design yield from the raw journals first (it was repair-only before).
5. **Mix.** `scripts/mix_tool_corpora.py` with `data/sft_tools_v3_h`, the new `data/frontier_tools_accepted`, `data/a_slice` -> `data/sft_mix_v3` / `data/grpo_mix_v3`; provenance audit (`scripts/audit_provenance_chain.py --strict`); upload with `scripts/upload_corpus_dataset.ps1`.
6. Training and scoring stay blocked until Reza says "datagen complete" and "go".

## Binding order and current datagen status (Reza directive, 2026-11-19)

Reza's directive (DECISIONS §4, which supersedes any conflicting line above) is binding:

1. **All datagen complete first** — deterministic tool corpora at the decided size
   (600 plants/task × 2 variants × 4 tasks = 4,800), all authorised frontier tool-CoT
   traces, the plain A-slice, and `heldout_v2`. Nothing trains until every one is built,
   audited and committed. No SFT, no GRPO, no training-based smoke shortcut, and no
   Kaggle run under this plan.
2. **Tiers are equal thirds per task** (`easy`/`medium`/`hard`) in *both* training and
   test data, derived from `thermo_agent/synth/tiers.py` (DECISIONS §4.2).
3. **Order after datagen:** SFT → GRPO (k = 4/8/16) → evaluation of every arm on
   `heldout_v2`. Zero-shot and harness baselines are reported *alongside*, never
   substituted for trained scores.
4. **The training path is audited and fixed before the first training run** (data loading
   with task + tier, prompt rendering, label masking, collator, checkpoint/resume,
   eval/save cadence, red smoke kernel), each fix with a test.

**Current reality (2026-11-20):** the input-side datagen items are **complete** except the frontier CoT
campaign, which is **in progress**; no training has started and no SFT/GRPO/eval result or checkpoint exists.
`data/processed` deterministic ground truth is intact (27,418 rows, nothing deleted/overwritten). The A-slice is
a complete copy of 1,600 plain rows (400/task = round(4,800 × 0.3333)). Deterministic tool corpora are **DONE at
4,800 rows per task family** (600 plants × 2 variants (`direct`, `recover`) × 4 tasks), with real tool results,
verifier acceptance and equal-thirds tiers per task. `heldout_v2` is **generated and FROZEN** (3,476 scored rows,
12 cells, equal thirds, contamination scan PASS). The compact harness (≤2,500 tokens, one text for every arm), the
`*_h`/`sft_mix_h` rebuild, the training-path fixes (task+tier loading, one harness rendering, tool-result label
masking, collator, checkpoint/resume, eval/save cadence, red smoke kernel), per-row provenance and the token gate
(8,192; over-gate rows dropped, never truncated) are all **DONE** with tests. Frontier harness CoT is **IN PROGRESS**:
only `gpt-6-luna` and `claude-haiku-5-5` are authorized; each has a 5-row proof (5/5 accepted) and one 100-row batch
completed, with the journals at gate/ingest/audit. The target is **thousands** of traces with **no cap**, equal thirds
across easy/medium/hard; the old 72-cell premium target is **discarded**. Premium models and gemini 3.8/3.5 are not
to be run. Test = `data/heldout_v2`; `data/heldout` (214 ids) is dev.

## Phase 0 — make the inputs correct (local, CPU, no Kaggle)

*Phase 0 status: **DONE** (2026-11-20). Items 0.1–0.7 are built, audited and covered by tests; the per-item notes
below are retained as the acceptance record.*

0.1 **Compact harness (J2/J3).** Cut the harness block to <= 2,500 tokens (real Qwen3.5 tokenizer), one text for every arm.
    Acceptance: `scripts/audit_harness_corpus.py` prints max harness-block tokens <= 2,500 and one sha for tracegen, training, eval.
0.2 **Fix the gate (defect #1/#7).** `build_trace` re-measures `tokens` after the harness prompt is swapped in and gates on that.
    Acceptance: test builds a row whose plain prompt fits but harness row does not and asserts it is dropped and counted.
0.3 **Rebuild `*_h` corpora and tracegen shards** on the compact harness; stats table (task x tier x variant, tokens p50/p95/max,
    dropped). Rule from DECISIONS section 1: audit losing > 10% -> cap tool outputs, never raise `max_length`.
0.4 **Deterministic traces to 600 plants/task** x 2 variants (direct, recover) = 4,800 tool rows (E1), tiers equal thirds for
    repair/audit, 50/50 easy/medium for ledger/design; 100% verifier-accepted; real tool results verbatim.
    *Status: **DONE**.* 4,800 rows (600 plants × 2 variants × 4 tasks) in each tool corpus, with real physics tool
    results and 100% verifier acceptance. Reza's directive (§4.2) applied equal-thirds tiers to **all four** tasks in
    training and test, not just repair/audit; exact counts are in the manifests.
0.5 **A-slice builder (E2).** One paraphrase per ground-truth plant from `sft_all`, task-balanced, size R_tool/3; plain prompt;
    marked `prompt_variant=plain`. Mixing script writes `data/sft_mix_h` and `data/grpo_tools_h`; `mix_tool_corpora.py --stats` must
    show `with tool turns > 0` for the tool part.
    *Status:* `data/a_slice` exists — a **copy** of 1,600 rows (400/task) = round(4,800 × 0.3333), one paraphrase per plant,
    seeded, plain prompt with no tool markers. It contains no AI-generated rows.
0.6 **`heldout_v2` (F1).** Generate on CPU with a new seed; >= 85 rows per reported (task, split) cell where a generator can reach it;
    document merged cells; difficulty tiers easy/medium/hard (hard = 19-component cycle where generators support it), equal thirds per task
    (Reza directive §4.2). Freeze manifest + hash, write the eval protocol, scan every training dir for content overlap (hash of
    plant/site/spec), rename current `data/heldout` role to dev.
    *Status: **DONE and FROZEN**.* Test = `data/heldout_v2` (3,476 scored rows, 12 cells, equal thirds per task,
    contamination scan PASS with zero overlap; test hash `3cc51773…` recorded in `docs/EVAL_PROTOCOL.md`);
    `data/heldout` (214 ids) is **dev**, not test. Note that `reports/heldout_dev_protocol.md` was frozen while
    `data/heldout` was still the eval set; its test-set role is superseded by F1 (its cell counts still describe the
    dev set). Phase 5 scores on `heldout_v2`, never `data/heldout`.
0.7 **Tests that make the decisions stick:** `tests/test_decisions_applied.py` (values), contamination test on content hashes, tool-trace round trip
    (record -> ingest -> row has tool turns and ends with the verified answer), `finalize` appends the final turn, ledger tool no crash on malformed
    candidate, audit tool depends on the candidate, GRPO stage names distinct per k, eval raises on a missing adapter.
Gate 0: fresh-clone `pytest` green (real-data tests marked `corpus`), stats and manifests committed.

## Phase 1 — smoke before anything big *[Kaggle, after "go"]*

1.1 Triage `thermo-train-smoke` v28 log (G1), fix, push again until green as defined in DECISIONS section 1 (SFT 20 steps on compact-harness
    tool rows at 8,192, step-20 eval, save+resume, then a 3-step multi-turn GRPO k=4). Record seconds/row and seconds/GRPO-step.
1.2 Replace the est. numbers in DECISIONS with the measurements (R_tool rule, S rule). Nothing else changes.
1.3 Hosted-task payload test: pilot of 5 rows per model, then 40 (Phase 2 pilot). A model that rejects the payload is dropped, not retried.

## Phase 2 — frontier tool traces *[Kaggle, after "go"]*

2.1 **Only `gpt-6-luna` and `claude-haiku-5-5` (DECISIONS §6.1, override of §6 point 1):** run larger batches with these
    two models; update local memory and the short success/failure memory served to the Kaggle harness after every batch.
    Premium models (`gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default`, `gemini-4-argon`) and
    `gemini-3.8-flash`/`gemini-3.5-flash-lite` are NOT to be run — none was run, no traces/costs/scores exist for them.
    The target is **thousands of verifier-accepted traces, NO CAP** on traces, per-model counts, cost or rows, in equal
    thirds across easy/medium/hard. **The old 72-cell premium target is DISCARDED.** Each model's 5-row proof (5/5
    accepted) and one 100-row batch are complete; journals are at the gate/ingest/audit stage. The Kaggle harness work
    is done by the main agent with full context. Use the one generic runner
    `notebooks/kaggle_model_runner.py` through the HF-only bridge; `limit 0`, rounds 6 (D4), `k_memory` 3 (D5), no
    cost/row cap (D1-D9).
2.2 Download each completed journal, id-dedupe into `data/teacher_runs/hosted_<model>.jsonl`, ingest with the compact harness prompt, accept by D8.
2.3 After every run: accepted per task x tier x model, cost per accepted trace, failure reasons -> `reports/teacher_campaign_log.md`; failures feed the memory
    (what failed, what worked; never model names or times).
2.4 Re-run `mix_tool_corpora.py --stats` and `scripts/check_datagen_complete.py`. **No training starts while the gate is red**:
    a proof or one batch alone is not enough; the campaign runs until thousands of accepted traces exist, in equal thirds
    per task, with no cap. The rejected 72-cell six-model target is not the gate.
2.5 Rebuild `sft_mix_h` with frontier traces first (E1); rows over the 8,192 gate dropped and counted. Every row keeps
    `source`, model id (AI rows) and `tier`.

## Phase 3 — SFT *[Kaggle, after "go"]*

3.1 Sync corpora (H1-H3) after the rebuild passes the audit. Upload code bundle + corpus dataset.
3.2 Run `kaggle/train_kernel` with `SFT_DATA=sft_mix_h`, all four tasks, A1-A10 values (epochs <= 3, lr 1e-4, r 32/alpha 64, 2x8x2, save/eval 50,
    cosine + 5% warmup, early stopping patience 3, best model loaded). Relay the resume dataset after every session (G3/G4).
3.3 Merge the best checkpoint (`runs/sft_merged`), push to the private HF repo for GRPO (C10). Dev-set check in the eval kernel before GRPO.

## Phase 4 — multi-turn GRPO *[Kaggle, after "go"]*

4.1 One joint run per k in {4, 8, 16}: G = k, per-device 2, `grad_accum = k` (4 prompts/step), lr 2e-5, S steps (rule C3), T 1.0 / top_p 1.0 / top_k 0,
    `max_completion_length 4096`, `max_tool_calling_iterations 6`, beta 0, dapo. Same prompt order for all k. Stage names `grpo_k{K}` identical locally,
    remotely and in eval. Joint-task reward dispatch by the row's task (C9) with a test.
4.2 Log reward, zero-variance share, truncation rate, turns and tool calls per rollout; stop-and-report rule from C-table.

## Phase 5 — score every arm on `heldout_v2` *[Kaggle, after "go"]*

5.1 Arms (F4): frontier Z/H, base 4B Z/H, base 9B Z/H (two T4 fp16 or 4-bit, state the precision; run the 4B at the same precision once),
    SFT Z/H, GRPO k4/k8/k16 Z/H, other-LLM placeholders. Same rows, same limits (6 turns, 8 tool calls, 4,096 per turn), same compact harness.
5.2 Decoding: greedy primary; pass@8 secondary with T 0.6, top_p 0.95, top_k 20, seeds 0-7 (add the sampling parameters to `HFGenerator`).
    No repetition or presence penalty anywhere.
5.3 `eval_runner.py`: `DATA_DIRS=heldout_v2`, `LIMIT=0`, fail on a missing adapter, fail if report split ids differ from the manifest.
5.4 Statistics: Wilson intervals, paired McNemar/permutation, Holm over the four headline comparisons (harness uplift, SFT gain, GRPO gain by k,
    ours vs 9B/frontier); per task, tier and OOD split; turns, tool calls, tokens, dollars per accepted answer; failure taxonomy.

## Phase 6 — docs, paper data, release

6.1 Regenerate `paper/assets/*.json`, run `paper/tools/verify_paper.py`, list stale prose in `STATUS.md`; figures are not redesigned here.
6.2 Release package: code, `heldout_v2` + hash, trace dataset card, adapters, environment lock.

## Done means

Fresh clone passes tests; every number traces to a report with eval-set version, harness hash and commit; zero held-out/dev overlap in every
training artifact; frontier tool traces exist and are counted; every arm has a `heldout_v2` report; DECISIONS, STATUS and manifests agree with code and data.
