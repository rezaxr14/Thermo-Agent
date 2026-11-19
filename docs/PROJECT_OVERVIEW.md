# thermo-agent — project overview (for external review)

Snapshot: 2026-11-18. All numbers are generated from files in this repository; `docs/PROVENANCE.md`,
`docs/API_USAGE_AND_PROGRESS.md`, `docs/STATUS.md` and `paper/Q1_CHECKLIST.md` carry the detailed records.

## Current implementation plan

The current state (verified in `docs/STATUS.md`) is: the LangGraph self-correcting harness with five real physics
tools and compact memory is done (`thermo_agent/teacher/agent_graph.py`, `toolkit.py`, `compact_memory.py`); the
held-out set is frozen at seed 987654 (`data/heldout/`); the evaluation arms include SFT+GRPO at k=4, 8 and 16 with
separate checkpoint dirs (`kaggle/eval_runner.py`); and the smoke kernel was pushed with the PyPI-pinned
numpy/scipy shim (`thermo_agent/kaggle_notebook.py`). The teacher tool-CoT campaign is **pending** — run
`scripts/run_teacher_campaign.ps1` (budget models first; then the expensive tier), with model order and the standing
`claude-sonnet-5-5` reminder in `docs/TEACHER_MODEL_CHOICES.md`. **No SFT or GRPO weights exist yet.** Exact commands
for the campaign, the three Kaggle kernel pushes and scoring every arm on `data/heldout` with `DATA_DIRS=heldout`
are in `docs/RUNBOOK.md`.

## 1. Goal

Train a 4B physics-grounded agent for four industrial-thermodynamics tasks (Repair, Design, Audit, Ledger) and
compare it against non-model baselines, frontier models zero-shot, frontier models with our physics tools, and
other models. All answers are graded by deterministic physics verifiers (mass/energy balances, second law,
component limits, site constraints), not by an LLM judge. Paper target: Q1.

## 2. Data — what exists and where it came from

**Synthetic ground truth.** 33,156 unique verifier-graded traces (`data/processed` 27,418;
`data/processed_arch` 5,544; `data/processed_big` 194). Tasks: repair, ledger, audit, design. Structural OOD
splits hold out whole diagram families. 20 paraphrase variants per repair base row (4,990 × 20 = 99,800 SFT
repair-train rows). New expert-scale cycle: 19-component / 34-stream two-pressure steam plant with reheat,
separate LP loop, electric chilling, district heat and wet cooling — solves and passes all hard verifiers
(60–120 MW IT, ~150–230 MW fuel).

**Teacher / frontier traces.** Two hosted models were called through Kaggle quota:
Gemini 3.1 Flash Lite (1,600 prompts, 513 accepted, 32.1%, $1.090957) and GPT-5.4-nano (1,600 prompts,
122 accepted, 7.6%, $0.658021). 3,668 raw training answers, 3,339 frontier ledger traces, 1,111 harness/tool
traces, 635 verifier-accepted rows, 635 used in training in both merged corpora, 965 graded rejections deduplicated into 342 memory
notes. Total recorded API spend across all independent batches: **$2.752407**.

**Per-trace provenance.** `data/provenance/trace_ledger.jsonl` holds **37,945 unique trace records** with exact
origin (generator+seed or model id), journal path/line, tokens/cost, harness mode, acceptance verdict and
downstream artifacts. `stats.json` and `docs/PROVENANCE.md` are regenerated from it; `build_corpus.ps1` runs it
automatically. Any new journal dropped into `data/teacher_runs/` is discovered additively — no code edit.

**Training corpora.** Original: `data/sft` 123,024 rows / 109,659 train; `data/grpo` 28,214 / 14,849.
Merged with expert cycles (originals preserved): `data/sft_all` **119,807 train rows including 1,140 big-cycle
rows**; `data/grpo_all` **15,022 train rows including 57 big rows**. Token gate raised from 4,000 to **8,000**
under `max_length` **8,192**; audit reports max row **6,405** tokens and **zero drops**. All **635** accepted
frontier rows are present in both merged corpora (the legacy 4k-gated `data/sft` / `data/grpo` held only 460).

### Tool-using CoT data generation (new objective)

The zero-shot frontier rows above are kept as legacy training data, but the current objective is to teach the 4B
student **tool use**. Frontier models run inside a **LangGraph harness**
(`thermo_agent/teacher/agent_graph.py`: `build_context -> propose -> {call_tools -> reflect -> propose}* ->
finalize -> END`) with **real physics tools** from `thermo_agent/teacher/toolkit.py`: `inspect_plant`,
`first_law_check`, `second_law_check`, `validity_check` and `solve_plant` (the old `check_answer` oracle is no longer
exposed). The tools build the plant from the model's
own candidate and run the actual sequential-modular solver, returning numbers with units. `build_context` injects
**AGENTS.md + component/task SKILL.md** (the same skills the student trains on) and **fresh failure memory** reloaded
from `data/memory/*.jsonl` on every row. The model may call tools over multiple rounds, read numeric feedback and
revise before finalizing.

When the verifiers accept a tool-using trace, the **entire trajectory** — reasoning, tool calls, numeric tool results
and revisions — is saved as the SFT completion (`source=frontier_tools`), so the student imitates the tool-using
behaviour, not merely the final answer. Generation uses `gpt-6-luna` then `claude-haiku-5-5` with reasoning ON, run
as the hosted task `thermo-teacher-tools-s00` via `scripts/run_teacher_hosted.ps1 -m <slug>`;
`gemini-3.5-flash-lite` is excluded and must not consume quota. The expensive tier (`gpt-6.2-sol`,
`claude-sonnet-5-5`, `claude-opus-5-5`, `gemini-4-argon`) follows only after the budget tier reports accepted
reasoning traces, and `claude-sonnet-5-5` needs Reza's explicit go-ahead.
**REMINDER: revisit `claude-sonnet-5-5` later** (deferred as too expensive). Every
arm is scored on the held-out `data/heldout` set (fresh seed 987654, excluded from all corpora), with each checkpoint
in its own path (`runs/sft`, `runs/grpo_k4`, `runs/grpo_k8`, `runs/grpo_k16`).

## 3. Verified measurements

- Frontier zero-shot (30/split): repair 0.633 `test_id`, 0.467 `ood_fault`, 0.433 `ood_topology`; design valid
  0.167/0.2/0.067/0.1; audit rank 0.086/0.033; ledger all_correct 0/0/0.067.
- Frontier + our tools (`test_id`): repair 0.667, design valid 0.433, audit 0.1, ledger 0.767.
  Tool usage: 120 answers, 242 `check_answer` calls, $0.2027, answer rate 1.0.
- Domain-model arm (GPT-nano stand-in) is weaker across the board.
- Design baselines on 120 held-out sites: enumeration 57% within 2% at 128 evals; random 67%;
  enumerate-then-polish 99%. Reports reproduced exactly on the same rows.
- Paper: 45 pages, `verify_paper.py --no-build` clean, no overfull boxes, 242 `\val` keys with sources.

## 4. Training infrastructure (Kaggle)

- Dual T4 DDP replication, full fp16 LoRA, GradScaler, clipping, fp32 reductions; 2 GPUs via
  `--accelerator NvidiaTeslaT4`. Chunked cross-entropy (512×8192 tiles) removes the 248,320-vocab OOM.
- Environment pinned once in `kaggle/environment.lock.json`: Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0,
  transformers 5.23.0, trl 1.18.1, peft 0.21.3, accelerate 1.16.0, datasets 5.2.0, einops 0.8.2,
  fla-core 0.6.0, flash-linear-attention 0.6.0, causal-conv1d 1.7.0 (nvcc wheel built for sm75 once).
  Wheels are transferred **Kaggle-to-Kaggle** via `kernel_sources` from kernel
  `rezanadimi/thermo-kernel-wheels`; no worker ever recompiles. `assert_pinned_environment()` runs inside the
  training and eval notebooks and fails loudly on drift.
- Kaggle artifacts: private datasets `thermo-agent-code`, `thermo-agent-corpus` (now includes `sft_all.zip`,
  `grpo_all.zip`), `thermo-teacher-prompts`, `thermo-teacher-prompts-big`, `thermo-agent-eval-replays`,
  `thermo-agent-resume`; kernels `thermo-train-smoke`, `thermo-train-run`, `thermo-eval-run`,
  `thermo-kernel-wheels`, hosted task `thermo-teacher-big-s00`. Local CLI calls use SOCKS proxy
  `socks5h://localhost:10808`.
- Resume without the HF token: kernel output → local CLI → private `thermo-agent-resume` dataset →
  `restore_checkpoints()` in the notebook. No credentials in notebooks, bundles or datasets.
- Smoke v17 completed 20 SFT steps (loss 1.59 → 0.61) but its step-20 eval OOM'd on reference linear-attention
  kernels. v19 adds fused kernels + pin checks and is queued on Kaggle.

## 5. What is NOT done yet

- No full SFT or GRPO weights exist; v19 smoke has not started, so fused-kernel eval is unverified.
- `ours_base`, `ours_sft`, `ours_harness`, `ours_grpo` eval arms have not run; the manifest has only 3 replay arms.
- Big hosted-model journals for the chosen cheap models (`gpt-6-luna` then `claude-haiku-5-5`, reasoning ON, run as
  hosted task `thermo-teacher-tools-s00` via `scripts/run_teacher_hosted.ps1 -m <slug>`) are pending;
  `gemini-3.5-flash-lite` is excluded and must not consume quota. The hosted task is at version 3. **REMINDER:**
  revisit `claude-sonnet-5-5` later — it was deferred as too expensive and needs Reza's explicit go-ahead.
- Q1 ablations, significance tests, GPU-hour totals, memory ablation, cost-quality frontier remain pending
  (`paper/Q1_CHECKLIST.md`). Three externally drawn figures still show pilot samples.
- No dedicated thermo LLM id was supplied; `other_model` is a GPT-nano stand-in.
- Git push is blocked by an invalid `gh` credential (local commits exist).

## 6. Key decisions

1. Verifier-gated teacher data, not LLM judging; rejections become hindsight memory.
2. DDP replication (not FSDP/ZeRO), fp16 LoRA (not QLoRA), no architecture change for OOM — chunk the loss.
3. 8k trace gate (user decision) to fit expert-scale cycles without truncation.
4. Additive data policy: old models/corpora are never deleted; new models and big cycles are appended.
5. Environment pinned once on Kaggle, wheels reused from Kaggle output; drift fails the notebook.
6. Per-trace provenance instead of aggregate-only claims.

## 7. Reproduction entry points

```powershell
pwsh scripts/build_corpus.ps1                    # ground truth -> corpora + provenance
python scripts/build_provenance_ledger.py        # ledger + stats + docs
python paper/tools/verify_paper.py --no-build    # LaTeX consistency
pwsh -File scripts/make_kaggle_bundle.ps1        # code dataset
pwsh -File scripts/upload_corpus_dataset.ps1     # corpus dataset
pwsh -File scripts/resume_kaggle_checkpoint.ps1 -Stage sft -Kernel rezanadimi/thermo-train-run
```
