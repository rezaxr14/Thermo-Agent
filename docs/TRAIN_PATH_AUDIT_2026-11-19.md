# Training-path audit and fixes — 2026-11-19

Scope: Phase 0.7 + Phase 1.1 of `docs/IMPLEMENTATION_PLAN.md`, checklist DECISIONS §4.4 items 1–7.
No training run was started (datagen is incomplete), no Kaggle command pushed/ran anything, no
token was printed or written to a file. All work is CPU/local and additive.

## Method

* Read the decided values in `docs/DECISIONS_2026-11-19.md` and the binding order in
  `docs/IMPLEMENTATION_PLAN.md`.
* Read every training module under `thermo_agent/train/**` and the Kaggle train runner/smoke kernel.
* Inspected the *actual* corpus shards under `data/sft*`/`data/grpo*` for the columns and prompt
  text the loaders claim to read.
* Fetched the v28 kernel log **read-only** with `kaggle kernels logs rezanadimi/thermo-train-smoke`
  (proxy set only for that command; `PYTHONIOENCODING=utf-8` to avoid the console `charmap`
  failure; the token was read from `.env` into an env var and never printed).
* Reproduced each suspected defect with a CPU test, fixed it, and re-ran the tests.

## Findings and fixes

### 1. Data loading reads `task` AND `tier`; uniform task × tier GRPO sampling — **DEFECT, FIXED**

**Root cause.** No real corpus row has a `tier`, `difficulty` or `level` column (verified on
`sft_all`, `sft_tools`, `sft_tools_h`, `grpo_all`, `grpo_tools`, `grpo_tools_h`). `grpo.row_tier`
therefore returned `<missing>` for 100% of rows and `sample_joint` visited 4 cells (one per task)
rather than 12 (task × tier). The docstring advertised uniform task × tier sampling; the code did
task-only sampling. Separately, `sft.read_rows` and `grpo.joint_rows` read only
`<task>_<split>.jsonl`, silently skipping `frontier_<task>_<split>.jsonl` (DECISIONS E1 puts accepted
frontier tool traces first in the mix).

**Fix.** New `thermo_agent/train/data.py` is the single loader for both stages:

* explicit `tier`/`difficulty`/`level` field wins and is validated against `synth.tiers.TIERS`
  (a typo is a hard error, not a new bucket);
* otherwise tiers are derived with `synth.tiers.assign_tiers_grouped` over the whole task slice —
  equal thirds, remainder to the higher tier, the two variants of a plant stay together;
* unscorable rows stay `<missing>` and are counted; `joint_rows`/`read_rows` call
  `assert_tiers_resolved` and refuse to train on them;
* `frontier_*` shards are loaded alongside the deterministic shards.

`thermo_agent/train/grpo.py` (`joint_rows`, `load_rows`, `row_tier`) and `thermo_agent/train/sft.py`
(`read_rows`) now delegate to it. `grpo.row_tier` and `grpo.UNKNOWN_TIER` remain as aliases so older
callers keep working.

**Tests.** `tests/test_train_data.py` (7 tests): equal thirds without a column, plant-pair grouping,
explicit-wins/typo-raises, unscorable-row refusal, 12-cell uniform sampling, frontier shard loading,
SFT task/tier attachment. Existing `tests/test_grpo_joint_and_plumbing.py` still passes unchanged.

### 2. Single compact harness; no bookkeeping in model prompts — **DEFECT, FIXED**

**Root cause.** `render_pair` fed `row["prompt"]` straight to the chat template with no check that
it came from the one J2/J3 builder. Inspection showed the checked-in `sft_tools_h` / `grpo_tools_h`
shards are **stale**: their system prompt contains `## Components` (the retired verbose builder)
and not the compact marker `# AGENTS.md: energy-plant tasks (compact)`. Training on them would
teach a prompt different from eval's compact-harness prompt and would push 8,192-token rows over the
T4 memory budget. Nothing rejected a message dict carrying extra bookkeeping keys either.

**Fix.** `thermo_agent/train/render.py` now has `model_messages()` (only `role`/`content` reach the
tokenizer; malformed messages are a hard error) and `assert_compact_harness()` (a harness row must
carry the compact marker and no verbose marker). `render_pair` and `prefix_report` use
`model_messages`; both `sft.main` and `grpo.main` validate every loaded row before training, so a
stale shard fails loudly instead of silently training the wrong prompt. The datagen rebuild of the
`*_h` corpora (owned by the datagen agent) is still required to make those shards pass.

**Tests.** `tests/test_train_render.py` (7 tests), including a fresh compact prompt from the shared
`build_harness_tool_messages(compact=True)` for all four tasks, rejection of a verbose prompt,
metadata stripping, and wiring checks in both trainers.

### 3. SFT label masking — **VERIFIED, no code change**

`preprocess_row` masks the prompt by token-id prefix and every `<tool_result>`/`<tool_response>`
span by character offsets (tags included, nested/unclosed handled), keeps tool *calls* trainable,
sets `completion_only_loss=True`, and the corpus gate stays 8,192 (dropped rows never truncated).
Existing `tests/test_tool_loss_masking.py` plus a new multi-turn integration assertion in
`tests/test_train_collator.py` cover this.

### 4. Collator, batching and length stats — **VERIFIED + TESTS ADDED**

Inspected TRL 1.18.1 `DataCollatorForLanguageModeling`: it pads `input_ids` with the pad id, builds
`attention_mask`, and pads `labels` with -100. `tests/test_train_collator.py` now checks a batch of
two variable-length multi-turn traces end-to-end (prompt and tool spans masked, padding -100,
attention mask zero on pads) and that `token_audit.audit` reports prompt/completion/total length
statistics plus a new per-file `tier_counts` breakdown.

### 5. Checkpoint/resume and stage names — **DEFECT, FIXED**

`checkpoint.latest()` picked the highest `checkpoint-N` directory without checking that a save had
finished; a session killed mid-save could resume from a directory with no optimizer, RNG or weights,
silently restarting those states under a "resumed" label. Added `checkpoint.is_complete()` and
`latest(out, complete=True)`; both `sft.main` and `grpo.main` now use the strict form. Save/eval
cadence is already 50/50 (`SftSettings`), best-model loading and early stopping are configured, and
`grpo.stage_name` writes distinct readable `grpo_k4`/`grpo_k8`/`grpo_k16` names (legacy names are
read-only aliases).

**Tests.** `tests/test_checkpoint.py` extended with completeness and interrupted-save cases; stage
names remain covered by `tests/test_grpo_joint_and_plumbing.py` and `tests/test_grpo_stage_names.py`.

### 6. Eval/save cadence and missing adapter — **VERIFIED, no code change**

`SftSettings.save_steps == eval_steps == 50`; GRPO checkpoints on `save_steps`. `kaggle/eval_runner.py`
already defines `_has_adapter_weights`/`require_adapter` and calls them for every trained arm, so a
missing or empty adapter is a `SystemExit` rather than a base-model score under a GRPO label.
Those files are outside this subagent's ownership and were read/verified, not edited.

### 7. Smoke kernel v28 (read-only triage) — **OOM DEFECT, LOCAL FIX; RERUN BLOCKED**

Fetched log via `kaggle kernels logs` (read-only). Timeline: setup and env-pin OK (torch 2.11.0+cu130,
transformers 5.23.0, TRL 1.18.1, 2×T4, sdpa resolved), SFT launched with
`--max-steps 20 --limit-rows 64 --per-device-batch 1 --grad-accum 8 --loss-token-chunk 512
--loss-vocab-chunk 8192`, then at step 0/20 the first backward failed:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.10 GiB. GPU 0 has a total
capacity of 14.56 GiB of which 1.01 GiB is free. ... 13.55 GiB memory in use; 12.52 GiB
allocated by PyTorch ... PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ...
thermo_agent.train.sft FAILED   (PapermillExecutionError at In [10])
```

The chunked LM head was active (`[loss] chunked cross-entropy: 512 x 8192 ...`), so the failing
1.10 GiB allocation is the per-layer activation working set of an 8,192-token row during backward,
not the full-vocabulary logits. `transformers` 5.23 supports pinning those saved activations to
host memory (`activation_checkpointing_kwargs={"offload": True}`), reached through the deprecated
alias already used by both trainers.

**Fix.** `config.activation_checkpointing_kwargs()` returns
`{"use_reentrant": False, "offload": True}`; both `sft.py` and `grpo.py` pass it as
`gradient_checkpointing_kwargs`. No change to weights, labels, loss, gate or batch settings.
**Tests.** `tests/test_train_memory.py` (3 tests).

**Remaining blocker for a green re-run (not fixable in this task without touching out-of-scope
files):** the checked-in smoke notebook still selects `SFT_DATA=sft_tools` / `GRPO_DATA=grpo_tools`,
i.e. plain-prompt corpora, not compact-harness tool rows as G1 requires; and the available `*_h`
shards are still the stale verbose-harness build described in finding 2. The smoke notebook's env
cell is pinned byte-for-byte by `tests/test_smoke_kernel_config.py` (not owned here) and the `*_h`
rebuild belongs to datagen, so neither can be changed by this subagent. With this fix the OOM is
addressed locally; the actual green re-run must wait for datagen and an authorised Kaggle push.

## Test results

All CPU-only. Per-file counts from the final run:

| File | Result |
|---|---|
| `tests/test_train_data.py` (new) | 7 passed |
| `tests/test_train_render.py` (new) | 7 passed |
| `tests/test_train_collator.py` (new) | 3 passed |
| `tests/test_train_memory.py` (new) | 3 passed |
| `tests/test_checkpoint.py` | 9 passed |
| `tests/test_token_audit.py` | 4 passed |
| `tests/test_grpo_joint_and_plumbing.py` | 15 passed |
| `tests/test_grpo_multiturn_wiring.py` | 13 passed |
| `tests/test_grpo_prompt_consistency.py` | 8 passed |
| `tests/test_grpo_stage_names.py` | 6 passed |
| `tests/test_tool_loss_masking.py` | 8 passed |
| `tests/test_render.py` | 2 passed |
| `tests/test_smoke_kernel_config.py` | 2 passed |
| `tests/test_ddp.py` | 6 passed |
| `tests/test_chunked_loss.py` | 7 passed |
| `tests/test_decisions_applied.py` | 4 passed |
| `tests/test_eval_runner_arms.py::test_h_missing_grpo_adapter_raises` | 1 passed |
| **Combined run of the 16 train-related files above** | **104 passed** |
| **All 23 test files importing `thermo_agent.train`** | **174 passed** |

Full-repository `pytest -q`: **15 failed, 845 passed, 9 skipped in 1468 s**. All 15 failures are in
files owned by the concurrent datagen work (not this audit): `test_check_datagen_complete.py` (1),
`test_heldout_v2_freeze.py` (3), `test_heldout_v2_overlap.py` (2), `test_heldout_v2_tiers.py` (8),
and `test_kaggle_notebook.py::test_notebook_writes_one_line_per_attempt_and_records_errors` (1,
which picked up the real 4,149-attempt shared-progress corpus instead of the two fixture rows).
No failure is in any train-path file changed here.

## Files changed

* `thermo_agent/train/data.py` (new) — shared task+tier loader, frontier shards, tier guard.
* `thermo_agent/train/grpo.py` — delegate loading to `data.py`; compact-harness guard; offload; strict resume.
* `thermo_agent/train/sft.py` — delegate loading to `data.py`; compact-harness guard; offload; strict resume.
* `thermo_agent/train/config.py` — `activation_checkpointing_kwargs()` (v28 OOM fix).
* `thermo_agent/train/render.py` — `model_messages`, `assert_compact_harness`, sanitised rendering.
* `thermo_agent/train/checkpoint.py` — `is_complete`, `latest(complete=True)`.
* `thermo_agent/train/token_audit.py` — per-file `tier_counts` length-stat breakdown.
* `tests/test_train_data.py`, `tests/test_train_render.py`, `tests/test_train_collator.py`,
  `tests/test_train_memory.py` (new) — one regression test per fix.
* `tests/test_checkpoint.py`, `tests/test_token_audit.py` — extended for the new behaviours.

No file outside the owned set was edited. No commit was made.
