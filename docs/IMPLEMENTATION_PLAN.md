# Implementation plan — trace-first campaign

Status: **draft, blocked on `docs/OPEN_PARAMETERS_FOR_REVIEW.md` answers.**
This file contains no chosen hyper-parameters. Every number is a reference to a parameter ID.

Order is fixed by the user: **make the AI traces first, then train.** Nothing in phases 3-6 starts early.

---

## Phase 0 — verify the pipeline actually accepts our inputs

Goal: prove a model accepts the exact payload (tools + reasoning + messages + memory) before any large run.

- [ ] 0.1 Smoke kernel green (G1). Read the v28 log, fix the failure, re-run until `sdpa` and loss fall are
      visible in the log.
- [ ] 0.2 Hosted-task pilot per model, 40 rows (D3 unchanged), body: tools + skills + memory.
      Success = journal rows with `tool_calls > 0`, no `BadRequestError`, `verdict.graded` true after ingest.
- [ ] 0.3 Record the exact model slug, reasoning mode, and cost of each pilot in
      `docs/TEACHER_CAMPAIGN_LOG.md`.
- [ ] 0.4 Any model that rejects the payload is dropped from the campaign, not retried blindly.

Gate: 0.1 + at least one 0.2 pass. No bulk run before both.

## Phase 1 — bulk trace generation

- [ ] 1.1 Bulk tier (luna, haiku) at the answered D2 counts, D3 rows/run, D4 rounds.
- [ ] 1.2 Download each completed journal → append id-deduped to `data/teacher_runs/hosted_<model>.jsonl`.
- [ ] 1.3 `python -m thermo_agent.teacher.transfer ingest …` with the harness prompt (E4).
- [ ] 1.4 Premium tier only after 1.1-1.3 show non-zero tool-call yield and per-model cost is recorded (D7).
- [ ] 1.5 Stop at the D1 accepted-trace cap.

Gate: accepted tool traces ≥ the D1 target, every row with ≥1 `<tool_call>` + `<tool_result>`.

## Phase 2 — corpus assembly (additive only)

- [ ] 2.1 Re-verify `data/sft_tools_h` / `data/grpo_tools_h` rows are harness-marked (960/960 today).
- [ ] 2.2 Build the mix with `--harness-prompt --frontier-per-task <D9> --out-sft data/sft_mix_h --out-grpo data/grpo_mix_h`.
- [ ] 2.3 Real-tokenizer audit of the mix at the 8,192 gate (E3): drop, never truncate.
- [ ] 2.4 Contamination check against `data/heldout` must show zero overlap.
- [ ] 2.5 Manifest records the exact frontier ids used and the sha256 of every input.

Gate: mix manifest exists, over-8k count recorded, contamination zero.

## Phase 3 — SFT (only if A1/A8 answered)

- [ ] 3.1 Launch with `SFT_DATA` = A8, epochs/steps = A1/A2.
- [ ] 3.2 Resume across sessions via G3; never restart from zero.
- [ ] 3.3 Merge adapter → `runs/sft_merged`.

## Phase 4 — GRPO (only if C1 answered)

- [ ] 4.1 One run per answered k, distinct stage names `grpo_<task>_k<k>`.
- [ ] 4.2 Merge each adapter.

## Phase 5 — held-out evaluation (F1/F2)

- [ ] 5.1 Score every arm in `kaggle/eval_runner.py` SPECS on `data/heldout` with `LIMIT=0`.
- [ ] 5.2 Reports must show per-arm `harness`, `tool_access`, `zero_shot`, `enable_thinking`.
- [ ] 5.3 Reject any report whose scored id count differs from the split manifest.

## Phase 6 — artifacts

- [ ] 6.1 HF corpus sync (H1/H2), Kaggle dataset version (H3) — dry-run first, no tokens.
- [ ] 6.2 Regenerate provenance + paper numbers from the actual reports only (H4).
- [ ] 6.3 Update `docs/DIRECTIVE_COMPLIANCE_AUDIT_2026-11-19.md` §2 with final statuses.

---

## Explicit non-goals

- No architecture change, no QLoRA/bf16/FSDP, no corpus deletion.
- No training until traces exist (user instruction, 2026-11-19).
- No Kaggle run launch before Phase 0 passes.
- No hyper-parameter chosen by the assistant.
