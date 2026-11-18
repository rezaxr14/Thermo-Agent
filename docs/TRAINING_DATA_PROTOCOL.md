# Training-data protocol (user decision, 2026-11-18)

This is a binding decision for the remaining campaign. It supersedes the earlier zero-shot-only teacher pass for
*training data*. All old data is kept; everything below is additive.

## Why (measured gap)

- Accepted frontier CoTs to date: **635** (Gemini 3.1 Flash Lite 513, GPT-5.4-nano 122). All **635** survive the
  8,000-token prompt gate (`max_length` 8,192) and are present as frontier train rows in **both** merged corpora,
  `data/sft_all` and `data/grpo_all` (verified: 635 `source=frontier` rows in `data/sft_all/*_train.jsonl`). The old
  **460** figure was a legacy count: it applied only to the superseded 4,000-token-gated `data/sft` / `data/grpo`
  corpora, not to the merged ones. Additionally **33** from `gemini-3.8-flash` were ingested. Every one of these was
  produced with `harness="none"`, `tools="none"` — they are **zero-shot** answers, not tool-using reasoning.
- A separate tool arm (`frontier_tools`) recorded 120 answers / 242 `check_answer` calls, but those rows are an
  **evaluation** artifact (`reports/eval_frontier_tools.json`), not training data, and the only tool exposed was a
  generic `check_answer` — there were no first-law, second-law or validity tools.
- Therefore the model currently has **no verifier-tool-using traces to imitate**. This decision closes that gap.

## The decision, in order

1. **Frontier zero-shot baseline on a dedicated held-out test set.** Freeze a new test set that no cohort has seen:
   `data/heldout`, generated with the fresh ground-truth seed **987654**, excluded from every `data/processed*`,
   `data/sft*`, `data/grpo*` corpus and from every teacher prompt shard. Score frontier models zero-shot on it first;
   this is the *frontier baseline*. All comparison arms below are scored on this same `data/heldout` set — frontier
   zero-shot, frontier+harness+tools+skills+memory, other thermo LLMs, our SFT, and our SFT+GRPO with k=4, 8 and 16
   (see `docs/USER_DIRECTIVES.md`, D8).
2. **Frontier models inside our harness produce CoT.** Re-run the frontier models with our physics tools exposed
   (`check_first_law`, `check_second_law`, `check_validity`, and generic `check_answer`), keeping the reasoning trace
   and every tool call. These are the **tool-using CoT traces**.
3. **Train on those CoTs plus the deterministic synthetic generation.** The merged SFT corpus = synthetic ground
   truth (unchanged) + verifier-accepted zero-shot frontier rows (kept) + the new verifier-accepted tool-using CoT
   rows. GRPO prompts come from the same held-out ground truth. Nothing old is deleted.
4. **Evaluate on the same held-out test set and record our score.** This makes the baseline (step 1) and our model
   (step 4) directly comparable on identical, previously unseen rows.
5. **Compare against other thermo models on that set and record the result** (`other_model` arm and any domain LLM
   ids supplied later).

## Tool contract (what "uses tools" means)

The tools must be the real physics checks, exposed to the model during generation, with feedback returned to the
model so it can self-correct. From `thermo_agent/verify/verifiers.py`:

The exposed tool names are those in `thermo_agent/teacher/toolkit.py`:

| Tool | Verifier checks / numbers it reports |
|---|---|
| `first_law_check` | per-component mass residual (kg/s) and energy residual (W), exergy closure, heat/power/fuel balance (`V1`, `V2`) |
| `second_law_check` | per-component exergy destruction (MW), non-negativity, exergy closure, eta_I/eta_II and Carnot/COP limits (`V3`, `V5`) |
| `validity_check` | unit/range violations, site constraints, minimum approach / pinch (K), failed checks and overall `valid`/`failed` (`V4b`, `V6`, `V8`) |
| `solve_plant` | full numeric stream state table (T, P, h, s, exergy) of the candidate plant |
| `check_answer` | full accept/reward/reasons verdict (kept for backward compatibility) |

A trace is "tool-using" only if it contains at least one of these calls **and** a subsequent revised final answer;
both are stored on the journal line (`tool_calls`, `reasoning`, `completion`).

## Model selection for this round (user decision)

- Cheap models now: `claude-haiku-5-5`, `gemini-3.5-flash-lite`, `gpt-6-luna`.
- Reasoning **ON** (`reasoning="low"`). Capture reasoning where the provider exposes it; leaving it on where it cannot
  be captured is acceptable.
- **REMINDER:** revisit `claude-sonnet-5-5` later — deferred as too expensive; decide after the cheap models report
  how many accepted reasoning traces they yield.
- Add other models only after seeing the accepted-CoT yield of these three.

## Where this is implemented

- Hosted zero-shot / reasoning journals: `kaggle/teacher_big_task_s00.py`, `scripts/run_big_teacher_v2.ps1`.
- Tool-using CoT journals: `kaggle/teacher_harness.py` (physics tools + `check_answer` + reasoning capture).
- Ingest is additive through `thermo_agent.teacher.transfer.ingest --harness tools`; provenance recorded per trace in
  `data/provenance/trace_ledger.jsonl` with origin model, harness mode, tool calls, cost and acceptance.
