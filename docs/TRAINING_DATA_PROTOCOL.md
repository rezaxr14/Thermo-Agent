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
- A separate tool arm (`frontier_tools`) recorded 120 answers / 242 generic answer-check calls, but those rows are an
  **evaluation** artifact (`reports/eval_frontier_tools.json`), not training data, and the only tool exposed was a
  generic answer checker — there were no first-law, second-law or validity tools.
- Therefore the model currently has **no verifier-tool-using traces to imitate**. This decision closes that gap.

## The decision, in order

1. **Frontier zero-shot baseline on a dedicated held-out test set.** Freeze a new test set that no cohort has seen:
   `data/heldout`, generated with the fresh ground-truth seed **987654**, excluded from every `data/processed*`,
   `data/sft*`, `data/grpo*` corpus and from every teacher prompt shard. Score frontier models zero-shot on it first;
   this is the *frontier baseline*. All comparison arms below are scored on this same `data/heldout` set — frontier
   zero-shot, frontier+harness+tools+skills+memory, other thermo LLMs, our SFT, and our SFT+GRPO with k=4, 8 and 16
   (see `docs/USER_DIRECTIVES.md`, D8).
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

## Tool contract (what "uses tools" means)

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

## Generation invariants (P0-9)

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

## Model selection for this round (user decision)

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

## Where this is implemented

- Hosted zero-shot / reasoning journals: `kaggle/teacher_big_task_s00.py`, `scripts/run_big_teacher_v2.ps1`.
- Tool-using CoT journals: `kaggle/teacher_harness.py` (physics tools + reasoning capture).
- Ingest is additive through `thermo_agent.teacher.transfer.ingest` (tool use is detected from the record itself;
  there is no `--harness` flag) — e.g. `python -m thermo_agent.teacher.ingest data/frontier_export
  harness_completions.jsonl data/frontier --model gpt-6-luna`. Provenance is recorded per trace in
  `data/provenance/trace_ledger.jsonl` with origin model, harness mode, tool calls, cost and acceptance.

## 2026-11-19 — tool-output masking coverage (sft_tools)

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
