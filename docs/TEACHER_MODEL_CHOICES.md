# Frontier teacher: model choice, cost, and quota-resume decisions

The teacher does not run on our hardware. Kaggle Benchmarks supplies both the model access and the quota
(`kaggle_benchmarks`, Kaggle's own proxy), and the physics tool suite on this machine decides what is true. This
file records *how we choose a model* and *how a run survives daily/monthly limits*, so the choice is reproducible
instead of a name in a notebook cell.

## Decision rule

1. **Quality is measured locally, not assumed.** A teacher answer is useful only if `teacher.validate` accepts it;
   the acceptance rate and the rejection reasons are the quality signal. A cheap model whose answers all fail the
   verifiers is more expensive per accepted training row than a pricier one that passes.
2. **Cost is measured, not quoted.** `kaggle_benchmarks.usage.Usage` on each chat reports `input_tokens`,
   `output_tokens`, `input_tokens_cost_nanodollars`, `output_tokens_cost_nanodollars` and backend latency. The
   teacher notebook writes these fields on every completion line; `teacher.progress.usage_totals` sums them and
   `teacher.transfer.ingest` stores the total (USD and nanodollars) in `reports/frontier_ingest.json`.
3. **Probe before committing.** With `PROBE_MODELS = True` the notebook sends one short prompt to each model in
   `PROBE_SHORTLIST`, records tokens/cost/latency in `model_probe.jsonl`, and selects the cheapest model that returns
   a usable answer. The probe is opt-in because the calls themselves consume quota.
4. **Keep the decision with the data.** The chosen `model` id is written on every completion line, on every accepted
   corpus row, and on every failure-memory note. A change of model never silently mixes provenance.

## Measured availability (verified 2026-11-19)

Which route a model uses is a fact about availability, not a preference. The SDK's `kbench.llms` exposes exactly
8 keys in this account; every other slug is reachable **only** as a hosted task via `-m <slug>` (per
`docs/DECISIONS.md`; no network probe was run offline).

| Route | Reachable slugs |
|---|---|
| Local SDK — `kbench.llms` (8 keys) | `anthropic/claude-sonnet-5@default`, `deepseek-ai/deepseek-r1-0528`, `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`, `ibm/granite-4.0-h-small`, `openai/gpt-5.4-nano-2026-03-17`, `openai/gpt-oss-120b`, `qwen/qwen3-next-80b-a3b-instruct` |
| Hosted task only (`-m <slug>`) | `gpt-6-luna`, `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5-default` (opus-5-5), `gemini-4-argon`, `gpt-6.2-sol` |

**Rule: choose the route by availability, never rely on `kbench.llms.get` fallback.** That call silently falls back
to the account default on a miss, so the local SDK driver hard-fails when the resolved `llm.name` is not the
requested model instead of writing rows mislabelled with a model that did not produce them. Consequently
`gpt-6-luna`, `claude-haiku-5-5` and the whole expensive tier run on the hosted route
(`scripts/run_teacher_hosted.ps1` / `thermo-teacher-tools-s00`, see `docs/TEACHER_CAMPAIGN.md`), never on the local
SDK driver (`scripts/run_teacher_campaign.ps1`).

## Current campaign order (budget first, then expensive)

The current campaign runs in this order, budget first (see `docs/TEACHER_CAMPAIGN.md` for the two routes):

1. **Budget tier:** `gpt-6-luna`, `claude-haiku-5-5` — reasoning ON. These are hosted-only slugs, so they run on
   Route A (`scripts/run_teacher_hosted.ps1`, task `thermo-teacher-tools-s00`), never on the local SDK driver.
2. **Expensive tier:** `gpt-6.2-sol`, `claude-sonnet-5-5`, `claude-opus-5-5-default` (the `claude-opus` family) and
   `gemini-4-argon` (the `gemini-4` family) — also hosted-only; only after the budget tier has reported its
   verifier-accepted reasoning-trace yield, and only with Reza's explicit go-ahead for Sonnet.

**Do NOT spend quota on `gemini-3.5-flash-lite`.** Per Reza's directive it is the dumbest of the three budget models;
avoid it entirely on both routes.

**Standing reminder:** revisit `claude-sonnet-5-5` later — the user deferred it as too expensive. Decide after the
budget models report how many accepted reasoning traces they yield. Do not spend on the expensive tier before that
check.

The cheap models chosen for this round are **`claude-haiku-5-5` and `gpt-6-luna`** (hosted-only slugs), with
reasoning **ON** (`reasoning="low"`, 1,536 output tokens) so reasoning traces are captured where the provider exposes
them; the probe shortlist is those two and may pick among them. `gemini-3.5-flash-lite` is excluded — see the
do-not-use note above. The old `MODEL = anthropic/claude-sonnet-5-5@default`
default is retired (Sonnet was too expensive and produced 0 usable rows in the earlier pass). **REMINDER: revisit
`claude-sonnet-5-5` later** — the user explicitly deferred it; decide after the cheap models report how many
accepted reasoning traces they yield. After the first ingest, fill this table from `reports/frontier_ingest.json` and
`data/memory/failure_reasons.json`:

| Model | Chosen for | Total cost (full pass) | Cost / accepted row | Accepted / attempts | Main rejection reason | Source |
|---|---|---|---|---|---|---|
| `google/gemini-3.1-flash-lite-preview` | primary teacher; highest acceptance per dollar | $1.090957 (1,090,957,000 nanodollars) | $0.002127 | 513 / 1,600 (32.1 %) | output format (ranking ids, design keys, `<fix>` block) dominates the 965 graded rejections | `data/teacher_runs/completions.jsonl`, `data/frontier/`, `data/memory/failure_reasons.json` |
| `openai/gpt-5.4-nano-2026-03-17` | second arm / domain-model stand-in | $0.658021 (658,021,050 nanodollars) | $0.005393 | 122 / 1,600 (7.6 %) | same verifier gates; lower rate makes it 2.5x dearer per accepted row | `data/teacher_runs/completions_gptnano.jsonl`, `data/frontier/`, `data/memory/failure_reasons.json` |

Total full-pass spend: **$1.748978**. Both journals carry a model id and usage fields on every line; accepted
rows are identified by id suffix in `data/frontier/*_train.jsonl` (audit rows: none accepted by either model).
Of the 635 accepted rows, all **635** survive the 8,000-token prompt gate (`max_length` 8,192) and are the AI-frontier
rows present in the merged corpora `data/sft_all` and `data/grpo_all`. The old **460** count is legacy: it was the
subset that survived the superseded 4,000-token gate and was present only in `data/sft` / `data/grpo`.

## Big-model hosted campaign (user-selected, additive)

The hosted task `thermo-teacher-big-s00` reuses the 974 base ids not already accepted, so its journals are
additive (ingest dedupes on base id). Reasoning is **ON** (`reasoning="low"`, 1,536 output tokens): the user wants
reasoning traces where the provider exposes them, and leaving reasoning on is acceptable where traces cannot be
captured. Reasoning also avoids the `thinking:{type:disabled}` 400 that `reasoning="none"` triggered on Claude.

| Model | Status | Verifier result | Source |
|---|---|---|---|
| `google/gemini-3.8-flash` | run complete, ingested | 200 attempted, 33 accepted, 167 rejected | `data/teacher_runs/completions_big_gemini-3.8-flash.jsonl` |
| `anthropic/claude-sonnet-5-5@default` | attempted, produced 0 usable rows | all 5 calls 400'd on the thinking parameter | `data/teacher_runs/completions_big_claude-sonnet-5-5-default.jsonl` |
| `claude-haiku-5-5` | round 2 (cheap Claude chosen by user) | pending | hosted task v3 |
| `gpt-6-luna` | round 2 (cheap GPT, chosen by user) | pending | hosted task v3 |

**Do NOT use `gemini-3.5-flash-lite`:** per Reza's directive it is the dumbest of the three budget models; it is
excluded from the budget tier and no quota should be spent on it.

**REMINDER (from the user):** revisit `claude-sonnet-5-5` in a later round after the cheap models report how many
verifier-accepted reasoning traces they yield; the user deferred Sonnet as too expensive for now.

Round-2 driver: `scripts/run_big_teacher_v2.ps1`; logs `logs/big_teacher_v2.log`. Decide on further models only after
the accepted-CoT yield of these three is known.

## Quota limits and changing account

A 1,600-prompt pass may exceed a daily or monthly Kaggle Benchmarks allowance. The run is resumable across accounts:

- The notebook appends one line per attempt to `/kaggle/working/completions.jsonl`, which is lost when the session
  ends. Every 25 attempts it pushes the merged history to `Reza-Nadimi/thermo-agent-checkpoints` under
  `teacher/progress.jsonl`, and it pushes a final merge when the loop ends. Failures (including quota errors) are
  kept as records with an `error` field; a later session does not retry an id that already has a usable answer.
- On start, the notebook pulls that file before computing `done`, so a fresh session with fresh quota skips what the
  previous account already answered. `teacher.progress.merge` is idempotent and keeps distinct attempts.
- To write to the same repo from a second Kaggle account, either reuse the same `HF_TOKEN` secret or add that
  account as a collaborator on the single private repo. `hf_sync.resolve_repo` refuses every other repo id, so a
  new account cannot accidentally create a second history under its own namespace.
- Download `completions.jsonl` from whichever account finished the pass and ingest it locally; ingest is itself
  resumable and skips ids already accepted into `data/frontier`.

## Grading and failure memory

`teacher.transfer.ingest` re-solves every plant and keeps only answers that pass the tool suite. Rejected answers
become failure-memory notes (`data/memory/frontier_notes.jsonl`). Since the last update, ingest also writes
`data/memory/failure_reasons.json` with counts by task, fault type and lesson, and adds synthetic
"Seen N times: ..." notes for lessons that recur, so the retrieval layer is more likely to show the common failure
modes. The same counts are included in the ingest report for the paper.

## Frontier in our harness (second comparison arm)

A separate notebook, `kaggle/teacher_harness.ipynb` (source `kaggle/teacher_harness.py`), runs the same prompts with
our physics tools exposed to the frontier model through `llm.prompt(..., tools=[...])`. It records tool calls,
reasoning traces and `Usage` cost in `harness_completions.jsonl`, under a different progress prefix so the two arms
never overwrite each other. This gives the paper three distinct frontier arms: zero-shot, with our inference text,
and with our tools — plus the trained 4B model and the non-model baselines.

## Measured decision (2026-11-18, CLI proxy)

A short probe through `python -m thermo_agent.teacher.cli_runner probe` tested the eight models in this account's
interactive roster. Two were unavailable (404: `deepseek-ai/deepseek-r1-0528`, `qwen/qwen3-next-80b-a3b-instruct`),
one returned no message content (`openai/gpt-oss-120b`), and `ibm/granite-4.0-h-small` returned a `<think>` block
that consumed the output budget. Four models answered cleanly. The two cheapest clean models were then graded on 12
real Repair prompts through `teacher.transfer.ingest` (source files `data/teacher_runs/probe_gemini.jsonl` and
`probe_gptnano.jsonl`):

| Model | Accepted / attempts | Cost (USD) | Cost per accepted row | Decision |
|---|---|---|---|---|
| `google/gemini-3.1-flash-lite-preview` | 10 / 12 | 0.00549 | 0.00055 | **primary teacher** |
| `openai/gpt-5.4-nano-2026-03-17` | 6 / 12 | 0.00367 | 0.00061 | second arm for comparison |

The choice is the lower cost per *accepted* row, not the lower token price: Gemini's higher acceptance rate more than
pays for its slightly higher token price. The full campaign therefore runs Gemini over all 1,600 prompts and GPT nano
over the same prompts as a second frontier arm. The primary run is resumable from
`data/teacher_runs/completions.jsonl` and from `teacher/progress.jsonl` in the private HF repo.

## Full-run result (2026-11-18)

Both passes completed all 1,600 prompts through the Kaggle CLI proxy with no quota pause. The local ingest graded
every answer; the figures below are from `reports/frontier_ingest.json` and the two journals in `data/teacher_runs/`.

| Model | Attempts | Accepted | Acceptance | Spend | Spend / accepted row | Main rejection reason |
|---|---|---|---|---|---|---|
| `google/gemini-3.1-flash-lite-preview` | 1,600 | 513 | 32.1 % | $1.091 | $0.00213 | `format` (771), then `grounding` (152) |
| `openai/gpt-5.4-nano-2026-03-17` | 1,600 | 122 | 7.6 % | $0.658 | $0.00539 | `format` (686), then `exact` (153) |

The primary model is 2.5x cheaper per accepted training row despite costing more per token, which confirms the
probe-based choice. Accepted rows from both models are appended to `data/frontier/` with their model tag; rejected
attempts feed `data/memory/frontier_notes.jsonl` and the frequency summary in `failure_reasons.json`. A second,
held-out export (`data/frontier_eval_export`) is being answered with the same models so the frontier arms can be
replayed through the evaluator on test rows rather than on the rows used to build the corpus.
