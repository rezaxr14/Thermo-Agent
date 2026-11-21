# Teacher (frontier AI) campaign: models, usage, audits

Models, budget rules, hosted-run assessments, token/memory efficiency. Generated batch logs live in `reports/` (teacher_campaign_log.md, budget_batches.md).

## Teacher campaign runbook (tool-using CoT, budget tier first)

*(merged from `docs/TEACHER_CAMPAIGN.md` on 2026-11-20; content kept, headings demoted one level)*

There are **two different routes** to the tool-using teacher campaign, and they are not
interchangeable. **Model availability decides the route:** the big slugs (`gpt-6-luna`,
`claude-haiku-5-5` and the whole expensive tier) are reachable **only** as a hosted Kaggle
Benchmarks task (`-m <slug>`). The local SDK driver sees only 8 keys and silently falls back
on a miss, so it must never be asked for a model outside that list. Route A is therefore the
route for luna / haiku / expensive; Route B is for the 8 SDK keys only. Both routes ingest
locally and share the same acceptance authority (`transfer.ingest`) and provenance destinations.

### Route A — hosted task `thermo-teacher-tools-s00` (use this for luna / haiku / expensive)

`scripts/run_teacher_hosted.ps1` pushes the hosted task `thermo-teacher-tools-s00`
(source `kaggle/teacher_harness_task_s00.py`), runs it once per hosted model with `-m <slug>`,
downloads that model's journal, appends it to a **per-model shard**, and ingests the shard with
`thermo_agent.teacher.transfer.ingest`. **Dry-run is the default:** no push, no run, no download,
no ingest and no API call happens unless `-Execute` is passed.

```powershell
# Safe default: print the full plan for the two budget models. No network calls, no writes.
pwsh -File scripts/run_teacher_hosted.ps1 -DryRun

# Actually run the budget models: luna first, then haiku.
pwsh -File scripts/run_teacher_hosted.ps1 -Execute -Models gpt-6-luna,claude-haiku-5-5 -Limit 200 -Rounds 3

# Only after the budget tier is accepted, and only with Reza's go-ahead for Sonnet:
pwsh -File scripts/run_teacher_hosted.ps1 -Execute -Models gpt-6.2-sol,claude-sonnet-5-5 -Limit 200 -Rounds 3
```

Parameters: `-Models` (comma/space separated override; order is respected), `-Task` (default
`thermo-teacher-tools-s00`), `-Limit` / `-Rounds` (set `HARNESS_LIMIT` / `HARNESS_ROUNDS`; 0 means
task default — 200 rows and 4 rounds / `agent_graph.DEFAULT_MAX_ROUNDS`), `-Wait` (per-model wait,
default 5400 s), `-DryRun` (default), `-Execute`. Tokens are read from the environment or `.env`
and are never printed; preflight refuses to start if `KAGGLE_API_TOKEN` is missing.

#### Model order (fixed, budget first)

1. `gpt-6-luna`
2. `claude-haiku-5-5`

Then the **expensive tier, only after the budget tier has reported accepted traces and only with
Reza's explicit go-ahead for Sonnet**: `gpt-6.2-sol`, `claude-sonnet-5-5`
(deferred — needs that go-ahead), `claude-opus-5-5-default` (the `claude-opus-*` family),
`gemini-4-argon` (the `gemini-4-*` family).

**Do NOT spend quota on `gemini-3.5-flash-lite`.** Per Reza's directive it is the dumbest of the
three budget models; avoid it entirely (do not add it to `-Models` on either route).

**Quota guard.** Quota is a single shared allowance, so the budget-first order above is the
primary guard; spending on the expensive tier first would forfeit budget-tier coverage. The
dry-run default and the explicit `-Execute` gate prevent accidental spend. On the hosted side,
`HARNESS_LIMIT` / `HARNESS_ROUNDS` (from `-Limit` / `-Rounds`) plus the harness's
`HARNESS_DEADLINE` and `HARNESS_MAX_COST_USD` bound each run, and the harness pauses on repeated
errors rather than looping forever.

#### Grading authority — INPUTS ONLY, graded at ingest

The hosted task sees **inputs only**: the compact prompt shard carries no plant/spec, so the task
cannot decide whether an answer is correct. Every journal row is therefore written with
`"verdict": {"accepted": null, "graded": false, "reason": "deferred_to_ingest"}` and top-level
`"graded": false`. **`transfer.ingest` is the ONLY acceptance authority.** It re-grades each
completion **by `record_id`** against the answers in `data/frontier_export`, using the local tool
suite, and appends only accepted rows to `data/frontier/`. A completion in a journal is never
evidence of acceptance.

#### Journal fields

Each record in a hosted shard carries: `id` (the `record_id` used at ingest), `task`, `model`,
`harness: "tools"`, `tools`, `completion`, `sft_completion`, `verdict` (the deferred verdict
above), `graded: false`, `trajectory` (`turns`, `notes_used`, `tools_available`, `final`),
`usage` (tokens / cost / latency), `error`, `ts`, `shard`, `row_source`, `spec_sha`.

#### Per-model shards, resume-by-id, append-only

The downloaded journal is appended to `data/teacher_runs/hosted_<slug>.jsonl`
(e.g. `hosted_gpt-6-luna.jsonl`, `hosted_claude-haiku-5-5.jsonl`); downloads are staged under
`data/teacher_runs/kaggle_hosted/<slug>/`. Model slugs are sanitized to `[A-Za-z0-9._-]`, and
each model has its own shard so no model can overwrite another's evidence. Before appending, the
driver reads the shard's existing `id` set and skips ids already present — **resume is by id** and
a partial/interrupted run continues cleanly. Ingest separately skips ids already accepted in
`data/frontier` (`skipped_already_accepted`), so replaying a shard never duplicates a training row.

### Route B — local SDK driver `scripts/run_teacher_campaign.ps1` (8 SDK keys only)

This route uses `scripts/kaggle_harness_cli.py` (LangGraph + real physics tools + fresh failure
memory + daily-quota guard) and ingests each model's shard with
`thermo_agent.teacher.transfer.ingest`. **Use it only for models in this exact 8-key SDK list**
(the `kbench.llms` roster for this account, verified 2026-11-19):

- `anthropic/claude-sonnet-5@default`
- `deepseek-ai/deepseek-r1-0528`
- `google/gemini-3-flash-preview`
- `google/gemini-3.1-flash-lite-preview`
- `ibm/granite-4.0-h-small`
- `openai/gpt-5.4-nano-2026-03-17`
- `openai/gpt-oss-120b`
- `qwen/qwen3-next-80b-a3b-instruct`

**Never use this driver for a model outside that list** — in particular not `gpt-6-luna` or
`claude-haiku-5-5`. `kbench.llms.get()` silently falls back to the account default on a miss, so
the driver checks `llm.name` against the requested model and **hard-fails** (`SystemExit`) on a
mismatch instead of producing rows mislabelled with a model that did not generate them.

```powershell
# Safe default: print the exact driver + ingest commands, generate nothing.
pwsh -File scripts/run_teacher_campaign.ps1 -DryRun

# Actually run a budget SDK model (200 rows), resuming from shards already on disk.
pwsh -File scripts/run_teacher_campaign.ps1 -Execute -Rows 200 -Resume

# One task only; -Resume also skips ids already accepted into data/frontier.
pwsh -File scripts/run_teacher_campaign.ps1 -Execute -Task repair -Rows 200 -Resume

# Only after the budget tier is exhausted and accepted (SDK expensive keys only):
pwsh -File scripts/run_teacher_campaign.ps1 -Execute -Tier expensive -AllowExpensive -Rows 200 -Resume
```

Parameters: `-Models` (comma-separated override), `-Tier budget|expensive`, `-Rows` per model
(default 200; must be ≥ 1 — the driver treats `--limit 0` as an empty slice),
`-Task all|repair|ledger|audit|design` (default `all`), `-Resume`, `-DryRun` (default),
`-Execute`, `-AllowExpensive`, `-Split`, `-ExportDir`, `-OutDir`. `-DryRun` wins if both
`-DryRun` and `-Execute` are passed. Environment tokens are read from the environment or `.env`;
preflight refuses to start if `KAGGLE_API_TOKEN` or `HF_TOKEN` is missing and reports only the key
name, never its value.

#### Shard naming and append-only contract

Each model writes `data/teacher_runs/harness_<model>_<task>.jsonl`, e.g.
`harness_google-gemini-3.1-flash-lite-preview_all.jsonl`. Model/task names are sanitized
to `[A-Za-z0-9._-]`. The driver opens the shard in append mode and never truncates it, so one
model cannot overwrite another's evidence. Ingest appends accepted rows to
`data/frontier/frontier_<task>_<split>.jsonl` and dedupes on `record_id`.

#### Resumability (local SDK)

- Driver level: on start it reads its own shard and skips every id that already has a usable
  completion; a quota pause or time limit leaves a partial shard that the next run continues.
- `-Resume` level: before the model loop, a one-shot Python filter writes
  `data/teacher_runs/_campaign_pending_export/` containing only export rows whose id is **not**
  already accepted in `data/frontier/`. The snapshot is built once so every model in the same
  campaign sees the same pending ids. In `-DryRun` the filter is printed, never executed.
- Ingest level: `transfer.ingest` skips ids already present in `data/frontier` and records the
  count as `skipped_already_accepted`.

#### Quota guard (local SDK)

Live in the driver, not in the orchestrator: before every row `teacher.cli_runner.quota()` is
checked and the loop pauses when daily remaining drops below `--reserve` (default 0.5); it also
stops after `--max-seconds` (default 4 h). The driver catches per-row exceptions and journals
them as `error` records rather than dying, so a failed row does not abort the shard. The
orchestrator checks each native exit code and stops that model's ingest on failure; the shard is
kept intact for the next `-Resume`.

### Where provenance lands (both routes)

- **Corpus:** accepted rows land in `data/frontier/*`, with the provenance twin in
  `data/frontier_prov/*`. Every accepted row carries `source=frontier:<model>`, `model` (slug),
  `provider` (coarse label derived from the slug), and `run_id` when one is supplied. Rejected
  attempts become memory notes tagged `frontier:<model>` in `data/memory/frontier_notes.jsonl`
  and counts in `data/memory/failure_reasons.json`.
- **Living stats:** `scripts/teacher_campaign_stats.py` reads every journal in `data/teacher_runs`
  and writes `data/teacher_runs/CAMPAIGN_STATS.json` (machine-readable) and
  `reports/teacher_campaign_log.md` (append-friendly human log, with older snapshots preserved under
  History). Rows are split into `accepted` / `rejected` / `ungraded`; a hosted completion with a
  deferred verdict counts as `ungraded`, never accepted. The acceptance rate uses only graded
  rows (`accepted / (accepted + rejected)`). Ingest grading is merged in by re-reading the
  journals after each ingest, so the stats reflect re-graded `record_id`s rather than the
  hosted task's deferred placeholders.

### Verify counts

```powershell
# Attempts per model across journals (model field is on every shard line).
.venv/Scripts/python.exe -c "import json,pathlib,collections; print(collections.Counter(json.loads(l).get('model','?') for p in pathlib.Path('data/teacher_runs').glob('*.jsonl') for l in p.open(encoding='utf-8') if l.strip()))"

# Accepted rows per model tag and source in the corpus (id ends in -t-<model>).
.venv/Scripts/python.exe -c "import json,pathlib,collections; print(collections.Counter((json.loads(l).get('source'), json.loads(l)['id'].split('-t-')[-1]) for p in pathlib.Path('data/frontier').glob('frontier_*_train.jsonl') for l in p.open(encoding='utf-8') if l.strip()))"

# Refresh the living stats/log after an ingest.
.venv/Scripts/python.exe scripts/teacher_campaign_stats.py
```

## Frontier teacher: model choice, cost, and quota-resume decisions

*(merged from `docs/TEACHER_CAMPAIGN.md` on 2026-11-20; content kept, headings demoted one level)*

The teacher does not run on our hardware. Kaggle Benchmarks supplies both the model access and the quota
(`kaggle_benchmarks`, Kaggle's own proxy), and the physics tool suite on this machine decides what is true. This
file records *how we choose a model* and *how a run survives daily/monthly limits*, so the choice is reproducible
instead of a name in a notebook cell.

### Decision rule

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

### Measured availability (verified 2026-11-19)

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

### Current campaign order (budget first, then expensive)

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

### Big-model hosted campaign (user-selected, additive)

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

### Quota limits and changing account

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

### Grading and failure memory

`teacher.transfer.ingest` re-solves every plant and keeps only answers that pass the tool suite. Rejected answers
become failure-memory notes (`data/memory/frontier_notes.jsonl`). Since the last update, ingest also writes
`data/memory/failure_reasons.json` with counts by task, fault type and lesson, and adds synthetic
"Seen N times: ..." notes for lessons that recur, so the retrieval layer is more likely to show the common failure
modes. The same counts are included in the ingest report for the paper.

### Frontier in our harness (second comparison arm)

A separate notebook, `kaggle/teacher_harness.ipynb` (source `kaggle/teacher_harness.py`), runs the same prompts with
our physics tools exposed to the frontier model through `llm.prompt(..., tools=[...])`. It records tool calls,
reasoning traces and `Usage` cost in `harness_completions.jsonl`, under a different progress prefix so the two arms
never overwrite each other. This gives the paper three distinct frontier arms: zero-shot, with our inference text,
and with our tools — plus the trained 4B model and the non-model baselines.

### Measured decision (2026-11-18, CLI proxy)

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

### Full-run result (2026-11-18)

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

## API usage, data provenance, and project status

*(merged from `docs/TEACHER_CAMPAIGN.md` on 2026-11-20; content kept, headings demoted one level)*

This is an audited record of which hosted models were called, what their outputs became, and how far the
project has come. Counts were obtained by streaming `json.loads` over the `.jsonl` files (see the commands in
§6); no number here is transcribed from a stale prose claim. The authoritative corpus report is
`data/sft/sft_report.json` / `data/grpo/sft_report.json` written by `synth/combine.py`.

### 1. Hosted models actually called

All frontier calls went through **Kaggle Benchmarks model access / quota**, not vendor keys. Prompt/answer journals
record the model id on every line, together with provider-reported token usage and nanodollar cost.

| Model (journal id) | Purpose | Prompts answered | Accepted by verifier | Acceptance | Recorded cost | Journal |
|---|---|---:|---:|---:|---:|---|
| `google/gemini-3.1-flash-lite-preview` | primary teacher + tool-harness arm | 1,600 full pass + 360 held-out + 444 tool answers (204 + 120 + 120) + 12 probe = **2,416** raw answers | 513 of 1,600 full-pass rows (plus held-out/tool answers) | **32.1 %** on the full pass | **$1.091** for the full pass | `data/teacher_runs/completions.jsonl`, `eval_*.jsonl`, `harness_*.jsonl`, `probe_gemini.jsonl` |
| `openai/gpt-5.4-nano-2026-03-17` | second teacher arm / domain-model stand-in | 1,600 full pass + 360 held-out + 12 probe = **1,972** raw answers | 122 of 1,600 full-pass rows | **7.6 %** on the full pass | **$0.658** for the full pass | `data/teacher_runs/completions_gptnano.jsonl`, `eval_gptnano_*.jsonl`, `probe_gptnano.jsonl` |

Full-pass spend ≈ **$1.75**. Counting every independent journal batch once (full passes, held-out eval answers,
both tool-harness versions, probes) the recorded API spend is **$2.752407**; excluding the superseded v1
tool-harness run it is **$2.561710**. Split journals and `eval_gptnano_replay.jsonl` are partitions/copies and
must not be added again. The decision rule, probe results and cost method are in
`docs/TEACHER_CAMPAIGN.md`; the cost fields come from `kaggle_benchmarks.usage.Usage` on each chat.

Held-out eval answers were generated for **360 prompts per model** (`data/frontier_eval_export/`, 30 per
task×split, 12 non-empty files): Gemini `eval_all.jsonl` ($0.184296), GPT-nano `eval_gptnano_all.jsonl` →
`eval_gptnano_replay.jsonl` ($0.131081). The tool arm is a further 120 Gemini answers with 242 `check_answer`
calls ($0.202697, `harness_eval_completions2.jsonl`).

**Models named but never successfully used for training data:** `anthropic/claude-sonnet-5-5@default` (the old
notebook default; 5 calls 400'd, now retired and deferred to a later round), `deepseek-r1-0528`, `qwen3-next-80b`,
`gpt-oss-120b`, `granite-4.0-h-small` (probe failures/no usable output), and the remaining Kaggle roster entries.
`google/gemini-3.8-flash` *was* called in the big-model campaign: 200 attempted, 33 accepted and ingested (see
`docs/TEACHER_CAMPAIGN.md`), so it is not in this list. No dedicated thermodynamics LLM was ever called: the
"other model" arm is GPT-5.4-nano's saved answers used as a stand-in, because no thermo-model id/endpoint was supplied.

### 2. Where each model's output went

```
1,600 exported ground-truth prompts (data/frontier_export, 400/task; no answers)
  -> raw teacher answers to those training prompts: 3,668 rows
       2,056 gemini  /  1,612 gpt-nano
     (held-out eval answers are a separate 360 prompts per model, scored as comparison arms)
       2,436 harness=none, 654 agents+skills, 444 tools, 134 skills
  -> verifier gate: 965 rejections -> 342 deduplicated failure-memory notes
                    635 accepts (data/frontier/*, reward=1.0)
  -> 8,000-token prompt gate (max_length 8,192): 635 accepts used in BOTH data/sft_all and data/grpo_all
      (legacy 4,000-token gate kept 460 accepts in the old data/sft and data/grpo)
```

| Artifact (training-prompt answers only) | Gemini | GPT-nano | Path |
|---|---:|---:|---|
| Raw answers | 2,056 | 1,612 | `data/teacher_runs/completions*.jsonl`, `harness_*.jsonl`, `eval_*.jsonl`, `probe_*.jsonl` |
| Accepted by verifier | 513 | 122 | `data/frontier/{repair,ledger,design,audit}_train.jsonl` (audit 0) |
| Actually in the legacy 4k-gated SFT/GRPO | 371 | 89 | `data/sft/*_train.jsonl`, `data/grpo/*_train.jsonl` (`source=frontier`) — all 635 are in the merged 8k-gated `data/sft_all` / `data/grpo_all` |
| Failure memory | 177 notes | 141 notes | `data/memory/frontier_notes.jsonl`, `data/memory/failure_reasons.json` |
| Frontier zero-shot eval arm | held-out answers | — | `reports/eval_frontier_zero.json`, `paper/assets/model_results.json` |
| Frontier + our tools arm | 120 tool answers | — | `reports/eval_frontier_tools.json`, `reports/tool_usage.json` |
| Domain-model arm | — | held-out answers | `reports/eval_other_model.json` |
| Paper examples / trace figure | 6 frontier + 3 harness examples | none promoted | `paper/assets/examples.json`, `paper/figures/svg/fig_trace_walkthrough.svg` |

All 635 accepted rows survive the 8,000-token prompt gate (max length 8,192) and are in training in the merged
corpora; none are dropped. (Under the superseded 4,000-token gate, 175 were dropped and the headline count of AI
traces used in the legacy `data/sft` / `data/grpo` corpora was **460**.) The 3,668 journal rows are raw captured
answers, not training rows; the headline count of AI traces actually used for training in the merged corpora is
**635**.

#### 2.1 Tool-using CoT data generation (current objective)

The zero-shot journals above are **legacy training data**. The current objective is to teach the 4B student *tool
use*, not just final answers. Frontier models run inside a **LangGraph harness** (`thermo_agent/teacher/agent_graph.py`,
`build_context -> propose -> {call_tools -> reflect -> propose}* -> finalize -> END`) with **real physics tools**
from `thermo_agent/teacher/toolkit.py`: `first_law_check`, `second_law_check`, `validity_check`, `solve_plant` and
`check_answer`. These build the plant from the model's own candidate, run the sequential-modular solver and return
numbers with units, so the feedback the model sees is solver output, not keyword matching. The context node injects
**AGENTS.md + component/task SKILL.md** — the same skills the student trains on — and reloads **fresh failure memory**
from `data/memory/*.jsonl` on every row. The model can call tools over several rounds, read numeric feedback and
revise before finalizing.

When the verifiers accept a tool-using trace, the **full trajectory** (reasoning, tool calls, numeric tool results,
revisions) is persisted as the SFT completion with `source=frontier_tools`, so the student imitates the whole
tool-using behaviour. Ingest is additive and provenance is recorded per trace in
`data/provenance/trace_ledger.jsonl`.

Current model choice for generating these traces: `gpt-6-luna` then `claude-haiku-5-5`, reasoning ON
(`reasoning="low"`), run as the hosted task `thermo-teacher-tools-s00` via
`scripts/run_teacher_hosted.ps1 -m <slug>`; `gemini-3.5-flash-lite` is excluded and must not consume quota.
The expensive tier follows only after the budget tier reports accepted reasoning traces.
**REMINDER: revisit `claude-sonnet-5-5` later** (deferred as too expensive — needs Reza's explicit go-ahead). All
arms are scored on the held-out `data/heldout` set (fresh seed 987654, excluded from every corpus), with each
checkpoint in its own path (`runs/sft`, `runs/grpo_k4`, `runs/grpo_k8`, `runs/grpo_k16`).

### 3. Chain-of-thought and harness counts

**Synthetic (our own) reasoning traces** — every row carries a task-tagged reasoning block
(`<diagnosis>`,`<edits>` repair/ledger; `<analysis>` audit; `<design>` design):

| Source | repair | ledger | audit | design | total |
|---|---:|---:|---:|---:|---:|
| `data/processed` | 11,810 | 8,999 | 2,916 | 3,693 | 27,418 |
| `data/processed_arch` | 2,063 | 2,115 | 1,366 | — | 5,544 |
| **Unique synthetic base traces** | **13,873** | **11,114** | **4,282** | **3,693** | **32,962** |

Literal `<analysis>` exists only in audit rows: 2,916 + 1,366 = **4,282** unique synthetic audit traces.

Corpus rows with synthetic reasoning: **122,564 SFT rows** and **27,754 GRPO rows**. Repair train shows the
20-paraphrase expansion exactly: 4,990 base rows × 20 = **99,800** rows.

**AI-frontier traces captured in harness:**

| Stage | Count | Where |
|---|---:|---|
| Raw answers with a harness/inference-text block | 788 (654 agents+skills + 134 skills) | teacher journals |
| Raw answers with tools exposed | 444 | `harness_completions.jsonl`, `harness_eval_completions*.jsonl` |
| Accepted teacher rows with a harness block | 108 of 635 | `data/frontier/*` |
| Teacher rows in SFT/GRPO with `harness=agents+skills` | **80 of 460** | `data/sft`, `data/grpo` |

Do not confuse this with harness text injected into synthetic rows: 22,535 SFT rows and 6,442 GRPO rows have
`harness != none`, but almost all of those are ground-truth traces with AGENTS.md/skills text in the prompt.
The effective harness share is 18.3 % (SFT) / 22.8 % (GRPO) after paraphrase expansion and the token gate, not
the nominal 35 % applied per base row.

**Failure memory:** 965 graded rejections were deduplicated into **342 stored notes**
(`frontier_notes.jsonl` 318 + `adversarial_repair.jsonl` 24); by task design 178, repair 68, ledger 52, audit 44.

### 4. Measured effectiveness

- Full teacher pass: Gemini accepted 513/1,600 at $1.091 (~$0.00213/accepted row); GPT-nano 122/1,600 at
  $0.658 (~$0.00539/accepted row). The cheaper model is ~2.5× more expensive per accepted row.
- Frontier zero-shot eval (30/split): repair `test_id` **0.633**, `ood_fault` 0.467, `ood_topology` 0.433;
  design valid 0.167/0.2/0.067/0.1; audit rank 0.086/0.033; ledger all_correct 0/0/0.067.
- Frontier + our tools (replay, `test_id`): repair **0.667**, design valid **0.433**, audit 0.1, ledger
  **0.767**. Tool usage: 120 answer records, 242 `check_answer` calls, **$0.2027**, answer rate 1.0;
  calls/answer: repair 1.5, ledger 2.1, audit 1.73, design 2.73.
- Other model (GPT-nano stand-in): weaker than Gemini across repair/design/audit/ledger.

### 5. Project status

#### Done
- Synthetic ground truth: 32,962 unique verifier-graded traces across four tasks and structural OOD splits;
  architecture catalog; 20 paraphrases per repair base row under an 8,000-token gate (`max_length` 8,192) in the
  merged corpora (the legacy 4,000-token gate applied only to `data/sft` / `data/grpo`).
- Verifier-gated teacher pass: 1,600 prompts × 2 models, 3,668 raw answers, 635 accepted, 342 memory notes,
  635 accepted traces in both merged training corpora (`data/sft_all` / `data/grpo_all`), all provenance preserved.
- Corpora: SFT **123,024** rows / 109,659 train; GRPO **28,214** rows / 14,849 train.
- Paper: 45-page draft, no overfull boxes, `verify_paper.py --no-build` clean, 242 `\val` keys, provenance
  for every generated number, static SVG figures preserved.
- Measured comparison arms: frontier zero-shot, frontier+tools, and domain-model stand-in (28 rows in the
  manifest); tool-usage accounting; deterministic paper examples and regenerable trace figure.
- Kaggle infrastructure: code/corpus/prompt/teacher-progress datasets, chunked cross-entropy that fixes the
  248k-vocab T4 OOM, smoke run that completed 20 training steps, cumulative logging/status, SOCKS proxy
  workflow, prebuilt fused-kernel wheel build, token-free cross-session resume relay.

#### In progress
- Smoke v18 (eval-only memory fix) queued; fused-kernel wheel build running on Kaggle.
- Re-upload of the code bundle with resume/wheel support and the kernel metadata that attaches the new datasets.

#### Not done
- **No SFT or GRPO weights exist yet.** The 4B model was never trained past the 20-step smoke; the full SFT
  epoch, merge and four GRPO stages are the remaining GPU work.
- `ours_base`, `ours_sft`, `ours_harness`, `ours_grpo` eval arms have not been run; their manifest rows are
  missing. The eval kernel and the `ours_sft`/`ours_grpo` specs are now prepared but unrun.
- GRPO-vs-SFT reward deltas, trained-policy zero-variance groups, memory ablation, GPU-hour totals,
  cost-quality frontier and significance tests are still pending (see `paper/Q1_CHECKLIST.md`).
- Three external figures (`fig_training`, `fig_corpus`, `fig_audit_first_vs_second_law`) still show pilot
  samples/split bars and must be redrawn from current assets before camera-ready.
- No thermo-specific external LLM was evaluated; no git push (invalid `gh` token).

### 6. Caveats found during the audit

- `reports/frontier_ingest.json` is cited in `docs/TEACHER_CAMPAIGN.md` but was never committed; cost is
  recoverable from the journals' `usage` fields and the acceptance counts from the id suffixes.
- Accepted rows in `data/frontier` have no `model` field: the model is encoded in the id suffix
  (`-t-google-gemini-...` / `-t-openai-gpt-...`).
- `data/memory/failure_reasons.json` grades 965 rejections, while `data/memory/frontier_notes.jsonl` stores 318
  deduplicated frontier notes (342 including adversarial); they are different stages, not a contradiction.
- `eval_frontier_tools.json` and `tool_usage.json` were computed from `harness_eval_completions2.jsonl` (242
  tool calls). The staged replay dataset previously shipped the superseded v1 (297 calls, 69 exhausted); it now
  ships v2 as well and `kaggle/eval_runner.py` points at v2.
- `paper/figures/figure_manifest.json` does not list `model_results.json` as a data source for
  `fig_trace_walkthrough`, although `figures_from_examples.py` consumes it.
- `docs/STATUS.md` still describes the unexecuted notebook path (Claude default) and some old counts; this report
  and `docs/TEACHER_CAMPAIGN.md` carry the measured values.

### 7. Additions after the original audit

- Synthetic traces grew from 32,962 to **33,156** unique base traces with the new expert-scale cycle
  (`data/processed_big`, 194 rows), and the per-trace ledger now covers **37,945** traces
  (`data/provenance/trace_ledger.jsonl`). The ledger auto-discovers new journals; big-model traces will be
  added without deleting the Gemini/GPT-nano records.
- The merged training corpora (`data/sft_all` / `data/grpo_all`) keep the old rows and add 1,140 / 57
  big-cycle training rows; the 8,000-token gate audited at max row 6,405 with zero drops.
- Kaggle datasets now attached to the train/smoke/eval kernels: `thermo-agent-code`, `thermo-agent-corpus`
  (with `sft_all.zip` / `grpo_all.zip`), `thermo-teacher-prompts`, `thermo-agent-resume`,
  `thermo-agent-eval-replays`; fused wheels come from kernel `thermo-kernel-wheels` via `kernel_sources`.
- Hosted-task campaign round 2 (`thermo-teacher-big-s00` v3, prompt dataset `thermo-teacher-prompts-big`):
  **`gpt-6-luna` then `claude-haiku-5-5`**, reasoning ON (`reasoning="low"`, 1,536 output tokens) on 974
  previously-unaccepted prompts, run as hosted task `thermo-teacher-tools-s00` via
  `scripts/run_teacher_hosted.ps1 -m <slug>`; `gemini-3.5-flash-lite` is excluded and must not consume quota.
  Additive to the old journals. **REMINDER:** revisit `claude-sonnet-5-5` later — the user deferred it as too
  expensive and it needs Reza's explicit go-ahead. No accepted rows from those models yet.

### 8. Reproduction commands

```powershell
# Corpus provenance and counts (authoritative)
.venv/Scripts/python.exe -c "import json;from collections import Counter;rows=[json.loads(l) for l in open('data/sft/sft_report.json')];print(rows)"

# Count frontier rows used in training
.venv/Scripts/python.exe -c "import json;from collections import Counter;c=Counter();[c.update({(json.loads(l)['task'],json.loads(l)['source']):1}) for f in __import__('glob').glob('data/sft/*_train.jsonl') for l in open(f,encoding='utf-8')];print(c)"

# Teacher cost and acceptance
.venv/Scripts/python.exe -c "import json,glob;from collections import Counter;a=Counter();[a.update({(r.get('model'),r.get('error') is None):1}) for f in glob.glob('data/teacher_runs/completions*.jsonl') for l in open(f,encoding='utf-8') for r in [json.loads(l)]];print(a)"
```

## Hosted teacher prompt audit — 2026-11-19

*(merged from `docs/TEACHER_CAMPAIGN.md` on 2026-11-20; content kept, headings demoted one level)*

Scope: the dataset actually read by hosted task `thermo-teacher-tools-s00`
(`kaggle/teacher_harness_task_s00.py`, `SPEC["dataset"]=rezanadimi/thermo-teacher-prompts-big`,
`SPEC["shard"]=prompts_s00.jsonl`, lines 23–34).

**Path note (verified):** there is no local `data/kaggle_datasets/thermo-teacher-prompts-big/`.
That string is the *Kaggle dataset id*. The local staging dir is
`data/kaggle_datasets/teacher_big_prompts/` (`scripts/make_hosted_prompts.py:28–30`,
`kaggle/teacher_harness_task_s00.py:201`). All shards were produced by
`scripts/make_hosted_prompts.py`, per `data/kaggle_datasets/teacher_big_prompts/manifest.json:2`
(`"generator": "scripts/make_hosted_prompts.py"`). The older compact generator
`scripts/make_teacher_big_shards.py` (SHARD_SIZE=200, `system`/`user` strings only) is **not** what
is on disk now.

---

### 1. Shard shape and inputs-only property

| Item | Value | Status |
|---|---|---|
| Shards | 5 (`prompts_s00.jsonl` … `prompts_s04.jsonl`) | verified |
| Rows / shard | 320, 320, 320, 320, 320 | verified |
| Total rows | 1,600 (400 repair / 400 audit / 400 ledger / 400 design) | verified |
| Message roles | system=1600, user=1600 (one each per row) | verified |
| Duplicate ids | 0 (within and across shards) | verified |

Observed keys (counts = rows carrying key):
`id` 1600, `split` 1600, `task` 1600, `prompt` 1600, `site` 1200, `topology` 1200,
`plant_broken` 400, `plant` 400, `ledger` 400. Per-task keysets match `FIELD_ALLOWLIST`
(`scripts/make_hosted_prompts.py:39–44`) exactly.

Forbidden/answer fields present in shards:
`completion`, `plant_original`, `min_edits`, `deducible`, `detected`, `ed_json`, `fuel_w`,
`net_w`, `best_gain_w`, `n_candidates`, `ref_spec`, `ref_fuel_w`, `n_valid`, `ref_evals`,
`fault_type`, `fault_types`, `validation`, `trajectory`, `answer`, `sft_completion`,
`verdict`, `graded`, `usage`, `model` — **0 occurrences (verified)**.

Independent answer-*value* leak check (replicating `make_hosted_prompts.py:121–149` against
`data/frontier_export`): **0 hits across all 1,600 rows (verified)**. Manifest also records
`leak_check.result = PASS`, `checked_rows = 1600`
(`data/kaggle_datasets/teacher_big_prompts/manifest.json:94–96`).

Conclusion: **every row is inputs-only — verified.**

### 2. Shard ids vs `data/frontier_export`

| Measure | Count | Status |
|---|---|---|
| `frontier_export` unique ids (4 × `*_train.jsonl`) | 1,600 | verified |
| Shard unique ids | 1,600 | verified |
| Missing (in export, not in shards) | 0 | verified |
| Extra (in shards, not in export) | 0 | verified |
| Duplicates | 0 | verified |
| Exact set equality | **True** | verified |

Source hashes in `manifest.json:136–153` match the live export files (all 4 MATCH, verified).
Conclusion: **exact 1:1 id correspondence — verified.**

### 3. Does the hosted task read heldout or answer fields?

**`data/heldout`: NOT read — verified.** A grep of `kaggle/teacher_harness_task_s00.py` finds no
`heldout` reference; the only data-root references are:

- `_find_shard()` — `kaggle/teacher_harness_task_s00.py:196–212`: searches `/kaggle/input`,
  bundle root, `.`, then `root/data/kaggle_datasets/teacher_big_prompts` (line 201).
- `_find_export_dir()` — lines 215–226: searches for `repair_train.jsonl` + all four
  `*_train.jsonl` under `/kaggle/input`/root, else `root/data/frontier_export` (lines 223–225).
- `_load_rows()` — lines 278–290: **prefers** `load_exports` when such a dir is found, and only
  falls back to the shard (`_find_shard`, line 288) when no export dir / no rows.

On Kaggle, the two attached datasets are `thermo-teacher-prompts-big` and `thermo-agent-code`;
`kaggle_bundle.zip` contains `data/memory/*` but **no `data/frontier_export` and no `heldout`**
(verified by listing the 178 zip entries). Therefore `_find_export_dir()` returns `None` on
Kaggle and the task reads `prompts_s00.jsonl` via `_find_shard()` → `shard` source path.

**Answer fields:** the shard rows carry none (§1). The task does name answer-derived columns in
`GRADABLE_KEYS` (lines 54–60: `plant_original`, `ed_json`, `fuel_w`, `net_w`) but only to test
gradability in `_row_is_gradable()` (lines 337–342); since shards lack them, rows get
`DEFERRED_VERDICT` and local `transfer.ingest` re-grades by id (lines 345–368, 475–484).
No ground-truth answer is ever loaded from the shard.

**Caveat (verified, local-only):** when a `frontier_export` dir *is* reachable (local dev:
`data/frontier_export` exists), `_load_rows()` lines 280–287 returns **full export rows** via
`transfer.load_exports` (`thermo_agent/teacher/transfer.py:99–104`, which copies every column —
those rows *do* contain `plant_original`, `ed_json`, `fuel_w`, `ref_spec`, `fault_types`, etc.).
This path is not reachable with the Kaggle bundle as configured. Also
`build_context()` parses `plant_original` if present (`thermo_agent/teacher/agent_graph.py:87`),
irrelevant for shards.

### 4. Prompt text / tool contract / bookkeeping

Scan of all 1,600 row `prompt` message contents (verified counts):

| Pattern | Occurrences |
|---|---|
| `<tool_call` | 0 |
| `tool_output` / `tool result` | 0 |
| `model_id` / `total_cost` / `nanodollar` | 0 |
| `timestamp` / `record_id` | 0 |
| `AGENTS.md` | 321 |
| `system` (as literal word) | 91 |

What is actually present:
- Each row is just a `system` instruction (task rules + answer XML-tag format) and a `user`
  message with the physical inputs (`SITE …`, `PLANT …`, `LEDGER …`).
- 321 system messages (repair 71, ledger 103, design 147 — exactly the `agents+skills`
  harness rows) embed an `# AGENTS.md: energy-plant tasks` guidance section plus component/task
  skill text. This is **guidance/task documentation inherited from the export prompt**, not tool
  bookkeeping — no model ids, costs, timestamps, record ids, or tool outputs.
- The literal `<tool_call …>` string exists in the codebase only at
  `thermo_agent/teacher/agent_graph.py:207`, inside `finalize()`, where it builds the **SFT
  completion (training target)**, not the model prompt.
- Runtime (not stored in the shard): `build_context()` (`agent_graph.py:82–129`) appends a compact
  guidance block (`compact_memory.py:83`), compact failure memory with model id / cost / ts /
  record id / run id stripped (`compact_memory.py:3–8, 43–58`), and a short tool contract naming
  tools via `tool_names()` (`agent_graph.py:113–120`). Tools are executed through kbench native
  tool calling (`make_respond`, `run_direct`, harness lines 463–469), so no literal tool contract
  is needed in the row prompt — consistent with the 0 `<tool_call` count.

Conclusion: **no literal tool contract and no tool-output/memory bookkeeping in the prompt text —
verified.**

### 5. Determinism and resumability

**Generation deterministic — verified.** `make_hosted_prompts.py:32–35, 168–188`: fixed
`TASK_ORDER`, `SEED=20261119`, `random.Random(SEED).shuffle`, round-robin into 5 shards. An
in-memory rebuild from the live export reproduced all 1,600 rows in **identical order and content**
for every shard (byte-level diff is only Windows CRLF translation by `Path.write_text`; parsed
rows match 100%). Source hashes also match (§2).

**Resumable at run time — verified with caveats.**
- `JOURNAL = /kaggle/working/harness_completions.jsonl` (lines 36–38); a `done` set of ids is read
  from it at lines 427–433; rows whose id is in `done` are skipped (lines 437–441).
- One JSON line is appended and `flush()`ed per row (lines 501–502), so a dead worker loses at
  most one answer. Malformed/partial journal lines are ignored by the `try/except` at lines
  428–433. Rerun after a partial journal resumes at the first un-journaled id in the fixed shard
  order; nothing is recomputed for done ids.
- Stops (deadline / cost cap / 5 errors with 0 answers) write `partial`/`paused` manifest status
  (lines 509–517) and are meant to be resumed by rerun.

Caveats:
- `LIMIT=40` is applied **before** the done-filter (lines 416–419). With the first 40 ids already
  journaled, a rerun at the default limit skips all 40 and answers 0; progress past the pilot
  requires raising `HARNESS_LIMIT`. (verified by code path; not executed end-to-end — **not
  verified** empirically.)
- Shard selection order flips if a `frontier_export` dir is attached (`load_exports`, lines
  280–287), which returns full answer-carrying rows; not applicable to the configured Kaggle
  inputs (§3).
- End-to-end hosted resume behavior was **not verified** (no Kaggle run inspected).

### 6. Training/eval contamination

| Check | Count | Status |
|---|---|---|
| Unique ids in `data/heldout` (20 `.jsonl` files) | 542 | verified |
| Shard ∩ heldout ids | **0** | verified |
| `frontier_export` train ∩ heldout ids | **0** | verified |
| `data/frontier_eval_export` unique ids | 360 | verified |
| Shard ∩ `frontier_eval_export` ids | **0** | verified |

Conclusion: **no contamination from these shards — verified.** The shards are exactly the
`frontier_export` *train* ids (intended teacher training set) and are disjoint from the held-out
set and from the separate `make_eval_export.py` export (`scripts/make_eval_export.py:1–8, 20–41`,
which strips `completion` and, per its default, sources `data/heldout`; the on-disk manifest
currently lists `data/processed`, `data/processed_arch`, limit 30).

### Summary table

| Question | Result | Status |
|---|---|---|
| Shards/rows/inputs-only | 5 shards × 320 = 1600; no answer fields or values | verified |
| Id match to frontier_export | exact 1:1 (0 missing/extra/dup) | verified |
| Reads heldout / answer fields | never reads heldout; no answers in shards (local export path caveat) | verified |
| Literal tool contract / bookkeeping in prompt | none; AGENTS.md guidance present, bookkeeping stripped | verified |
| Deterministic + resumable | deterministic; journal-resumable; LIMIT-before-done caveat | verified (code/rebuild); hosted rerun not verified |
| Contamination | 0 overlap with heldout or eval export | verified |

## Hosted run v6 assessment — `thermo-teacher-tools-s00` version 6

*(merged from `docs/TEACHER_CAMPAIGN.md` on 2026-11-20; content kept, headings demoted one level)*

Assessed on 2026-11-19 from the downloaded journal, not from the log tail. Every number below is reproducible with the
commands in §7.

### 1. Run identity

| Field | Value |
|---|---|
| Kaggle task | `rezanadimi/thermo-teacher-tools-s00` version 6 |
| Model | `google/gemini-3.8-flash` (single model in the version) |
| Run id | `7179338` |
| Task created / model started / ended | 2026-11-19 11:29:34 / 11:30:36 / 13:09:20 |
| Journal | `data/teacher_runs/kaggle_hosted/thermo-teacher-tools-s00/6/gemini-3.8-flash/7179338/harness_completions.jsonl` (935,803 bytes) |
| Shard | `prompts_s00.jsonl`, 320 rows, harness `"tools"`, 5 tools offered, `notes_used` = 3 |
| Attempts | 40 |
| Cost | **$0.842202** (1,210,642 input tokens, 66,217 output tokens) |

The version ran with `SPEC["limit"] = 0`, 6 rounds and no cost/deadline cap (D1-D9), which is why it attempted the whole
40-row shard instead of stopping at 40 accepted.

### 2. Journal counts (raw)

| Bucket | Count |
|---|---|
| rows in journal | 40 |
| API errors (`RateLimitError` 429 "model is currently experiencing heavy load") | 5 |
| rows with a non-empty `sft_completion` | 35 |
| rows whose `sft_completion` contains `<tool_call>` **and** `<tool_result>` | 35 |
| turns per non-error row | 5 |
| rows with `completion` (final answer) non-empty | 34 |

So the model did the multi-turn loop on 35 of 35 non-error attempts. The failure mode of this run is provider load, not
the payload: the 5 failures are all 429s, no 400/format rejection.

### 3. Ingested traces

First pass (only `data/processed` as the export source) accepted **22**. The other 12 non-error rows were counted as
`skipped_unknown_or_empty` because their ids are not in `data/processed/*_train.jsonl`. That was an ingest defect, not a
model failure (§6).

After fixing id resolution, the same journal yields:

| Task | Accepted traces |
|---|---|
| repair | 9 |
| ledger | 10 |
| audit | 6 |
| design | 7 |
| **total** | **32** |

- 32/32 carry a real tool turn (`source_detail = "frontier_tools"`, i.e. `<tool_call>` + `<tool_result>` in the saved
  completion), 32 unique `record_id`s, no duplicate rows.
- 1 rejection: design, `grounding: only 56% of the numbers in <rationale> are in the plant`.
- 2 rows unusable: 1 empty completion (`led-arch-0-02876`), 1 rejected (above). 5 rows were 429s.
- **Cost per accepted tool trace: $0.842202 / 32 = $0.0263.**
- **Yield: 32 accepted / 40 attempts = 80%**; 32/35 non-error = 91%.

This is the first non-zero frontier tool-trace yield in the project. `docs/DECISIONS.md` said "zero
verifier-accepted frontier traces containing a real tool call"; that statement was true before this run and is now
superseded by this document.

### 4. Gate measurement with the real Qwen3.5 tokenizer

Measured on the saved row (prompt messages joined + completion + 96 reserved tokens), tokenizer
`C:\Users\rezax\.cache\thermo\qwen3.5-4b\tokenizer.json`:

| | raw | after removing `call_id` blobs |
|---|---|---|
| completion tokens p50 | 5,698 | 2,019 |
| completion tokens p95 | 14,643 | 4,529 |
| completion tokens max | 15,840 | 4,689 |
| total tokens max | **17,212** | **7,280** |
| rows over the 8,192 gate | **11 / 32** | **0 / 32** |

The 11 over-gate rows are over *because of* the `ToolInvocationResult(..., call_id='call_...__mpthought__<base64>')`
wrapper, which is a provider-side invocation id and carries no training signal. Removing it (keeping tool name +
`output=` JSON) puts every row under the gate with 912 tokens of headroom to spare at the max. That is the sanitizer
task from the handoff, defect #2; the accepted fix is a rewrite of the completion text only, ids and the final answer
unchanged, output written to a new dir (`data/frontier_tools_clean`) so `data/frontier_tools` stays as received.

Note the prompt stored in these rows is the **plain** 2-message prompt (493-3,010 tokens by task), not the harness
prompt. The harness text is applied when the row is rebuilt for training, so the trace is reusable under the compact
harness; the model's assistant turns do not embed the harness.

### 5. What this run proves and does not prove

Proves:
- The hosted payload is accepted by `gemini-3.8-flash` end to end: tools are called, results are returned, the model
  self-corrects over 5 turns, and the project's own validators accept 32 of those answers.
- The verifier is not rubber-stamping: 1 of 33 answered rows was rejected for grounding.

Does not prove:
- Nothing about training. No checkpoint exists, no SFT/GRPO step has run, no eval has been scored.
- Nothing about the compact harness. This run used the 17.2k-char harness prompt; the traces are model output only, so
  they survive the compaction, but a later run must confirm the model behaves the same under the compact text.
- Nothing about scale. 32 traces against the E1 plan of ~4,800 deterministic tool rows and a frontier share at
  R_tool; this is a pilot yield and a unit cost, not a corpus.

### 6. Defects found while ingesting (all fixed in this pass)

1. **`load_exports` did not tag the task.** `data/processed/*_train.jsonl` rows have no `task` column; every row fell
   back to `"repair"` and the first design row crashed with `KeyError: plant_broken`. Fixed: the file name tags the row.
2. **Ids were resolved from one corpus only.** The trace shard is built from `data/processed` **plus**
   `data/processed_arch`; 7 traces have `*-arch-*` ids. 5 more are `#vN` paraphrase ids that only exist in the already
   ingested `data/frontier_prov`. All 12 were silently dropped as "unknown". Fixed: `ingest --extra-export-dir`
   (repeatable), keyed by `record_id` when present, primary dir wins. Two tests added in `tests/test_transfer.py`.
3. **The provider wrapper leaks into the training text** (see §4). Fix in progress: `scripts/clean_frontier_traces.py`
   writing `data/frontier_tools_clean`.
4. **Cosmetic, known:** the 5 `#vN` rows resolved through `data/frontier_prov` inherited that corpus's id suffix, e.g.
   `rep-42-00045#v17-t-google-gemini-3-1-flash-lite-...`. The authoritative producer is the row's `model` field
   (`google-gemini-3-8-flash`); `record_id` is correct. Left as-is rather than rewriting accepted rows.

### 7. Reproduce

```powershell
# 1. status + journal download (SOCKS proxy as always)
$env:HTTP_PROXY='socks5h://localhost:10808'; $env:HTTPS_PROXY='socks5h://localhost:10808'
.venv/Scripts/kaggle.exe benchmarks tasks status thermo-teacher-tools-s00
.venv/Scripts/kaggle.exe benchmarks tasks download thermo-teacher-tools-s00 -m gemini-3.8-flash -o data/teacher_runs/kaggle_hosted

# 2. ingest (idempotent: accepted record_ids are skipped)
.venv/Scripts/python.exe -m thermo_agent.teacher.transfer ingest data/processed `
  "data/teacher_runs/kaggle_hosted/thermo-teacher-tools-s00/6/gemini-3.8-flash/7179338/harness_completions.jsonl" `
  data/frontier_tools --model google/gemini-3.8-flash --run-id hosted-v6-7179338 --memory-dir data/memory `
  --extra-export-dir data/processed_arch --extra-export-dir data/frontier_prov

# 3. counts and gate (real tokenizer)
.venv/Scripts/python.exe scripts/clean_frontier_traces.py --src data/frontier_tools --out data/frontier_tools_clean
.venv/Scripts/python.exe -m thermo_agent.train.token_audit data/frontier_tools_clean
```

### 8. Next gate

Per `docs/IMPLEMENTATION_PLAN.md` Phase 0 and J4, this run authorises nothing further by itself. Order stays: compact
harness (0.1) -> gate re-measure (0.2) -> clean traces (0.3) -> 40-row pilot with `gemini-3.5-flash-lite`, then
`claude-haiku-5-5`, then `gpt-6-luna`. A model that rejects the payload is dropped, not retried.

## Token efficiency and RAG-memory serving — 2026-11-19

*(merged from `docs/TEACHER_CAMPAIGN.md` on 2026-11-20; content kept, headings demoted one level)*

Scope: **model-visible text only**. On-disk ledgers, manifests, journals and raw
verifier logs stay intact for provenance and papers. This note records the prompt
paths that were mapped, the bookkeeping surfaces found, the fix, and the measured
token reduction. No training/eval files were edited.

### 1. Prompt paths mapped

| # | Serving path | Builder before | Model saw | Bookkeeping surface found |
|---|---|---|---|---|
| 1 | Kaggle hosted tracegen (`kaggle/teacher_harness*.py` → `teacher/agent_graph.py::build_context` → `teacher/kbench_adapter.py::make_respond`) | full `AGENTS.md` + *all* component skills (`agent/harness.py::augment`), then D10 guidance, compact memory, and a ~180-token tool-name paragraph | 3,028–3,882 tokens of system text on one real row per task | row fields available but not serialised; harness text not shared with training/eval |
| 2 | Compact harness (`agent/tool_prompt.py`) — training `*_h` corpora (`scripts/build_tool_corpus.py`, `scripts/mix_tool_corpora.py`), eval H (`eval/model_eval.py::ToolLoop`), GRPO | already the single shared builder: compact AGENTS.md + the *one* selected skill + the five-tool contract | correct; 661–1,385 tokens per real row | none |
| 3 | Eval Z arm / harness-text arms (`eval/model_eval.py::_wrap`) | `build_compact_harness_messages(..., include_contract=False)` | task + artefact + compact AGENTS/skill, no tool contract | none |
| 4 | RAG memory (`teacher/memory.py::retrieve` → `teacher/compact_memory.py::compact_notes`) | retriever filters task, excludes the current `record_id`, reserves a failure slot; renderer emits one `OK`/`FAIL` recap line per note | fault class + reusable fix pattern only | **dropped:** `id`, `record_id`, `source`, `extra`, `architecture`, `symptoms`, model slugs, timestamps, `ts=`/`cost=`/`tokens=`/`latency=` |
| 5 | `agent/harness.py::augment_with_memory` (shared agent-layer memory builder) | `memory.augment_with_memory` → verbose `format_notes` | verbose `Lesson:/Fix:` block; note text was passed through verbatim | could carry bookkeeping from `lesson`/`fix`/`symptoms` |
| 6 | `teacher/sample.py::build_messages`, `synth/combine.py::_augment` (corpus building, **not owned**) | compact/verbose harness + `memory.augment_with_memory` | see §5 | verbose `format_notes` leak risk (reported, not edited) |

Bookkeeping on the **rows** themselves — `id`, `run_id`, `manifest`,
`export_manifest`, `journal`, `source`, `model`, `ts`, `record_id` — is never read
by any of the builders: they render only `row["prompt"]` (task statement +
artefact) plus the harness and the retrieved recaps. This is now pinned by a
sentinel-injection test on all four tasks and all three serving paths.

### 2. Changes made (files owned)

- **`thermo_agent/teacher/agent_graph.py`** — `build_context` now appends the
  harness through the shared `build_harness_tool_messages` (same function training,
  eval and GRPO use), passing the enforced round count as `max_turns`. The verbose
  full AGENTS.md / component-skill block is gone. The trailing ~180-token paragraph
  that re-listed every tool name was replaced by one short reminder that keeps the
  enforced round count in the prompt (D4). Retrieved memory is still rendered once
  by `compact_notes`, and `notes_used` still matches exactly the notes served.
- **`thermo_agent/agent/harness.py`** — `augment_with_memory` now retrieves once
  and serves `compact_notes` recaps instead of the verbose `format_notes` ledger;
  falls back to the task text as the retrieval query when there is no plant.
- **`tests/test_rag_memory_serving.py`** (new, 11 tests) — sentinel bookkeeping
  injected into rows and notes never reaches the model; OK/FAIL success/failure
  recaps appear when relevant; the hosted harness is byte-identical to the shared
  builder; the agent memory layer serves recaps, not the verbose ledger.
- **`tests/test_tool_prompt_token_audit.py`** (new, 1 test) — reproducible p50/max
  before/after measurement with the cached Qwen3.5 tokenizer and the 2,500-token gate.
- **`docs/TEACHER_CAMPAIGN.md`** — this document.

Physics content, tool results in SFT completions, final-answer tags and the
`<tool_call>`/`<tool_result>` format are unchanged. The compact harness builder's
text was **not** modified, so eval prompts and `harness_config_hash` are unaffected.

### 3. Token measurements (cached real Qwen3.5 tokenizer)

#### 3a. Harness block, 8 real rows per task (32 rows)

| task | n | compact p50 / max | verbose p50 / max | saved p50 |
|---|---:|---:|---:|---:|
| repair | 8 | 661 / 661 | 3,921 / 3,921 | 3,260 |
| design | 8 | 1,041 / 1,041 | 2,894 / 2,894 | 1,853 |
| audit | 8 | 1,385 / 1,385 | 4,869 / 5,516 | 3,484 |
| ledger | 8 | 1,204 / 1,204 | 3,046 / 3,046 | 1,842 |
| **ALL** | **32** | **1,122 / 1,385** | **3,046 / 5,516** | **1,924** |

p50 reduction ≈ **63 %** (1,924 / 3,046). **Compact harness max = 1,385 ≤ 2,500**
(DECISIONS J2/J3 satisfied with 1,115 tokens of headroom).

#### 3b. Hosted tracegen system prompt (one real row per task, same rows before/after)

| task | before | after | saved |
|---|---:|---:|---:|
| repair | 3,882 | 1,067 | 2,815 |
| design | 3,028 | 1,619 | 1,409 |
| audit | 3,676 | 1,813 | 1,863 |
| ledger | 3,047 | 1,649 | 1,398 |

Across 8 rows per task after the fix: p50 **1,634**, max **1,813** tokens.
The remaining ~220 tokens are the D10 method guidance (kept because a non-owned
test pins `"FIRST inventory"`), not bookkeeping.

### 4. Tests run (exact results)

- Changed-path suite (memory / agent / tool-prompt / RAG / harness / hosted /
  eval-arm guards / transfer / validate, including tests I do not own):
  **163 passed in 46.51 s** (re-run; earlier identical run: 58.22 s).
- New files only: `test_rag_memory_serving.py` **11 passed in 4.29 s**;
  `test_tool_prompt_token_audit.py` **1 passed in 10.47 s**.
- `scripts/audit_prompt_separation.py`: **24 arms, 0 mismatches** (Z/H separation intact).

No test file owned by the eval-arm work was edited.

### 5. Conflicts / gaps outside this agent's ownership (reported, not edited)

- **Eval-arm work** (`thermo_agent/eval/model_eval.py`, `kaggle/eval_runner.py`,
  `tests/test_eval_arm_prompts.py`, `tests/test_eval_runner_arms.py`,
  `tests/test_eval_f4_arms.py`, …) is concurrently modified in the working tree.
  This task made **no** edits there. The change in `agent_graph.py` affects only the
  hosted tracegen path; all eval arms still call the unchanged shared compact builder.
  The separation audit is green (24/24), so there is no prompt-contract conflict.
- **`thermo_agent/teacher/sample.py::build_messages`** uses the compact harness but
  still appends `memory.augment_with_memory`, which renders the verbose `format_notes`
  block verbatim. If a mining step ever writes a model id, timestamp or `run_id=` into
  `lesson`/`fix`, that text would reach the hosted teacher. Recommend routing it through
  `teacher.compact_memory.compact_notes` (a one-line change, outside owned files).
- **`thermo_agent/synth/combine.py::_augment`** still uses `harness.augment` (verbose)
  plus verbose memory for the A-slice. DECISIONS E4 intends the A-slice to be plain,
  so this is a lower priority, but it is a second non-compact serving surface.
- `tests/test_kbench_adapter.py`, `tests/test_sample.py` and `tests/test_combine.py`
  (not owned) assert the verbose markers `"FIRST inventory"` / `"KNOWN FAILURE MODES"`.
  They were deliberately kept green: the D10 guidance block remains in `build_context`,
  and the global `memory.format_notes` was not changed. Removing the guidance would
  save another ~220 tokens per hosted prompt but requires updating those three tests
  in the eval/teaching workstream first.
