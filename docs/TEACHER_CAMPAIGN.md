# Teacher campaign runbook (tool-using CoT, budget tier first)

There are **two different routes** to the tool-using teacher campaign, and they are not
interchangeable. **Model availability decides the route:** the big slugs (`gpt-6-luna`,
`claude-haiku-5-5` and the whole expensive tier) are reachable **only** as a hosted Kaggle
Benchmarks task (`-m <slug>`). The local SDK driver sees only 8 keys and silently falls back
on a miss, so it must never be asked for a model outside that list. Route A is therefore the
route for luna / haiku / expensive; Route B is for the 8 SDK keys only. Both routes ingest
locally and share the same acceptance authority (`transfer.ingest`) and provenance destinations.

## Route A — hosted task `thermo-teacher-tools-s00` (use this for luna / haiku / expensive)

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

### Model order (fixed, budget first)

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

### Grading authority — INPUTS ONLY, graded at ingest

The hosted task sees **inputs only**: the compact prompt shard carries no plant/spec, so the task
cannot decide whether an answer is correct. Every journal row is therefore written with
`"verdict": {"accepted": null, "graded": false, "reason": "deferred_to_ingest"}` and top-level
`"graded": false`. **`transfer.ingest` is the ONLY acceptance authority.** It re-grades each
completion **by `record_id`** against the answers in `data/frontier_export`, using the local tool
suite, and appends only accepted rows to `data/frontier/`. A completion in a journal is never
evidence of acceptance.

### Journal fields

Each record in a hosted shard carries: `id` (the `record_id` used at ingest), `task`, `model`,
`harness: "tools"`, `tools`, `completion`, `sft_completion`, `verdict` (the deferred verdict
above), `graded: false`, `trajectory` (`turns`, `notes_used`, `tools_available`, `final`),
`usage` (tokens / cost / latency), `error`, `ts`, `shard`, `row_source`, `spec_sha`.

### Per-model shards, resume-by-id, append-only

The downloaded journal is appended to `data/teacher_runs/hosted_<slug>.jsonl`
(e.g. `hosted_gpt-6-luna.jsonl`, `hosted_claude-haiku-5-5.jsonl`); downloads are staged under
`data/teacher_runs/kaggle_hosted/<slug>/`. Model slugs are sanitized to `[A-Za-z0-9._-]`, and
each model has its own shard so no model can overwrite another's evidence. Before appending, the
driver reads the shard's existing `id` set and skips ids already present — **resume is by id** and
a partial/interrupted run continues cleanly. Ingest separately skips ids already accepted in
`data/frontier` (`skipped_already_accepted`), so replaying a shard never duplicates a training row.

## Route B — local SDK driver `scripts/run_teacher_campaign.ps1` (8 SDK keys only)

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

### Shard naming and append-only contract

Each model writes `data/teacher_runs/harness_<model>_<task>.jsonl`, e.g.
`harness_google-gemini-3.1-flash-lite-preview_all.jsonl`. Model/task names are sanitized
to `[A-Za-z0-9._-]`. The driver opens the shard in append mode and never truncates it, so one
model cannot overwrite another's evidence. Ingest appends accepted rows to
`data/frontier/frontier_<task>_<split>.jsonl` and dedupes on `record_id`.

### Resumability (local SDK)

- Driver level: on start it reads its own shard and skips every id that already has a usable
  completion; a quota pause or time limit leaves a partial shard that the next run continues.
- `-Resume` level: before the model loop, a one-shot Python filter writes
  `data/teacher_runs/_campaign_pending_export/` containing only export rows whose id is **not**
  already accepted in `data/frontier/`. The snapshot is built once so every model in the same
  campaign sees the same pending ids. In `-DryRun` the filter is printed, never executed.
- Ingest level: `transfer.ingest` skips ids already present in `data/frontier` and records the
  count as `skipped_already_accepted`.

### Quota guard (local SDK)

Live in the driver, not in the orchestrator: before every row `teacher.cli_runner.quota()` is
checked and the loop pauses when daily remaining drops below `--reserve` (default 0.5); it also
stops after `--max-seconds` (default 4 h). The driver catches per-row exceptions and journals
them as `error` records rather than dying, so a failed row does not abort the shard. The
orchestrator checks each native exit code and stops that model's ingest on failure; the shard is
kept intact for the next `-Resume`.

## Where provenance lands (both routes)

- **Corpus:** accepted rows land in `data/frontier/*`, with the provenance twin in
  `data/frontier_prov/*`. Every accepted row carries `source=frontier:<model>`, `model` (slug),
  `provider` (coarse label derived from the slug), and `run_id` when one is supplied. Rejected
  attempts become memory notes tagged `frontier:<model>` in `data/memory/frontier_notes.jsonl`
  and counts in `data/memory/failure_reasons.json`.
- **Living stats:** `scripts/teacher_campaign_stats.py` reads every journal in `data/teacher_runs`
  and writes `data/teacher_runs/CAMPAIGN_STATS.json` (machine-readable) and
  `docs/TEACHER_CAMPAIGN_LOG.md` (append-friendly human log, with older snapshots preserved under
  History). Rows are split into `accepted` / `rejected` / `ungraded`; a hosted completion with a
  deferred verdict counts as `ungraded`, never accepted. The acceptance rate uses only graded
  rows (`accepted / (accepted + rejected)`). Ingest grading is merged in by re-reading the
  journals after each ingest, so the stats reflect re-graded `record_id`s rather than the
  hosted task's deferred placeholders.

## Verify counts

```powershell
# Attempts per model across journals (model field is on every shard line).
.venv/Scripts/python.exe -c "import json,pathlib,collections; print(collections.Counter(json.loads(l).get('model','?') for p in pathlib.Path('data/teacher_runs').glob('*.jsonl') for l in p.open(encoding='utf-8') if l.strip()))"

# Accepted rows per model tag and source in the corpus (id ends in -t-<model>).
.venv/Scripts/python.exe -c "import json,pathlib,collections; print(collections.Counter((json.loads(l).get('source'), json.loads(l)['id'].split('-t-')[-1]) for p in pathlib.Path('data/frontier').glob('frontier_*_train.jsonl') for l in p.open(encoding='utf-8') if l.strip()))"

# Refresh the living stats/log after an ingest.
.venv/Scripts/python.exe scripts/teacher_campaign_stats.py
```
