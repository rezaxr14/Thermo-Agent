---
name: kaggle-benchmarks
description: Run evaluations on frontier models through Kaggle Benchmarks (interactive proxy or hosted tasks), with quota-aware pause/resume and result capture. Use when a project wants to call many hosted LLM/VLMs billed to a Kaggle quota, schedule server-side runs, or save and resume benchmark results.
---

# Kaggle Benchmarks: project-agnostic field guide

Measured on one Kaggle account on 2026-11-12 to 2026-11-14 (CLI `kaggle` 2.2.5, `kaggle-benchmarks` 0.6.3). Kaggle changes this surface
often: **re-run the checks in section 11 before trusting any number here.** Things not observed are in section 10, never assumed.

## 1. What it is

Kaggle Benchmarks lets you call many frontier models (Anthropic, Google, OpenAI, xAI) billed to a **Kaggle quota** instead of per-vendor keys.
Two routes, same SDK and CLI, **different credentials and different model access**:

| Route | Your code runs | Models reachable | Use for |
|---|---|---|---|
| Interactive proxy | your own session (Kaggle/Colab/local) calling the Model Proxy | a curated subset (ours: 6 vision + 2 text-only) | quick checks, probes |
| Hosted task (`tasks push` + `tasks run`) | a Kaggle worker, server-side; you can close the page | the full catalogue (37 listed, 34 read images) | real evaluations over many models |

Hosted run lifecycle:

```
tasks push NAME -f task.py -d owner/dataset   -> task source + attached datasets
tasks run  NAME -m <cli-slug>                  -> one run = (task version, model); ONE model per call
tasks status NAME                              -> queued / running / completed / failed, per model
tasks download NAME -o OUT -m <slug> -s -f     -> OUT/NAME/<version>/<model>/<run_id>/<files>
```

**The account-default trap.** `tasks push` immediately starts one run on the account's
default model, which on this account is `gemini-3.8-flash`. That run is not selected by
`LLM_DEFAULT`; the only reliable defence is a guard inside the task that returns before
the first model call when `llm.model` is not on an allow-list. The generic runner in
`notebooks/kaggle_model_runner.py` does exactly this, so its push auto-run spends nothing.
Never batch `-m a -m b` (HTTP 400); loop over models locally.

## 2. Setup and credentials

* `pip install "kaggle>=2.2.5" "kaggle-benchmarks>=0.5.0"`; pin the versions you tested.
* Interactive: `kaggle benchmarks init -y --env-file F` (or `auth -y`) writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY` (**about 1 h** expiry:
  renew by re-running it), `LLM_DEFAULT`, `LLMS_AVAILABLE`. Without `init`, `kbench.llms == {}`.
* Hosted: the platform injects the proxy variables; the key lasts about **12 h**; runs of **at least 150 min** are allowed.
* **Kaggle Secrets do NOT work inside hosted runs.** Put your own tokens in a **private Kaggle dataset** holding a file named `env` or `.env`
  (uploads can drop dotfiles), attach it to the task, read it from `/kaggle/input/...`. Never print values, never put them in task source,
  and never let it override `MODEL_PROXY*`, `KAGGLE_*`, `LLM*`.

## 3. Quota and cost

* `kaggle benchmarks quota --format json` -> Daily/Monthly `remaining`, `used`, `total`, `refillAt` (ours: Daily $10, Monthly $100). It moves in
  step with summed response costs.
* Per request: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.cost.{input,output}_tokens_cost_nanodollars`, `total_backend_latency_ms`.
  `completion_tokens` **includes hidden reasoning**, so `usage.cost` is the true spend. Treat a missing cost or unknown unit as `None`, never 0.
* Multi-turn chats re-send history (images included) every turn: input tokens grow roughly quadratically with turns. Estimate before you spend.

## 4. Calling a model

### Proxy (OpenAI-compatible)

`POST {MODEL_PROXY_URL}/openapi/chat/completions` with `Authorization: Bearer $MODEL_PROXY_API_KEY`. Inside a hosted run `MODEL_PROXY_URL`
already ends in `/openapi`. Measured working: system message, base64 `image_url` parts, multi-turn with a later image, `max_tokens` enforced
(`finish_reason: length`), 16384 output tokens, 5 MB images, **16 parallel requests**, no rate-limit headers. `generateContent` also works for Gemini.

```python
import os, requests
r = requests.post(os.environ["MODEL_PROXY_URL"].rstrip("/") + "/chat/completions"   # drop the suffix if the URL does not already end in /openapi
    , headers={"Authorization": "Bearer " + os.environ["MODEL_PROXY_API_KEY"]},
    json={"model": "<proxy-slug>", "max_tokens": 512, "temperature": 0,
          "messages": [{"role": "user", "content": [
              {"type": "text", "text": "Describe the image."},
              {"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>"}}]}]}, timeout=300)
```

Learn per-model parameters from the provider's HTTP 400 and record what you used on every request:

| Quirk (measured) | Models |
|---|---|
| rejects `temperature` (not greedy) | claude-sonnet-5 |
| not repeatable at temperature 0 | gemini-3.6-flash |
| hidden reasoning billed by default; `thinking_budget: 0` / `reasoning_effort: none` turns it off on 3.6 Flash | gemini-3-flash-preview, gemini-3.6-flash |
| `temperature` accepted only while `reasoning_effort` is unset | gpt-5.4-nano |
| some models want `max_completion_tokens` instead of `max_tokens` | learn from the 400 |
| image tokens: Gemini ~1.1k at any size; GPT/Claude scale with pixels (2048x1024: 2466 / 2753); `detail`/`media_resolution` ignored | all |
| prompt caching: no discount seen | all |

### SDK

`import kaggle_benchmarks as kbench`; `@kbench.task(name=...)` (return annotation required if it returns a value); `llm.prompt(text, image=, temperature=)`;
images via `kaggle_benchmarks.content_types.images` (`from_path/from_url/from_base64`); `chats.new(name, system_instructions=)`;
manual model: `ModelProxy(model=, api_key=, base_url=)`. The SDK disables `temperature` on the OpenAI route. If you need full control of the
transcript (agent loops), call the proxy directly.

## 5. Model roster

Two lists, do not confuse them:

* `kaggle benchmarks tasks models` -> **37** bare slugs.
* Interactive key: only `anthropic/claude-sonnet-5@default`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.5-flash-lite`,
  `gemini-3.6-flash`, `gpt-5.4-nano-2026-03-17` read images; `gpt-oss-120b`, `ibm/granite-4.0-h-small` are text-only; the rest: "model not found".
* Hosted: **34 read images**; `gpt-6-luna` answered wrongly; `gpt-oss-120b` text-only; `grok-4.5-0708` listed but 404.

Hosted slug rule: `tasks run -m` takes the bare CLI slug (`claude-haiku-5-5`); inside the run the proxy wants the provider-prefixed one
(`anthropic/claude-haiku-5-5`, read it from `llm.model`). Vision-capable hosted models (CLI slug, approx. $/Mtok in/out from one tiny request, indicative):

* Anthropic: claude-haiku-4-5-20251001 1/5, claude-haiku-5-5 0.1/0.5, claude-sonnet-4-6-default 3/15, claude-sonnet-5-default 2/10, claude-sonnet-5-5-default 2/10,
  claude-opus-4-5-20251101 5/25, claude-opus-4-6/4-7/4-8-default 5/25, claude-opus-5-default 5/25, claude-opus-5-5-default 4/20
* Google: gemini-2.5-flash 0.3/2.5, gemini-2.5-pro 1.25/10, gemini-3-flash-preview 0.5/3, gemini-3.1-flash-lite-preview 0.25/1.5, gemini-3.1-pro-preview 2/12,
  gemini-3.5-flash-lite 0.3/2.5, gemini-3.6-flash 0.75/3.75, gemini-3.8-flash 0.75/3.75 (the account default), gemini-4-argon 2/10, gemma-4-26b-a4b-it 0.15/0.6, gemma-4-31b-it 0.15/0.6
* OpenAI: gpt-5.4-2026-03-05 2.5/15, gpt-5.4-mini-2026-03-17 0.75/4.5, gpt-5.4-nano-2026-03-17 0.2/1.25, gpt-5.5-2026-04-23 5/30, gpt-5.6-luna 0.2/1.2,
  gpt-5.6-sol 4/20, gpt-5.6-terra 2/12, gpt-6-astra 10/50, gpt-6-sol 2/10, gpt-6.1-sol 2/10
* xAI: grok-4.6 2/6, grok-4.7 1.3/6

## 6. Hosted worker environment (measured)

* Python **3.12.10**, **4 CPU, no GPU, no torch**. pip, git and Hugging Face reachable; most other sites blocked.
* Attached datasets mount at `/kaggle/input/<name>` (interactive notebooks: `/kaggle/input/datasets/<owner>/<name>`). Read the mount path back; do not type ids.
* `/kaggle/working` is what `tasks download` returns. Write a small **manifest** (status, stage, error) so failures are diagnosable without logs.
* `tasks push` **also starts one run on the account's default model**: adopt it instead of scheduling again.
* `tasks run` with several `-m` returns **HTTP 400**: one model per call.
* The worker can `git clone` your repo (clone the default branch: **push first**) and `pip install`. A bare worker needs every third-party import
  your package pulls in (including package `__init__` side effects): list them and test the list against the real import chain.
* Hashes of `ast.dump` text differ between Python 3.12 (hosted) and 3.13 (some interactive notebooks): do not pin such hashes.

## 7. Task template

```python
import json, os, kaggle_benchmarks as kbench
SPEC = json.loads(__SPEC_JSON__)            # everything the shard needs, embedded in the source (no secrets)

@kbench.task(name=SPEC["task_name"], description="...")
def shard(llm) -> None:                      # llm = proxy client for the model this run was scheduled with
    # 1) clone/pip (record each step's rc and tail in a manifest)  2) load .env from the attached dataset
    # 3) run your loop using llm.model as the slug   4) write transport results + manifest to /kaggle/working
    ...
shard.run(kbench.llm)
```

Split work into **small shards** (we used 25 items) so one failure or one daily cap costs a shard, not the campaign. A task name is its identity:
change it when the cohort or settings change.

## 8. Pause, resume, never pay twice

1. **Classify failures:** `auth` (expired key) and `quota` (cap hit) -> **pause**, record nothing for the item in flight; `rate_limit` -> back off (30/60/120 s) then pause;
   anything else -> store as evidence and stop after N in a row (a broken endpoint would fail everything).
2. **Journal every answered turn** immediately; a resumed item replays answered turns for free and only re-sends the failed one. If the request would differ
   from the journal, fail closed.
3. **Identity on every record:** provider, model, prompts (hashes), decoding actually used, tools, dataset/cohort hash. Resume refuses a mismatch and names the differing keys.
4. **Persist driver state** (which task/model runs exist, status) remotely as well as locally so a new session adopts it instead of pushing/scheduling twice.
5. **Remainder tasks** for items a model still lacks, for that model only.
6. **Smoke test first:** one cheap model, two items, through every condition, as real hosted tasks; block the campaign until it passes.
7. Check quota (`quota --format json`) between items; pause when remaining drops under a reserve.

## 9. Saving results and capturing findings

* Two layers: **transport records** (written by the worker: answers, raw turns, usage, identity, no ground truth) and **evaluated records** (scored locally, so labels never travel to Kaggle).
* Keep one capture format for probes and real runs: store each request/response verbatim (finish reason, usage, latency, errors; images redacted),
  unknowns as `None`, and never delete a failed attempt.
* Persist to storage you already own (e.g. an existing private Hugging Face repo) with atomic, identity-checked bundles per shard so several Kaggle accounts can contribute;
  do not create repos from code.
* Per-episode "hindsight" rows (trajectory, tool calls, answer, truth, usage, outcome success/partial/failure/format_failure/error) make later retrieval or analysis possible.
* Report: a markdown table per model/condition with a paired bootstrap CI for any "A vs B" claim.

## 10. Open / unverified (do not assume)

* Heavy-dependency installs on a worker (torch, ultralytics, model weights): only the smoke test proves them.
* Claude determinism, run-to-run variance of thinking models, any prompt-caching discount.
* Whether the catalogue or prices changed since 2026-11-14.
* How daily caps behave across several accounts; free-plan limits on scheduled runs.
* SDK internals we avoided (`chats.new` details, assistant-message injection, any SDK max-token option).

## 11. Cheap checks before you spend

```
kaggle --version && kaggle benchmarks --help
kaggle benchmarks quota --format json
kaggle benchmarks tasks models
# one tiny vision request per candidate model (cents): who answers, who reads images, what parameters they need
```

## 12. Pitfalls

* Assuming the interactive roster equals the hosted catalogue.
* Batch `-m a -m b` (400).
* Expecting `tasks push` not to start a run.
* Using Kaggle Secrets in hosted runs; use the private env dataset.
* Cloning an unpushed branch: the worker only sees what is on the remote default branch.
* Bare-worker import failures: write an import-failure manifest and test the requirements list.
* Python 3.12/3.13 drift in any `ast`-based hash.
* Sharing one scratch folder between models' result bundles (they collide): key it by the settings fingerprint.
* Recording quota or key expiry as a model error: pause instead.
* Re-running the smoke test every new session: restore the campaign state first and skip it if it already passed.

## 13. Probe history (what each settled)

| Probe | Settled |
|---|---|
| 1 (2026-11-12) | SDK basics, raw proxy reachability |
| v5 | credential flow, `/openapi/chat/completions`, system + image + history, usage/cost fields, quota |
| v6 | per-model token parameter and temperature policy, vision, later image in multi-turn, big images, concurrency, real prices |
| v7 | full roster (37 listed vs served), image-token fidelity, reasoning knobs |
| v8 | hosted runs over the full catalogue; batch `-m` of 37 models fails with 400 |
| v9 | one model at a time works; worker environment; dataset mount; Secrets do not work; >=150 min runs; 34 vision models |

## 14. Thermo-agent-specific lessons (2026-11-25)

These are the project rules layered on top of the generic guide. They live here so a new
session does not rediscover them by spending quota.

### 14.1 Models

- Only `gpt-6-luna` and `claude-haiku-5-5` may be called. No premium, mid-tier, or
  "just one probe" model without explicit user permission. No bulk campaign.
- The reusable runner is `notebooks/kaggle_model_runner.py` (the `.ipynb` is generated
  from it). It hard-blocks `gemini-3.8-flash` and `gemini-3.5-flash-lite`, and it
  returns before any model call when the scheduled slug is not allowed. That is why a
  `tasks push` auto-run on the account default costs nothing.
- The skip branch is at the very start of the task, before `llm.prompt` or any client
  construction. It now writes `/kaggle/working/manifest.json` with
  `status="skipped"`, `stage="model_not_allowed"`, the actual model slug,
  `allowed_models`, `blocked_models`, `model_calls=0`, `quota_spent_usd=0.0`, and
  `note="returned before any model call"`, then returns. This is the exact "3.8 runs
  empty and spends nothing" behaviour; do not replace it with cancel-after-start.
- Never try to select the push auto-run model with `LLM_DEFAULT`. Only explicit
  `tasks run -m <slug>` selects a model.
- If a blocked run ever executes because the guard was bypassed, keep its journal as
  evidence, record the actual model and `usage.cost`, mark its rows rejected, and never
  ingest it into a corpus.
- `allowed_models` is in `harness_config.json` and can be overridden with
  `HARNESS_ALLOWED_MODELS`; keep both at the two budget slugs.
- One `tasks run -m <slug>` per model. The repo scripts loop over models; the CLI does
  not accept a batch.
- The provider-prefixed slug inside a run is read from `llm.model`; the CLI slug is the
  bare one (`claude-haiku-5-5`).

### 14.2 Launch and download (repo commands)

```powershell
# push once; the default-model auto-run is skipped by the in-task guard
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks push `
  thermo-teacher-cotproof-s00 --file notebooks/kaggle_model_runner.py `
  --kaggle-dataset rezanadimi/thermo-agent-data

# one explicit budget model per call
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks run `
  thermo-teacher-cotproof-s00 --model gpt-6-luna --wait 1800 -v
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks run `
  thermo-teacher-cotproof-s00 --model claude-haiku-5-5 --wait 1800 -v

# download; then gate and ingest locally
powershell -NoProfile -File scripts/kaggle_project.ps1 benchmarks tasks download `
  thermo-teacher-cotproof-s00 --model gpt-6-luna `
  --output data/teacher_runs/kaggle_hosted/budget_batches
```

`scripts/kaggle_project.ps1` reads the single token from `.env`, scopes the SOCKS proxy
to that one command, and never prints the value. A raw `kaggle` command needs the same
proxy scoped around it.

### 14.3 Data and credentials

- Local data goes to the private HF dataset, never directly to Kaggle; the private
  bridge kernel downloads HF and publishes the Kaggle dataset version.
- The Kaggle dataset copy can lag the HF copy. A preflight 404 for a new shard means the
  dataset version is stale, not that the model or task is broken. Refresh the bridge,
  verify `datasets status <id>`, then run.
- Keep dataset upload directories flat. Uploaded zips auto-extract; a nested or renamed
  path breaks the runner's shard lookup.
- One Kaggle account only. Exactly one `KAGGLE_API_TOKEN` in the repo-root `.env`; no
  second token, no account switch to work around a proxy abort. Retry the read-only
  command instead.
- `HF_TOKEN`, `KAGGLE_API_TOKEN`, `MODEL_PROXY_*`, and `LLM*` come only from the
  repo-root `.env` (locally) or an attached private `env` dataset (hosted). Never print,
  journal, or commit them.
- Never create another dataset, hosted task, or kernel; never delete or modify anything
  else on Kaggle without explicit permission.

### 14.4 Hosted-run shape that worked

- 40-row pilot shards, 10 per task, equal easy/medium/hard tiers.
- `tasks push` then explicit per-model `tasks run`; download the journal, gate at 8,192
  tokens locally, then ingest with the model tag and run id.
- The task source embeds a `SPEC` JSON; prompts and configs are attached datasets, not
  secrets. The worker writes a manifest so a failure is diagnosable without logs.
- Actual pilot spend: luna `$0.07797`, haiku `$0.14529`. Treat any order-of-magnitude
  jump as a configuration bug, not normal variance.

### 14.5 Known failure signatures

| Symptom | Real cause | Action |
|---|---|---|
| push seems to start an unwanted run | account-default auto-run | guard returns `skipped` with `model_calls=0` and writes a skip manifest; do not cancel by pushing again |
| `tasks run -m a -m b` -> 400 | batch models in one call | loop locally |
| new shard preflight 404 | stale Kaggle dataset version | refresh HF -> Kaggle bridge, verify status |
| auth/quota error mid-run | key expiry or cap | pause, write no result for the in-flight item, resume |
| journal missing after download | wrong output dir / model path | scan download root for the bare or provider-prefixed slug |
| runner exits before any call | scheduled model not in allow-list | expected; schedule `gpt-6-luna` or `claude-haiku-5-5` |

