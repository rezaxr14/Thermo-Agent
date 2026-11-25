---
name: kaggle-hosted-models-and-data
description: Run hosted frontier-model evaluations through Kaggle Benchmarks without spending quota on the account default (gemini-3.8-flash), keep per-model runs and journals resumable, and bridge local corpora to Kaggle through Hugging Face. Repo-agnostic field guide.
---

# Kaggle hosted models + HF data bridge: reusable field guide

This is the short, opinionated version of what actually works. It is written to be copied
into any repository. Replace the `<...>` examples with your own repo, model allow-list,
dataset slugs, and paths.

Measured against Kaggle CLI 2.2.5 / `kaggle-benchmarks` 0.6.3 in November 2026. Kaggle
changes this surface often. Re-run the checks in **§12** before trusting a number.

---

## 0. The one-paragraph mental model

Kaggle Benchmarks runs your Python task on a Kaggle worker and bills model calls to your
**Kaggle quota**, not to vendor API keys. There are two routes with different credentials
and different model access: an **interactive proxy** from your own session, and a
**hosted task** (`tasks push` + `tasks run`) that keeps running after you close the tab.
A hosted task is identified by its name and version; a run is one `(task version, model)`
pair. Results and journals land in `/kaggle/working` and come back with
`tasks download`.

The account has a **default model**, and `tasks push` immediately starts one run on it.
In the account this guide was measured on that default is `gemini-3.8-flash`. If you do
not want to pay for it, make the task exit before its first model call when the scheduled
model is not on your allow-list. That is cheaper and more reliable than trying to cancel
a run after it starts.

---

## 1. Choose the route

| Route | Your code runs | Models reachable | Credentials | Use for |
|---|---|---|---|---|
| Interactive proxy | your own session (local/Colab/notebook) | curated subset | `kaggle benchmarks init` / `auth` | quick probes, one-off checks |
| Hosted task | Kaggle worker, server-side | full catalogue | platform-injected proxy | real evaluations, long runs, many items |

Hosted lifecycle:

```text
tasks push NAME -f task.py -d owner/dataset     # upload task source + attach datasets
tasks run  NAME -m <cli-slug>                   # ONE model per call
tasks status NAME                               # queued / running / completed / failed
tasks download NAME -o OUT -m <slug> -f         # OUT/NAME/<version>/<model>/<run_id>/...
```

`tasks push` also starts the account-default run. Design the task so a non-allowed model
returns immediately; see **§4**.

---

## 2. Install, authenticate, verify

```bash
pip install "kaggle>=2.2.5" "kaggle-benchmarks>=0.5.0"
kaggle --version
kaggle benchmarks --help
kaggle benchmarks quota --format json
kaggle benchmarks tasks models
```

Interactive:

```bash
kaggle benchmarks init -y --env-file .env
# or: kaggle benchmarks auth -y
```

This writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`, `LLM_DEFAULT`, `LLMS_AVAILABLE`.
The interactive key expires in roughly **1 hour**; re-run `init`/`auth` to renew.
Without `init`, `kbench.llms` is empty.

Hosted:

- The platform injects the proxy variables; the key lasts about **12 hours**.
- Runs of at least **150 minutes** are allowed.
- **Kaggle Secrets do not work inside hosted runs.** Put your own tokens in a private
  Kaggle dataset containing a file named `env` (uploads can drop dotfiles), attach it,
  and read it from `/kaggle/input/...`.
- Never print, journal, or commit a token. Never let an attached env file override
  `MODEL_PROXY*`, `KAGGLE_*`, or `LLM*`.

Local rule: one credential file, at the repo root, read at call time. Helpers should pass
tokens directly to the client and never echo them.

---

## 3. Quota and cost

```bash
kaggle benchmarks quota --format json
```

Returns daily/monthly `remaining`, `used`, `total`, `refillAt`. The measured account had
a **$10 daily / $100 monthly** allowance. Quota moves in step with summed response costs.

Per response:

- `usage.prompt_tokens`, `usage.completion_tokens`
- `usage.cost.input_tokens_cost_nanodollars`, `usage.cost.output_tokens_cost_nanodollars`
- `total_backend_latency_ms`

`completion_tokens` **includes hidden reasoning**, so `usage.cost` is the true spend.
A missing cost or unknown unit is `None`, never `0`.

Multi-turn chats re-send the history on every turn, including images. Input tokens grow
roughly quadratically with turns. Estimate before a long campaign.

---

## 4. The default-model trap: run only what you intended

This is the most expensive mistake in the whole surface.

1. **The account default is not your choice.** In the measured account it was
   `gemini-3.8-flash`. `tasks push` starts a run on it automatically.
2. **`LLM_DEFAULT` is not a selector.** It tells the SDK which model a convenience
   helper should use. It does not stop `tasks push` from using the account default.
3. **`tasks run` takes one model per call.** `-m a -m b` returned HTTP 400 in the
   measured CLI. Loop over models locally; do not batch them.
4. **Guard inside the task, before the first model call.** If the scheduled slug is not
   on the allow-list, write a small "skipped" manifest and return. Then the auto-run
   costs nothing.

Minimal guard. The important part is the **early return before the first model
call**, plus a manifest so the run is visibly skipped rather than an empty success:

```python
ALLOWED = ("<budget-model-a>", "<budget-model-b>")
BLOCKED = ("gemini-3.8-flash", "gemini-3.5-flash-lite")

def allowed(model: str) -> bool:
    m = (model or "").lower()
    return not any(b in m for b in BLOCKED) and any(a in m for a in ALLOWED)

@kbench.task(name="my-task")
def shard(llm) -> None:
    model = getattr(llm, "model", "") or getattr(llm, "name", "") or "unknown"
    if not allowed(model):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "manifest.json").write_text(json.dumps({
            "status": "skipped",
            "stage": "model_not_allowed",
            "model": model,
            "allowed_models": list(ALLOWED),
            "blocked_models": list(BLOCKED),
            "model_calls": 0,
            "quota_spent_usd": 0.0,
            "note": "returned before any model call",
        }, indent=1), encoding="utf-8")
        return
    ...
```

Exact skip recipe:

1. Read the scheduled slug from `llm.model` (or `llm.name`).
2. Check the blocked list first, then the allow-list.
3. Create the output directory and write the skipped manifest.
4. `return` before any prompt, tool, or client construction that could bill.
5. Do **not** cancel and re-push to "fix" it; that just creates another default run.
6. Do **not** try to select the model with `LLM_DEFAULT`; it does not control the
   push auto-run.
7. If a blocked journal appears anyway, keep it as evidence, record its `usage` and
   actual model slug, mark every row rejected, and never ingest it.

Project rules that fall out of this:

- Keep a hard **blocked** list, not only an allow-list. A config typo must not be able
  to turn on a premium/default model.
- Pin budget models in code and config. Require explicit human permission to add any
  other model.
- `tasks push` is not free by itself, but a guarded task makes its auto-run free.
- Never run a "quick test" on the default model "just to see"; it is the exact model you
  were trying to avoid.
- If you inherit a stale campaign, read the task name/version and run status **before**
  pushing again.

---

## 5. Calling a model

### Proxy (OpenAI-compatible)

```python
import os, requests

base = os.environ["MODEL_PROXY_URL"].rstrip("/")
url = base + "/chat/completions"          # drop the suffix if base already ends in /openapi
r = requests.post(
    url,
    headers={"Authorization": "Bearer " + os.environ["MODEL_PROXY_API_KEY"]},
    json={
        "model": "<proxy-slug>",
        "max_tokens": 512,
        "temperature": 0,
        "messages": [{"role": "user", "content": "..."}],
    },
    timeout=300,
)
```

Measured working: system messages, base64 `image_url` parts, multi-turn with a later
image, `max_tokens` enforced (`finish_reason: length`), 16,384 output tokens, 5 MB images,
**16 parallel requests**, no rate-limit headers. `generateContent` also works for Gemini.

### Per-model quirks

Learn parameters from the provider's HTTP 400 and record what you used on every request.

| Quirk | Models seen |
|---|---|
| rejects `temperature` (not greedy) | claude-sonnet-5 |
| not repeatable at temperature 0 | gemini-3.6-flash |
| hidden reasoning billed by default; `thinking_budget: 0` / `reasoning_effort: none` disables it | gemini-3-flash-preview, gemini-3.6-flash |
| `temperature` accepted only while `reasoning_effort` is unset | gpt-5.4-nano |
| wants `max_completion_tokens` instead of `max_tokens` | learn from the 400 |
| image tokens: Gemini ~1.1k at any size; GPT/Claude scale with pixels (2048x1024: 2466 / 2753); `detail`/`media_resolution` ignored | all |
| no prompt-caching discount observed | all measured |

### SDK

```python
import kaggle_benchmarks as kbench

@kbench.task(name="...")
def run(llm) -> None:          # return annotation required if it returns a value
    llm.prompt("...")
```

- Images: `kaggle_benchmarks.content_types.images` (`from_path`, `from_url`, `from_base64`).
- Chats: `chats.new(name, system_instructions=...)`.
- Manual model: `ModelProxy(model=..., api_key=..., base_url=...)`.
- The SDK disables `temperature` on the OpenAI route. For full control of the transcript
  (agent loops), call the proxy directly.

---

## 6. Model roster and slugs

Two lists, do not confuse them:

- `kaggle benchmarks tasks models` -> bare CLI slugs.
- Interactive key -> only a curated subset; the rest return "model not found".
- Hosted -> most of the catalogue, including vision models. Some listed models still
  404. Do not assume a listing is callable.

Slug rule: `tasks run -m` takes the **bare CLI slug** (for example
`claude-haiku-5-5`). Inside the run, the proxy wants the **provider-prefixed** slug
(for example `anthropic/claude-haiku-5-5`), which you can read from `llm.model`.

Measured vision-capable hosted models and indicative prices ($/Mtok in/out from one tiny
request; treat as a probe, not a price list):

- Anthropic: `claude-haiku-4-5-20251001` 1/5, `claude-haiku-5-5` 0.1/0.5,
  `claude-sonnet-4-6-default` 3/15, `claude-sonnet-5-default` 2/10,
  `claude-sonnet-5-5-default` 2/10, `claude-opus-4-5-20251101` 5/25,
  `claude-opus-4-6/4-7/4-8-default` 5/25, `claude-opus-5-default` 5/25,
  `claude-opus-5-5-default` 4/20
- Google: `gemini-2.5-flash` 0.3/2.5, `gemini-2.5-pro` 1.25/10,
  `gemini-3-flash-preview` 0.5/3, `gemini-3.1-flash-lite-preview` 0.25/1.5,
  `gemini-3.1-pro-preview` 2/12, `gemini-3.5-flash-lite` 0.3/2.5,
  `gemini-3.6-flash` 0.75/3.75, `gemini-3.8-flash` 0.75/3.75 (**the account default**),
  `gemini-4-argon` 2/10, `gemma-4-26b-a4b-it` 0.15/0.6, `gemma-4-31b-it` 0.15/0.6
- OpenAI: `gpt-5.4-2026-03-05` 2.5/15, `gpt-5.4-mini-2026-03-17` 0.75/4.5,
  `gpt-5.4-nano-2026-03-17` 0.2/1.25, `gpt-5.5-2026-04-23` 5/30,
  `gpt-5.6-luna` 0.2/1.2, `gpt-5.6-sol` 4/20, `gpt-5.6-terra` 2/12,
  `gpt-6-astra` 10/50, `gpt-6-sol` 2/10, `gpt-6.1-sol` 2/10
- xAI: `grok-4.6` 2/6, `grok-4.7` 1.3/6

Budget defaults in the project this guide came from: `gpt-6-luna` and
`claude-haiku-5-5`. Everything else needs explicit permission.

---

## 7. Task template

```python
import json, os, pathlib
import kaggle_benchmarks as kbench

SPEC = json.loads(__SPEC_JSON__)       # embed non-secret settings in the source
OUT = pathlib.Path("/kaggle/working")
ALLOWED = ("<budget-model-a>", "<budget-model-b>")
BLOCKED = ("gemini-3.8-flash", "gemini-3.5-flash-lite")

def write_manifest(**fields):
    (OUT / "manifest.json").write_text(json.dumps(fields, indent=1), encoding="utf-8")

@kbench.task(name=SPEC["task_name"], description="...")
def shard(llm) -> None:
    if not allowed(getattr(llm, "model", "")):
        write_manifest(status="skipped", reason="model not allowed")
        return
    # 1) load inputs from attached datasets
    # 2) clone/install only if needed; record rc + tail in the manifest
    # 3) run the loop using llm.model as the provider-prefixed slug
    # 4) journal every answered turn immediately
    # 5) write results + manifest to /kaggle/working
    ...

shard.run(kbench.llm)
```

Split work into **small shards** (25-100 items). One failure, one daily cap, or one bad
item then costs a shard, not the campaign. A task name is its identity: change it when
the cohort or settings change.

---

## 8. Pause, resume, never pay twice

1. **Classify failures** before retrying:
   - `auth` (expired key) and `quota` (cap hit) -> **pause**, write nothing for the
     in-flight item.
   - `rate_limit` -> back off 30/60/120 s, then pause if it persists.
   - anything else -> store the evidence; stop after N in a row.
2. **Journal every answered turn immediately.** A resumed item replays answered turns
   for free and re-sends only the failed one. If the request would differ from the
   journal, fail closed.
3. **Identity on every record:** provider, model, prompt hash, decoding actually used,
   tools, dataset/cohort hash. Resume refuses a mismatch and names the differing keys.
4. **Persist driver state** (which task/model runs exist and their status) remotely as
   well as locally, so a new session adopts it instead of pushing/scheduling twice.
5. **Remainder tasks** only for the items a model still lacks.
6. **Smoke test first:** one cheap allowed model, two items, through every condition, as
   a real hosted task. Block the campaign until it passes.
7. Check quota between items; pause below a reserve.

---

## 9. Data movement: local -> HF -> Kaggle

Keep three things separate: **code**, **data**, and **credentials**.

- **One repo per purpose.** Data goes to one dataset repo; model weights/adapters go to a
  different model repo; never mix them and never create a third mirror.
- **Hard allow-lists in the sync helper.** The helper refuses any repo/dataset id not in
  a constant list, before a network call. A typo must fail closed.
- **Additive only.** A refresh is a new version/commit with a message, tag, and manifest.
  Do not delete remote files.
- **HF is the only bridge to Kaggle.** Upload local data to the private HF dataset, then
  let a Kaggle kernel download it into a Kaggle dataset. Do not upload local data
  directly to Kaggle.
- **Flat upload directories.** Kaggle dataset uploads that are zipped are auto-extracted;
  keep each file at a predictable relative path and verify the mount after upload.
- **Manifests.** Every sync writes a local manifest with repo id, UTC timestamp,
  message, tag, and per-file path/sha256/size, and uploads a copy. Review a dry-run
  manifest before the real sync.
- **Prompts only where answers must not leak.** If a dataset exists only for evaluation,
  filter at every nesting depth to prompts and identifiers; never restore answers from it.

Minimal pattern:

```bash
# local -> HF (dry-run first; prints the plan and writes sha256s, sends nothing)
python scripts/sync_data.py --dry-run
python scripts/sync_data.py --message "corpus v3" --tag 2026-11-25

# HF -> Kaggle: run the private bridge kernel; it downloads the HF dataset and
# publishes a new version of the existing Kaggle dataset. No local upload.
```

Inside a Kaggle notebook:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="<owner>/<dataset>",
    repo_type="dataset",
    token=None,              # inject from a secret or attached private env file
    allow_patterns=["data/<corpus>/*", "configs/*"],
)
```

---

## 10. Saving results and capturing findings

- Two layers:
  - **transport records** written by the worker: answers, raw turns, usage, identity,
    no ground truth;
  - **evaluated records** scored locally, so labels never travel to Kaggle.
- One capture format for probes and real runs: store each request/response verbatim
  (finish reason, usage, latency, errors; images redacted), unknowns as `None`, and
  never delete a failed attempt.
- Persist to storage you already own, with atomic, identity-checked bundles per shard,
  so several accounts can contribute without collisions.
- Per-episode "hindsight" rows (trajectory, tool calls, answer, truth, usage, outcome)
  make later retrieval and analysis possible.
- Report a markdown table per model/condition, with a paired bootstrap CI for any
  "A vs B" claim.

---

## 11. Security and account hygiene

- One account, one token. Never mix a dataset-access account with an AI-billing account.
- Every credential comes from the repo-root `.env` or an attached private env dataset.
  No token in a notebook, task source, dataset, manifest, or log.
- Scope a proxy variable to the single command that needs it; do not proxy unrelated
  network calls.
- Redact secret-looking strings before journalling provider errors.
- Never create another dataset, hosted task, or kernel, and never delete or modify
  anything else on Kaggle, without explicit permission.
- Verify access with a read-only `datasets status <id>`; if a SOCKS proxy aborts the
  first attempt, retry the same read-only command rather than switching accounts.

---

## 12. Cheap checks before you spend

```bash
kaggle --version && kaggle benchmarks --help
kaggle benchmarks quota --format json
kaggle benchmarks tasks models
kaggle benchmarks tasks status <name>
# one tiny request per candidate model (cents): who answers, who reads images,
# what parameters they need, and which slug the proxy actually reports
```

If a dataset upload or preflight returns 404, stop and verify the dataset version and
mount path. A stale Kaggle dataset copy is not a model failure.

---

## 13. Pitfalls

- Assuming the interactive roster equals the hosted catalogue.
- Pushing a task and forgetting that the account-default run started.
- Trying to select the default model with `LLM_DEFAULT`.
- Batching several `-m` values in one `tasks run` (one model per call in the measured CLI).
- Using Kaggle Secrets inside hosted runs; use a private env dataset.
- Cloning an unpushed branch: the worker sees only the remote default branch.
- A bare worker missing a third-party import that your package pulls in indirectly.
  Write an import-failure manifest and test the exact import chain.
- Python 3.12 (worker) vs 3.13 (local) drift in `ast`-based hashes.
- Sharing one scratch folder between models; key result bundles by settings fingerprint.
- Recording quota or key expiry as a model error; pause instead.
- Re-running the smoke test every session; restore campaign state first.
- Uploading local data straight to Kaggle instead of through HF.
- Deleting remote files during a "refresh"; versions are additive.

---

## 14. Probe history (what each settled)

| Probe | Settled |
|---|---|
| 1 | SDK basics, raw proxy reachability |
| v5 | credential flow, `/openapi/chat/completions`, system + image + history, usage/cost, quota |
| v6 | per-model token/temperature policy, vision, later image in multi-turn, big images, concurrency, real prices |
| v7 | full roster (listed vs served), image-token fidelity, reasoning knobs |
| v8 | hosted runs across the catalogue; batch `-m` of many models fails with 400 |
| v9 | one model at a time works; worker environment; dataset mount; Secrets do not work; >=150 min runs; vision roster |

Anything not in this guide was not observed. Do not assume it.

---

## 15. Copy-paste checklist

- [ ] `kaggle --version`, quota, and model list checked.
- [ ] Task has an allow-list **and** a blocked-list guard before the first model call.
- [ ] `tasks push` auto-run is known and either adopted deliberately or skipped.
- [ ] One `tasks run -m <slug>` per model; no batch `-m`.
- [ ] Credentials only from `.env` or a private attached env dataset; no token in logs.
- [ ] Journal is written after every answered turn.
- [ ] Failure classes are handled (auth/quota/rate-limit/other) with pause/resume.
- [ ] Data went local -> HF -> Kaggle, additive, with a manifest.
- [ ] Dataset version and mount path verified before the campaign.
- [ ] Results downloaded and evaluated locally; labels never sent to Kaggle.
