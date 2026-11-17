"""Run the thermo-agent teacher prompts through a frontier model on Kaggle Benchmarks.

Two ways to run this, both with Kaggle's own model access and quota (no tunnel, no self-hosted teacher):

1. Notebook import (no API credentials needed). Build the notebook and upload the prompts once:

       python kaggle/make_notebook.py                      # writes kaggle/teacher.ipynb
       # upload data/frontier_export/ as a private dataset at kaggle.com/datasets (drag the folder or its zip)
       # then: kaggle.com/notebooks -> File -> Import notebook -> kaggle/teacher.ipynb -> Run all

2. Paste this file into a cell of a notebook created from https://www.kaggle.com/benchmarks/tasks/new, which
   already has `kaggle_benchmarks` installed.

Either way the output is `/kaggle/working/completions.jsonl`, one JSON line per attempt, and the local machine
decides what is true:

    python -m thermo_agent.teacher.transfer ingest data/frontier_export completions.jsonl data/frontier \\
        --memory-dir data/memory --model anthropic/claude-sonnet-5-5@default

Every prompt is sent as its own system instruction plus its user message, with no tools and no thinking budget: the
answer is graded afterwards by our own tool suite, and only what passes becomes a training row. The loop skips ids
it has already answered and appends one line per attempt, so a session that dies keeps its progress; failures are
recorded with their error instead of aborting the run.
"""

# %% [markdown]
# # thermo-agent: frontier teacher pass
#
# Sends every exported prompt to one frontier model and writes `completions.jsonl`. Nothing here grades anything:
# the answers travel back to the local machine, where `thermo_agent.teacher.transfer ingest` re-solves every plant
# and keeps only the answers that pass the physics verifiers.

# %%
import json
import pathlib
import time
import os

import kaggle_benchmarks as kbench

# Pick any key printed below (or "google/gemini-3.8-flash", "anthropic/claude-sonnet-5-5@default", ...).
MODEL = "anthropic/claude-sonnet-5-5@default"
TASKS = ("repair", "ledger", "audit", "design")
LIMIT = 0            # 0 = every exported row
OUT = pathlib.Path("/kaggle/working/completions.jsonl")

# Optional, opt-in: probe a shortlist of models with one small prompt and pick the cheapest that returns a usable
# answer. The probe results (model, tokens, nanodollars, latency) are written next to OUT and pushed to the repo.
PROBE_MODELS = False
PROBE_SHORTLIST = ("anthropic/claude-sonnet-5-5@default", "google/gemini-3.8-flash")

# Resume: progress lives in the project's private HF repo, so a fresh Kaggle account with new quota can continue.
# The token is a Kaggle Secret. Without the code dataset or a token the notebook still runs, but progress stays local.
REPO = "Reza-Nadimi/thermo-agent-checkpoints"
PROGRESS_PREFIX = "teacher/progress.jsonl"
SYNC = False
progress = None
try:
    from thermo_agent.teacher import progress as _progress
    from thermo_agent.train import hf_sync
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    if os.environ.get("HF_TOKEN"):
        hf_sync.resolve_repo(REPO)
        progress = _progress
        SYNC = True
except Exception as exc:
    print("HF progress sync unavailable (local-only):", exc)


def _usage(chat):
    """Kaggle Benchmarks Usage: tokens, cost in nanodollars and latency, when the backend reports them."""
    try:
        u = chat.usage
        return {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None),
                "total_cost_nanodollars": getattr(u, "total_cost_nanodollars", None),
                "total_backend_latency_ms": getattr(u, "total_backend_latency_ms", None)}
    except Exception:
        return {}


# %% [markdown]
# ## Model choice (recorded decision)
# A short probe is optional. It measures the actual token cost and latency Kaggle reports and writes the decision
# to `model_probe.jsonl`. We prefer the cheapest model that returns a non-empty answer; with PROBE_MODELS=False the
# chosen MODEL above is kept and its id and cost are still recorded on every completion line.

# %%
probe = []
if PROBE_MODELS:
    for key in PROBE_SHORTLIST:
        if key not in kbench.llms:
            continue
        started = time.time()
        try:
            with kbench.chats.new(f"probe_{len(probe)}", system_instructions="Answer briefly.") as chat:
                answer = str(kbench.llms[key].prompt("Reply with the word READY."))
            probe.append({"model": key, "ok": bool(answer.strip()), "answer": answer[:80], "usage": _usage(chat),
                          "seconds": round(time.time() - started, 2)})
        except Exception as exc:
            probe.append({"model": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    usable = [p for p in probe if p["ok"]]
    if usable:
        MODEL = sorted(usable, key=lambda p: (p["usage"].get("total_cost_nanodollars") or 0))[0]["model"]
    (OUT.parent / "model_probe.jsonl").write_text("\n".join(json.dumps(p) for p in probe) + "\n", encoding="utf-8")
    print("model decision:", json.dumps(probe, indent=1))

print("models available in this session:")
for key in sorted(kbench.llms):
    print(" ", key)
llm = kbench.llms[MODEL] if MODEL in kbench.llms else kbench.llm
print("using:", MODEL if MODEL in kbench.llms else "the session default model")


def find_export_root() -> pathlib.Path:
    """The uploaded dataset, wherever Kaggle mounted it."""
    for base in (pathlib.Path("/kaggle/input"), pathlib.Path(".")):
        for path in sorted(base.rglob("export_manifest.json")):
            return path.parent
    raise SystemExit("no export_manifest.json under /kaggle/input: attach the frontier_export dataset")


root = find_export_root()
rows = []
for task in TASKS:
    path = root / f"{task}_train.jsonl"
    if path.exists():
        rows += [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
if LIMIT:
    rows = rows[:LIMIT]
if SYNC:
    # Pull the history written by another session/account before deciding what is already answered.
    merged = progress.sync(OUT, REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
    print("shared progress:", len(merged), "attempts;", len(progress.answered_ids(merged)), "answered")
    done = progress.answered_ids(merged)
else:
    done = {json.loads(l)["id"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
print(f"export root: {root}")
print(f"{len(rows)} prompts, {len(done)} already answered, model={llm.__class__.__name__}")

# %%
started = time.time()
written = 0
with OUT.open("a", encoding="utf-8") as fh:
    for i, row in enumerate(rows):
        if row["id"] in done:
            continue
        messages = row["prompt"]
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        record = {"id": row["id"], "task": row.get("task"), "model": MODEL,
                  "harness": row.get("harness", "none"), "topology": row.get("topology"),
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tools": "none"}
        try:
            with kbench.chats.new(f"thermo_{i}", system_instructions=system or None) as chat:
                record["completion"] = str(llm.prompt(user))
                record["usage"] = _usage(chat)
                traces = []
                try:
                    traces = [m.reasoning_traces for m in chat.messages if getattr(m, "reasoning_traces", None)]
                except Exception:
                    pass
                if traces:
                    record["reasoning"] = "\n".join(str(t) for t in traces)
        except Exception as exc:                       # keep going; the local ingest counts the misses
            record["error"] = f"{type(exc).__name__}: {exc}"
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        written += 1
        if SYNC and written % 25 == 0:                # checkpoint progress to the shared repo without rewriting OUT
            try:
                remote = progress.pull(REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
                progress.push(progress.dedupe(remote + progress.load(OUT)), REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
            except Exception as exc:
                print("progress push failed (will retry):", type(exc).__name__, exc, flush=True)
        if written % 25 == 0:
            rate = written / max(time.time() - started, 1e-9)
            print(f"{i + 1}/{len(rows)}  {written} new  {rate:.2f}/s  eta {(len(rows) - i - 1) / rate / 60:.0f} min",
                  flush=True)

print(f"wrote {written} rows to {OUT}")
if SYNC:
    final = progress.sync(OUT, REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
    print("final shared progress:", len(final), "attempts;", len(progress.answered_ids(final)), "answered")
    print("recorded cost:", json.dumps(progress.usage_totals(final)))
print("download this file from the notebook's output panel, then run the local ingest command")
