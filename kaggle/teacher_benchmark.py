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

import kaggle_benchmarks as kbench

# Pick any key printed below (or "google/gemini-3.8-flash", "anthropic/claude-sonnet-5-5@default", ...).
MODEL = "anthropic/claude-sonnet-5-5@default"
TASKS = ("repair", "ledger", "audit", "design")
LIMIT = 0            # 0 = every exported row
OUT = pathlib.Path("/kaggle/working/completions.jsonl")

# %%
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
                  "harness": row.get("harness", "none"), "topology": row.get("topology")}
        try:
            with kbench.chats.new(f"thermo_{i}", system_instructions=system or None):
                record["completion"] = str(llm.prompt(user))
        except Exception as exc:                       # keep going; the local ingest counts the misses
            record["error"] = f"{type(exc).__name__}: {exc}"
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        written += 1
        if written % 25 == 0:
            rate = written / max(time.time() - started, 1e-9)
            print(f"{i + 1}/{len(rows)}  {written} new  {rate:.2f}/s  eta {(len(rows) - i - 1) / rate / 60:.0f} min",
                  flush=True)

print(f"wrote {written} rows to {OUT}")
print("download it from the notebook's output panel, then run the local ingest command")
