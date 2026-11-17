"""Kaggle notebook source: the frontier model inside our harness, with physics tools exposed.

Build with `python kaggle/make_notebook.py`. This is the third comparison arm in the paper: the same prompts and
ground-truth rows as the teacher pass, but the model may call our verifiers while it works and we keep its reasoning
trace. The fourth arm is our trained 4B model; zero-shot and prompt-only frontier arms come from teacher.ipynb.
"""

# %% [markdown]
# # thermo-agent: frontier model with our physics tools
#
# Attach `thermo-agent-code` (the zip from `scripts/make_kaggle_bundle.ps1`) and `thermo-teacher-prompts`.
# Add the `HF_TOKEN` Kaggle Secret. Choose **GPU T4 x2** only if you also want to train; this notebook itself needs
# no GPU — the tools run on the CPU and the model runs on Kaggle's side.
#
# Output: `/kaggle/working/harness_completions.jsonl` with answer, tool calls, reasoning trace, token usage and
# cost per row. Progress is mirrored to `teacher/harness_progress.jsonl` in the same private repo, so an account
# with fresh quota can resume.

# %%
import json
import os
import pathlib
import sys
import time
from thermo_agent.kaggle_notebook import WORK, find_input, setup, token

OUT = WORK / "harness_completions.jsonl"
REPO = "Reza-Nadimi/thermo-agent-checkpoints"
PROGRESS_PREFIX = "teacher/harness_progress.jsonl"
MODEL = "anthropic/claude-sonnet-5-5@default"
LIMIT = 0
TOOL_ROUNDS = 3

# %%
ROOT = setup(corpora=())          # this arm needs the code and prompt datasets, not the training corpora

from thermo_agent.teacher import progress, validate
from thermo_agent.train import hf_sync
from thermo_agent.teacher.transfer import load_exports

# %%
exports = load_exports(find_input("thermo-teacher-prompts"))
rows = list(exports.values())
if LIMIT:
    rows = rows[:LIMIT]
print(f"{len(rows)} prompts loaded")


# %%
# Our tools. kbench builds the JSON schema from the type annotations, so keep them plain strings.
def check_answer(task: str, answer: str, row_id: str) -> str:
    """Run the project's physics tool suite on a candidate answer and report what failed."""
    row = exports.get(row_id)
    if row is None:
        return "Unknown row id."
    verdict = validate.validate(task, answer, row)
    return json.dumps({"accepted": verdict["accepted"], "reward": verdict["reward"],
                       "reasons": verdict["reasons"]})


TOOLS = [check_answer]
print("exposed tools:", [f.__name__ for f in TOOLS])

# %%
import kaggle_benchmarks as kbench

token()
SYNC = bool(os.environ.get("HF_TOKEN"))
if SYNC:
    hf_sync.resolve_repo(REPO)
    merged = progress.sync(OUT, REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
    done = progress.answered_ids(merged)
else:
    done = {json.loads(l)["id"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
llm = kbench.llms.get(MODEL, kbench.llm)
print(f"model={MODEL}; {len(done)} rows already answered")


def _usage(chat):
    try:
        u = chat.usage
        return {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None),
                "total_cost_nanodollars": getattr(u, "total_cost_nanodollars", None),
                "total_backend_latency_ms": getattr(u, "total_backend_latency_ms", None)}
    except Exception:
        return {}


def _trace(chat):
    try:
        return "\n".join(str(m.reasoning_traces) for m in chat.messages if getattr(m, "reasoning_traces", None))
    except Exception:
        return ""


def _tool_calls(chat):
    calls = []
    try:
        for m in chat.messages:
            for c in getattr(m, "tool_calls", None) or []:
                calls.append({"name": getattr(c, "name", str(c)), "arguments": getattr(c, "arguments", "")})
    except Exception:
        pass
    return calls

# %%
SYSTEM = (
    "You are an expert thermodynamic agent. You may call check_answer(task, answer, row_id) as often as you need "
    "(at most {} times) to test a candidate answer against our physics verifiers; use the feedback to improve it. "
    "Finish with the final answer in the task's XML tags, as in the training corpus. Do not reveal private data."
).format(TOOL_ROUNDS)

started = time.time()
written = 0
with OUT.open("a", encoding="utf-8") as fh:
    for i, row in enumerate(rows):
        rid = str(row["id"])
        if rid in done:
            continue
        task = row.get("task", "repair")
        user = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] != "system")
        record = {"id": rid, "task": task, "model": MODEL, "harness": "tools", "tools": [f.__name__ for f in TOOLS],
                  "topology": row.get("topology"), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            with kbench.chats.new(f"harness_{i}", system_instructions=SYSTEM) as chat:
                record["completion"] = str(llm.prompt(user, tools=TOOLS))
                record["usage"] = _usage(chat)
                record["reasoning"] = _trace(chat)
                record["tool_calls"] = _tool_calls(chat)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        written += 1
        if written % 25 == 0:
            if SYNC:
                try:
                    remote = progress.pull(REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
                    progress.push(progress.dedupe(remote + progress.load(OUT)), REPO,
                                  os.environ["HF_TOKEN"], PROGRESS_PREFIX)
                except Exception as exc:
                    print("progress push failed (will retry):", exc, flush=True)
            rate = written / max(time.time() - started, 1e-9)
            print(f"{i + 1}/{len(rows)}  {written} new  {rate:.2f}/s", flush=True)

print(f"wrote {written} rows to {OUT}")
if SYNC:
    final = progress.sync(OUT, REPO, os.environ["HF_TOKEN"], PROGRESS_PREFIX)
    print("recorded cost:", json.dumps(progress.usage_totals(final)))
print("download harness_completions.jsonl and ingest it with --harness tools")
