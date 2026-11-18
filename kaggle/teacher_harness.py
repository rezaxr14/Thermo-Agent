"""Kaggle notebook: frontier model in the LangGraph harness with REAL physics tools + skills + memory.

Built from this source with `python kaggle/make_notebook.py`. This is the tool-using CoT generator:
the frontier model proposes an answer, calls first/second-law, validity and solve tools, reads the
numeric results, self-corrects, and finalizes. The full trajectory is kept for SFT so our 4B learns
to use tools/skills/memory. Old zero-shot teacher rows are kept; these rows are additive.
"""

# %% [markdown]
# # thermo-agent: frontier model with physics tools, skills and memory
#
# Attach `thermo-agent-code` (zip from `scripts/make_kaggle_bundle.ps1`) and `thermo-teacher-prompts`.
# The model runs on Kaggle's side; the physics tools (solver + verifiers) run on CPU. No GPU needed.
# Output: `harness_completions.jsonl` with the answer, the full tool-using trajectory, reasoning,
# token usage and cost per row. Ingest accepted rows with `transfer.ingest` (trajectory preserved).

# %%
import json
import os
import pathlib
import shutil
import sys
import time
import zipfile

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "thermo-agent"
bundles = []
for _root, _dirs, _files in os.walk("/kaggle/input", followlinks=True):
    if "kaggle_bundle.zip" in _files:
        bundles.append(pathlib.Path(_root) / "kaggle_bundle.zip")
if not bundles:
    raise SystemExit("attach the Kaggle dataset thermo-agent-code")
shutil.rmtree(ROOT, ignore_errors=True)
with zipfile.ZipFile(bundles[0]) as zf:
    zf.extractall(WORK)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = WORK / "harness_completions.jsonl"
MODEL = os.environ.get("HARNESS_MODEL", "gemini-3.5-flash-lite")
LIMIT = int(os.environ.get("LIMIT", "0"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))

# %%
from thermo_agent.kaggle_notebook import find_input, setup, token
ROOT = setup(corpora=())   # code + prompt datasets only, no training corpora

from thermo_agent.teacher import memory as mem
from thermo_agent.teacher.agent_graph import run as run_graph, build_context
from thermo_agent.teacher.kbench_adapter import make_respond, usage_of
from thermo_agent.teacher.toolkit import tool_names
from thermo_agent.teacher.transfer import load_exports

# %%
exports = load_exports(find_input("thermo-teacher-prompts"))
rows = list(exports.values())
if LIMIT:
    rows = rows[:LIMIT]
print(f"{len(rows)} prompts loaded", flush=True)

# Fresh failure memory (reloaded each row so it is always current).
MEM = pathlib.Path("data/memory")
def fresh_notes():
    notes = []
    for p in MEM.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            try:
                notes.append(mem.Note(**json.loads(line)))
            except Exception:
                pass
    return mem.dedupe(notes)

# %%
import kaggle_benchmarks as kbench
from kaggle_benchmarks import actors

token()
llm = kbench.llms.get(MODEL, kbench.llm)
print(f"model={MODEL}; tools={tool_names()}", flush=True)

# %%
started, written = time.time(), 0
with OUT.open("a", encoding="utf-8") as fh:
    for i, row in enumerate(rows):
        rid, task = str(row["id"]), row.get("task", "repair")
        rec = {"id": rid, "task": task, "model": MODEL, "harness": "tools",
               "tools": tool_names(), "topology": row.get("topology"),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            ctx = build_context({"row": row, "task": task, "notes": fresh_notes(),
                                 "k_memory": 3, "mode": "agents+skills"})
            usage_holder = {}
            respond = make_respond(llm, kbench, actors, system_prompt=ctx["system_prompt"],
                                   max_tool_rounds=ROUNDS, usage_holder=usage_holder)
            state = run_graph(row, task, respond, notes=fresh_notes(), max_rounds=ROUNDS)
            rec["completion"] = state["final"]
            rec["verdict"] = {k: v for k, v in state["verdict"].items() if k != "failed"}
            rec["trajectory"] = {"turns": state["turns"], "notes_used": state["notes_used"],
                                 "tools_available": tool_names()}
            rec["sft_completion"] = state["sft_completion"]
            rec["usage"] = usage_of(usage_holder["chat"]) if "chat" in usage_holder else {}
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        fh.write(json.dumps(rec) + "\n"); fh.flush(); written += 1
        if written % 25 == 0:
            print(f"{i+1}/{len(rows)}  {written} new", flush=True)

print(f"wrote {written} rows to {OUT}; ingest with transfer.ingest --harness tools")
