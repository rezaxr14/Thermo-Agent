"""Run the thermo-agent teacher prompts through a frontier model on Kaggle Benchmarks.

Paste this file into a notebook created from https://www.kaggle.com/benchmarks/tasks/new (`kaggle_benchmarks` is
pre-installed there), attach the exported prompts as a notebook input, set MODEL and DATASET, and run.

    # locally
    python -m thermo_agent.teacher.transfer export data/sft data/frontier_export --per-task 400
    kaggle datasets create -p data/frontier_export -u        # or upload the folder in the UI

    # then download /kaggle/working/completions.jsonl and, locally:
    python -m thermo_agent.teacher.transfer ingest data/frontier_export completions.jsonl data/frontier \
        --memory-dir data/memory --model anthropic/claude-sonnet-5-5@default

Every prompt is sent verbatim (system message, then the user message) with no tools and no thinking budget: the
answer is graded afterwards by our own tool suite on the local machine, and only what passes becomes a training
row. The loop is resumable and writes one JSON line per attempt, so a session that dies keeps its progress.
"""
import json
import pathlib

import kaggle_benchmarks as kbench

MODEL = "anthropic/claude-sonnet-5-5@default"   # any key of kbench.llms
DATASET = "/kaggle/input/thermo-teacher-prompts"  # folder holding the exported {task}_train.jsonl files
OUT = pathlib.Path("/kaggle/working/completions.jsonl")
LIMIT = 0                                       # 0 = every exported row

llm = kbench.llms[MODEL]
rows = []
for path in sorted(pathlib.Path(DATASET).glob("*.jsonl")):
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
if LIMIT:
    rows = rows[:LIMIT]
done = {json.loads(l)["id"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
print(f"{len(rows)} prompts, {len(done)} already answered, model={MODEL}")

with OUT.open("a", encoding="utf-8") as fh:
    for i, row in enumerate(rows):
        if row["id"] in done:
            continue
        prompt = "\n\n".join(m["content"] for m in row["prompt"])
        record = {"id": row["id"], "task": row["task"], "model": MODEL}
        try:
            with kbench.chats.new(f"thermo_{i}"):
                record["completion"] = str(llm.prompt(prompt))
        except Exception as exc:                      # keep going; the local ingest counts the misses
            record["error"] = f"{type(exc).__name__}: {exc}"
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        if i % 25 == 0:
            print(f"{i}/{len(rows)}", flush=True)

print("wrote", OUT)
