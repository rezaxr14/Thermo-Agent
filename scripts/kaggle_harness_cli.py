"""Frontier CoT data generation: LangGraph harness with REAL tools + skills + fresh memory.

    PYTHONIOENCODING=utf-8 python scripts/kaggle_harness_cli.py --limit 200

For each row the model sees AGENTS.md + component skills + freshly retrieved failure memory,
proposes an answer, calls the real thermodynamics tools (first/second law, validity, solve),
reads the numeric feedback, self-corrects, and finalizes. The full trajectory (reasoning,
tool calls, numeric results, revisions) is journaled; on verifier acceptance it becomes the
SFT completion so the student learns to use tools/skills/memory and self-correct.

No ground truth is shown to the model. Quota is checked before every call.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from thermo_agent.teacher import cli_runner, progress
from thermo_agent.train import hf_sync
from thermo_agent.teacher import memory as mem
from thermo_agent.teacher.agent_graph import run as run_graph, build_context
from thermo_agent.teacher.kbench_adapter import make_respond, usage_of
from thermo_agent.teacher.toolkit import tool_names
from thermo_agent.teacher.transfer import load_exports

OUT = pathlib.Path("data/teacher_runs/harness_completions.jsonl")
PREFIX = "teacher/harness_progress.jsonl"
MEMORY_DIR = pathlib.Path("data/memory")


def fresh_notes():
    """Load failure memory fresh from disk so it is always current."""
    notes = []
    for p in MEMORY_DIR.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            try:
                notes.append(mem.Note(**json.loads(line)))
            except Exception:
                pass
    return mem.dedupe(notes)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--export-dir", default="data/frontier_export")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--reserve", type=float, default=0.5)
    ap.add_argument("--max-seconds", type=float, default=4 * 3600)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k-memory", type=int, default=3)
    a = ap.parse_args()
    OUT = pathlib.Path(a.out)

    for k, v in cli_runner.dotenv(cli_runner.PROXY_ENV).items():
        os.environ[k] = v
    import kaggle_benchmarks as kbench
    from kaggle_benchmarks import actors

    exports = load_exports(a.export_dir, split=a.split)
    rows = list(exports.values())
    done = {r["id"] for r in cli_runner.journal(OUT) if r.get("completion") and not r.get("error")}
    pending = [r for r in rows if r["id"] not in done][:a.limit]

    llm = kbench.llms.get(a.model, kbench.llm)
    print(f"pending={len(pending)} done={len(done)} model={a.model} tools={tool_names()}", flush=True)
    started, written = time.time(), 0
    for i, row in enumerate(pending):
        if time.time() - started > a.max_seconds:
            print("[pause] time limit", flush=True); break
        q = cli_runner.quota()
        if q.get("daily", {}).get("remaining", 0) < a.reserve:
            print("[pause] daily quota exhausted", flush=True); break
        rid, task = str(row["id"]), row.get("task", "repair")
        rec = {"id": rid, "task": task, "model": a.model, "harness": "tools",
               "tools": tool_names(), "topology": row.get("topology"),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            if pathlib.Path(cli_runner.PROXY_ENV).stat().st_mtime < time.time() - 3000:
                for k, v in cli_runner.proxy_env(refresh=True).items():
                    os.environ[k] = v
            # Build the full system prompt once so the adapter can deliver it to the LLM.
            ctx = build_context({"row": row, "task": task, "notes": fresh_notes(),
                                 "k_memory": a.k_memory, "mode": "agents+skills"})
            usage_holder = {}
            respond = make_respond(llm, kbench, actors, system_prompt=ctx["system_prompt"],
                                   max_tool_rounds=a.rounds, usage_holder=usage_holder)
            state = run_graph(row, task, respond, notes=fresh_notes(),
                              max_rounds=a.rounds, k_memory=a.k_memory)
            rec["completion"] = state["final"]
            rec["verdict"] = {k: v for k, v in state["verdict"].items() if k != "failed"}
            rec["trajectory"] = {"turns": state["turns"], "notes_used": state["notes_used"],
                                 "tools_available": tool_names()}
            rec["sft_completion"] = state["sft_completion"]
            rec["usage"] = usage_of(usage_holder["chat"]) if "chat" in usage_holder else {}
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        cli_runner.append(OUT, [rec]); written += 1
        if written % 25 == 0:
            try:
                token = hf_sync.read_token(cli_runner.ROOT / ".env")
                merged = progress.dedupe(progress.pull(hf_sync.PROJECT_REPO_ID, token, PREFIX) + cli_runner.journal(OUT))
                progress.push(merged, hf_sync.PROJECT_REPO_ID, token, PREFIX)
                print(f"[progress] {len(merged)} harness attempts mirrored", flush=True)
            except Exception as exc:
                print("[progress] push failed:", exc, flush=True)
    print(f"wrote {written} rows to {OUT}")


if __name__ == "__main__":
    main()
