"""Run the frontier model with our physics tool exposed, through the Kaggle CLI proxy (no notebook needed).

    PYTHONIOENCODING=utf-8 python scripts/kaggle_harness_cli.py --limit 200

The answers, tool calls, reasoning traces and Kaggle usage are journaled to `data/teacher_runs/harness_completions.jsonl`
and mirrored to `teacher/harness_progress.jsonl` in the private HF repo. This is the third comparison arm: frontier
model with our harness/tools, scored locally by the same verifiers as every other arm.
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

from thermo_agent.teacher import cli_runner, progress, validate
from thermo_agent.teacher.transfer import load_exports
from thermo_agent.train import hf_sync

OUT = pathlib.Path("data/teacher_runs/harness_completions.jsonl")
PREFIX = "teacher/harness_progress.jsonl"


def system_prompt():
    return ("You are an expert thermodynamic agent. You may call check_answer(answer) at most TWICE to test a "
            "candidate answer against our physics verifiers. After at most two calls, STOP using tools and finish "
            "with your final answer in the task's XML tags. Never invent numbers.")


def usage_of(chat):
    try:
        u = chat.usage
        return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                "total_cost_nanodollars": u.total_cost_nanodollars,
                "total_backend_latency_ms": u.total_backend_latency_ms}
    except Exception:
        return {}


def trace_of(chat):
    traces, calls = [], []
    try:
        for m in chat.messages:
            if getattr(m, "reasoning_traces", None):
                traces.append(str(m.reasoning_traces))
            for c in getattr(m, "tool_calls", None) or []:
                calls.append({"name": getattr(c, "name", "?"), "arguments": getattr(c, "arguments", {})})
    except Exception:
        pass
    return "\n".join(traces), calls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--export-dir", default="data/frontier_export")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    ap.add_argument("--reserve", type=float, default=0.5)
    ap.add_argument("--max-seconds", type=float, default=4 * 3600)
    a = ap.parse_args()
    global OUT
    OUT = pathlib.Path(a.out)

    for k, v in cli_runner.dotenv(cli_runner.PROXY_ENV).items():
        os.environ[k] = v
    import kaggle_benchmarks as kbench

    from kaggle_benchmarks import actors
    from kaggle_benchmarks.tools.base import ToolInvocationLimitExhausted
    from kaggle_benchmarks.tools.native import native_tool_agent

    exports = load_exports(a.export_dir, split=a.split)
    rows = list(exports.values())
    done = {r["id"] for r in cli_runner.journal(OUT) if r.get("completion") and not r.get("error")}
    pending = [r for r in rows if r["id"] not in done][:a.limit]
    print(f"pending={len(pending)} done={len(done)} model={a.model}", flush=True)

    def make_tool(task, row):
        def check_answer(answer: str) -> str:
            """Check a candidate answer for THIS row against the physics verifiers; no id is needed."""
            verdict = validate.validate(task, answer, row)
            return json.dumps({"accepted": verdict["accepted"], "reward": verdict["reward"],
                               "reasons": verdict["reasons"]})
        return check_answer

    llm = kbench.llms.get(a.model, kbench.llm)
    started = time.time()
    written = 0
    for i, row in enumerate(pending):
        if time.time() - started > a.max_seconds:
            print("[pause] time limit", flush=True)
            break
        q = cli_runner.quota()
        if q.get("daily", {}).get("remaining", 0) < a.reserve:
            print("[pause] daily quota exhausted", flush=True)
            break
        rid = str(row["id"]); task = row.get("task", "repair")
        user = "\n\n".join(m["content"] for m in row["prompt"] if m["role"] != "system")
        rec = {"id": rid, "task": task, "model": a.model, "harness": "tools",
               "tools": ["check_answer"], "topology": row.get("topology"),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            if pathlib.Path(cli_runner.PROXY_ENV).stat().st_mtime < time.time() - 3000:
                for k, v in cli_runner.proxy_env(refresh=True).items():
                    os.environ[k] = v
            sysmsg = system_prompt() + f" The row id is {rid}. The tool checks this row only: call it with just your answer."
            with kbench.chats.new(f"harness_{i}", system_instructions=sysmsg) as chat:
                actors.user.send(user)                      # native_tool_agent needs a user turn in the chat
                try:
                    msg = native_tool_agent(llm, [make_tool(task, row)], max_tool_rounds=3)
                except ToolInvocationLimitExhausted:
                    # Some small models keep rechecking. Ask once, without tools, for the final answer.
                    actors.user.send("Stop using tools. Return your final answer now in the required XML tags.")
                    msg = llm.respond()
                    rec["tool_limit_exhausted"] = True
                rec["completion"] = str(msg.content)
                rec["usage"] = usage_of(chat)
                rec["reasoning"], rec["tool_calls"] = trace_of(chat)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        cli_runner.append(OUT, [rec]); written += 1
        if written % 25 == 0:
            try:
                token = hf_sync.read_token(cli_runner.ROOT / ".env")
                repo = hf_sync.PROJECT_REPO_ID
                merged = progress.dedupe(progress.pull(repo, token, PREFIX) + cli_runner.journal(OUT))
                progress.push(merged, repo, token, PREFIX)
                print(f"[progress] {len(merged)} harness attempts mirrored", flush=True)
            except Exception as exc:
                print("[progress] push failed:", exc, flush=True)
            cost = sum((r.get("usage") or {}).get("total_cost_nanodollars") or 0 for r in cli_runner.journal(OUT)) / 1e9
            print(f"[{i + 1}/{len(pending)}] written={written} cost=${cost:.4f}", flush=True)
    print("done; journal:", OUT, flush=True)


if __name__ == "__main__":
    main()
