"""Hosted teacher pass over one compact prompt shard (big models, additive journal).

No secrets, no ground truth. Built from kaggle_benchmarks 0.6.3 API. The journal is
flushed per row so a dead worker loses at most one answer; re-running skips answered ids.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import kaggle_benchmarks as kbench

SPEC = {
    "task_name": "thermo-teacher-big-s00",
    "dataset": "rezanadimi/thermo-teacher-prompts-big",
    "shard": "prompts_s00.jsonl",
    "limit": 200,
    # Reasoning ON: the user wants reasoning traces where the provider exposes them; where
    # they cannot be captured, leaving reasoning enabled is acceptable. Reasoning also avoids
    # the Anthropic `thinking:{type:disabled}` 400 that reasoning="none" triggered on Claude.
    "max_output_tokens": 1536,   # headroom for a reasoning block plus the answer
    "reasoning": "low",
    "deadline_seconds": 7200,
    "max_cost_usd": 1.6,
}
SPEC_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:16]
OUT = pathlib.Path("/kaggle/working")
JOURNAL = OUT / "completions.jsonl"
MANIFEST = OUT / "manifest.json"
MODEL_SLUG = "unknown"
SPEND = 0
COUNTS = {"attempted": 0, "answered": 0, "error": 0}
ERROR = None


def _manifest(status: str, stage: str, **kw):
    MANIFEST.write_text(json.dumps({
        "status": status, "stage": stage, "error": ERROR,
        "task": SPEC["task_name"], "model": MODEL_SLUG, "spec_sha": SPEC_SHA,
        "counts": COUNTS, "spend_usd": round(SPEND / 1e9, 6), **kw,
    }, indent=1), encoding="utf-8")


def _dataset_root() -> pathlib.Path:
    slug = SPEC["dataset"].split("/", 1)[-1]
    candidates = [pathlib.Path("/kaggle/input") / slug]
    candidates += [p.parent for p in pathlib.Path("/kaggle/input").rglob(SPEC["shard"])]
    candidates += [pathlib.Path(".")]
    for candidate in candidates:
        if (candidate / SPEC["shard"]).exists():
            return candidate
    raise SystemExit(f"shard {SPEC['shard']} not found; attach the dataset via tasks push -d")


def _kind(message: str) -> str:
    text = message.lower()
    if any(x in text for x in ("quota", "billing", "daily limit", "monthly limit")):
        return "quota"
    if any(x in text for x in ("401", "403", "unauthorized", "api key", "expired")):
        return "auth"
    if "429" in text or ("rate" in text and "limit" in text):
        return "rate_limit"
    return "other"


def _invoke(llm, user: str, system: str, params: dict):
    with kbench.chats.new("row", system_instructions=system or None) as chat:
        completion = str(llm.prompt(user, reasoning=SPEC["reasoning"], extra_api_params=params))
        usage = chat.usage
        traces = [m.reasoning_traces for m in chat.messages
                  if getattr(m, "reasoning_traces", None)]
        return completion, {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_cost_nanodollars": getattr(usage, "total_cost_nanodollars", None),
            "total_backend_latency_ms": getattr(usage, "total_backend_latency_ms", None),
            "finish_reason": None,
        }, ("\n".join(str(t) for t in traces) or None)


@kbench.task(name=SPEC["task_name"], description="Thermo teacher: big-model CoT over a prompt shard")
def thermo_teacher_big_s00(llm):
    global MODEL_SLUG, SPEND, ERROR
    MODEL_SLUG = getattr(llm, "model", None) or "unknown"
    COUNTS.update(attempted=0, answered=0, error=0)
    SPEND, ERROR = 0, None
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _manifest("running", "start")
    root = _dataset_root()
    rows = [json.loads(line) for line in (root / SPEC["shard"]).open(encoding="utf-8") if line.strip()]
    if SPEC["limit"]:
        rows = rows[:SPEC["limit"]]
    done = {json.loads(line)["id"] for line in JOURNAL.open(encoding="utf-8") if line.strip()} \
        if JOURNAL.exists() else set()
    consecutive = 0
    with JOURNAL.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            if time.time() - started > SPEC["deadline_seconds"]:
                ERROR = "deadline"
                break
            if SPEND / 1e9 >= SPEC["max_cost_usd"]:
                ERROR = "cost_cap"
                break
            record = {
                "id": row["id"], "task": row.get("task"), "model": MODEL_SLUG,
                "topology": row.get("topology"), "harness": row.get("harness", "none"),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tools": "none", "shard": SPEC["shard"], "spec_sha": SPEC_SHA,
                "dataset": SPEC["dataset"], "prompt_sha256": row.get("prompt_sha256"),
                "decoding": {"temperature": None, "reasoning": SPEC["reasoning"],
                             "max_output_tokens": SPEC["max_output_tokens"]},
            }
            t0 = time.time()
            try:
                try:
                    completion, usage, traces = _invoke(
                        llm, row["user"], row.get("system", ""), {"max_tokens": SPEC["max_output_tokens"]})
                except Exception:
                    completion, usage, traces = _invoke(
                        llm, row["user"], row.get("system", ""),
                        {"max_completion_tokens": SPEC["max_output_tokens"]})
                record.update(completion=completion, usage=usage)
                if traces:
                    record["reasoning"] = traces
                SPEND += int(usage["total_cost_nanodollars"] or 0)
                record["usage"]["seconds"] = round(time.time() - t0, 2)
                COUNTS["answered"] += 1
                consecutive = 0
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                record.update(error=message[:500], error_kind=_kind(message), usage={})
                COUNTS["error"] += 1
                consecutive += 1
                ERROR = message[:500]
            COUNTS["attempted"] += 1
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            if COUNTS["attempted"] % 10 == 0:
                _manifest("running", "rows", next_index=i + 1)
            kind = record.get("error_kind")
            if kind in ("quota", "auth"):
                _manifest("paused", "error", error=record["error"])
                break
            if kind == "rate_limit":
                time.sleep(60)
                if consecutive >= 2:
                    _manifest("paused", "rate_limit")
                    break
            if consecutive >= 5:
                _manifest("paused", "too_many_errors")
                break
    _manifest("completed" if ERROR in (None, "cost_cap", "deadline") else "partial", "done")


thermo_teacher_big_s00.run(kbench.llm)
