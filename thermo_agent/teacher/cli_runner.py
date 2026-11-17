"""Quota-aware frontier-teacher runner for Kaggle's interactive Model Proxy (CLI token, no browser).

    python -m thermo_agent.teacher.cli_runner probe
    python -m thermo_agent.teacher.cli_runner run --model google/gemini-3.1-flash-lite-preview \
        --out data/teacher_runs/completions.jsonl --all-tasks

It reads `KAGGLE_API_TOKEN` from `.env`, refreshes the proxy credential when it expires, checks the daily and
monthly quota between batches, journals every answered and failed attempt, and mirrors the journal to the project's
private HF repo (`teacher/progress.jsonl`) so another account or session can resume. No ground-truth completion is
ever sent or stored here: the local ingest step grades the answers against the physics tool suite.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
PROXY_ENV = ROOT / "data" / "kaggle_proxy.env"
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"


def dotenv(path=ENV_FILE) -> dict:
    out = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def refresh_proxy() -> dict:
    env = dotenv()
    token = env.get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("KAGGLE_API_TOKEN is missing from .env")
    PROXY_ENV.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(KAGGLE), "benchmarks", "init", "-y", "--env-file", str(PROXY_ENV)],
                   check=True, capture_output=True, text=True,
                   env={**os.environ, "KAGGLE_API_TOKEN": token}, cwd=ROOT)
    return dotenv(PROXY_ENV)


def proxy_env(refresh=False) -> dict:
    if refresh or not PROXY_ENV.exists():
        return refresh_proxy()
    env = dotenv(PROXY_ENV)
    expiry = env.get("MODEL_PROXY_EXPIRY_TIME", "")
    try:
        import datetime
        exp = datetime.datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        left = (exp - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        if left < 300:
            return refresh_proxy()
    except Exception:
        pass
    if not env.get("MODEL_PROXY_URL") or not env.get("MODEL_PROXY_API_KEY"):
        return refresh_proxy()
    return env


def quota() -> dict:
    env = dotenv()
    out = subprocess.run([str(KAGGLE), "benchmarks", "quota", "--format", "json"],
                         capture_output=True, text=True, env={**os.environ, "KAGGLE_API_TOKEN": env["KAGGLE_API_TOKEN"]}, cwd=ROOT)
    if out.returncode:
        raise RuntimeError(out.stderr[-500:])
    rows = json.loads(out.stdout)
    def money(s):
        try:
            return float(str(s).replace("$", "").replace(",", ""))
        except Exception:
            return 0.0
    return {r.get("period", "?").lower(): {"used": money(r.get("used")), "remaining": money(r.get("remaining")),
                                          "total": money(r.get("total"))} for r in rows}


def _usage(raw: dict) -> dict:
    u = raw.get("usage") or {}
    cost = u.get("cost") or {}
    return {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
            "total_cost_nanodollars": (cost.get("input_tokens_cost_nanodollars") or 0)
                                      + (cost.get("output_tokens_cost_nanodollars") or 0)
                                      if cost else None,
            "total_backend_latency_ms": u.get("total_backend_latency_ms"),
            "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason")}


def chat(model: str, messages: list[dict], max_tokens=800, env=None, timeout=300) -> dict:
    env = env or proxy_env()
    url = env["MODEL_PROXY_URL"].rstrip("/")
    if not url.endswith("/openapi"):
        url += "/openapi"
    url += "/chat/completions"
    payloads = [
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        {"model": model, "messages": messages, "max_tokens": max_tokens},
        {"model": model, "messages": messages, "max_completion_tokens": max_tokens},
    ]
    last = None
    for payload in payloads:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + env["MODEL_PROXY_API_KEY"]})
        try:
            started = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            usage = _usage(data)
            usage["seconds"] = round(time.time() - started, 2)
            return {"completion": str(text), "usage": usage, "params": list(payload), "model": model}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:400]
            last = f"HTTP {exc.code}: {body}"
            if exc.code not in (400, 422):
                return {"error": last, "model": model}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "model": model}
    return {"error": last or "all parameter variants failed", "model": model}


def load_rows(export_dir="data/frontier_export", tasks=("repair", "ledger", "audit", "design"), split="train") -> list[dict]:
    rows = []
    for task in tasks:
        path = pathlib.Path(export_dir) / f"{task}_{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line); r["task"] = task; rows.append(r)
    return rows


def journal(path) -> list[dict]:
    p = pathlib.Path(path)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def append(path, records):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _one(row, model, max_tokens):
    messages = row.get("prompt") or []
    out = chat(model, messages, max_tokens)
    return {"id": row["id"], "task": row.get("task"), "topology": row.get("topology"),
            "harness": row.get("harness", "none"), "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **out}


def push_progress(path, max_bytes=None):
    """Mirror the local journal into the shared private repo; failures are logged, never fatal."""
    try:
        from . import progress
        from ..train import hf_sync
        token = hf_sync.read_token(ROOT / ".env")
        repo = hf_sync.PROJECT_REPO_ID
        records = journal(path)
        remote = progress.pull(repo, token, "teacher/progress.jsonl")
        merged = progress.dedupe(remote + records)   # keep attempts from every model and every account
        progress.push(merged, repo, token, "teacher/progress.jsonl")
        return len(merged)
    except Exception as exc:
        print("[progress] push failed:", type(exc).__name__, exc, flush=True)
        return 0


def probe(models):
    env = proxy_env()
    prompt = [{"role": "user", "content": "Reply with exactly: READY"}]
    for model in models:
        out = chat(model, prompt, max_tokens=48, env=env)
        print(json.dumps({"model": model, "completion": out.get("completion", "")[:60],
                          "error": out.get("error"), "usage": out.get("usage")}, sort_keys=True), flush=True)


def run(args):
    env = proxy_env()
    rows = load_rows(args.export_dir, tuple(args.tasks), args.split)
    done = {r["id"] for r in journal(args.out) if r.get("completion") and not r.get("error")}
    pending = [r for r in rows if r["id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"rows={len(rows)} pending={len(pending)} done={len(done)} model={args.model}", flush=True)
    if not pending:
        return
    started = time.time()
    for i in range(0, len(pending), args.batch):
        batch = pending[i:i + args.batch]
        q = quota()
        spend = sum(v["used"] for k, v in q.items() if k != "monthly")
        if q.get("daily", {}).get("remaining", 0) < args.reserve or spend >= args.max_spend:
            print(f"[pause] quota daily={q.get('daily')} spend={spend:.2f} max={args.max_spend}; sleeping", flush=True)
            if not args.wait:
                break
            time.sleep(300)
            env = proxy_env(refresh=True)
            continue
        if time.time() - started > args.max_seconds:
            print("[pause] time limit reached", flush=True)
            break
        t = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda r: _one(r, args.model, args.max_tokens), batch))
        append(args.out, results)
        if (i // args.batch) % args.push_every == 0:
            n = push_progress(args.out)
            print(f"[progress] {n} journal records in the shared repo", flush=True)
        ok = sum(bool(r.get("completion")) for r in results)
        cost = sum((r.get("usage") or {}).get("total_cost_nanodollars") or 0 for r in results) / 1e9
        print(f"[{i + len(batch)}/{len(pending)}] ok={ok} batch={time.time() - t:.1f}s batch_cost=${cost:.4f}", flush=True)
        if int(time.time() - started) % 1800 < args.batch:
            env = proxy_env(refresh=True)
    print("done; journal:", args.out, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("probe", "run", "quota", "sync"))
    ap.add_argument("--model", default="google/gemini-3.1-flash-lite-preview")
    ap.add_argument("--out", default="data/teacher_runs/completions.jsonl")
    ap.add_argument("--export-dir", default="data/frontier_export")
    ap.add_argument("--split", default="train", help="export split: train, test_id, test_ood_fault, ...")
    ap.add_argument("--tasks", nargs="+", default=["repair", "ledger", "audit", "design"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--reserve", type=float, default=0.5)
    ap.add_argument("--max-spend", type=float, default=9.0)
    ap.add_argument("--max-seconds", type=float, default=6 * 3600)
    ap.add_argument("--push-every", type=int, default=3)
    ap.add_argument("--wait", action="store_true", help="sleep and retry when the daily quota is exhausted")
    a = ap.parse_args()
    if a.action == "probe":
        env = dotenv(PROXY_ENV)
        models = [m.strip() for m in env.get("LLMS_AVAILABLE", "").split(",") if m.strip()]
        probe(models)
    elif a.action == "quota":
        print(json.dumps(quota(), indent=1))
    elif a.action == "sync":
        print("pushed", push_progress(a.out), "records")
    else:
        run(a)


if __name__ == "__main__":
    main()
