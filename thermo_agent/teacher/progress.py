"""Append-only teacher progress in the same private HF repo, so a Kaggle session can resume on another account.

The notebook writes one JSON line per attempt to `/kaggle/working/completions.jsonl`, which is lost when the session
ends. `sync()` merges that local file with `teacher/progress.jsonl` in the project repo and uploads the result after
each batch. Both successful answers and failures are kept (a failure records the quota/model error that caused it),
so the local ingest can grade everything and the failure memory sees the reasons.

No other Hugging Face repo is touched: the repo id passes through `train.hf_sync.resolve_repo`, which only accepts
the fixed project repo (or `<token owner>/thermo-agent-checkpoints`).
"""
from __future__ import annotations

import io
import json
import pathlib
import tempfile
import time

DEFAULT_PREFIX = "teacher/progress.jsonl"


def _key(record: dict) -> tuple:
    return (str(record.get("id", "")), str(record.get("completion", "")), str(record.get("error", "")))


def dedupe(records) -> list[dict]:
    """Keep every distinct attempt (success or failure), drop exact retries."""
    seen, out = set(), []
    for r in records:
        key = _key(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def answered_ids(records) -> set[str]:
    """Ids with at least one usable answer; errors alone do not count as answered."""
    return {str(r["id"]) for r in records if r.get("completion") and not r.get("error")}


def load(path) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def save(records, path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + ("\n" if records else ""), encoding="utf-8")


def _api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def pull(repo_id: str, token: str, prefix: str = DEFAULT_PREFIX, api=None) -> list[dict]:
    """Download the shared progress file; a missing file is an empty history, not an error."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    api = api or _api(token)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = hf_hub_download(repo_id=repo_id, filename=prefix, token=token, local_dir=tmp)
            return load(path)
    except EntryNotFoundError:
        return []


def push(records, repo_id: str, token: str, prefix: str = DEFAULT_PREFIX, api=None) -> str:
    api = api or _api(token)
    payload = ("\n".join(json.dumps(r, sort_keys=True) for r in records) + ("\n" if records else "")).encode()
    api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=prefix, repo_id=repo_id, token=token,
                    commit_message=f"teacher progress: {len(records)} attempts")
    return prefix


def sync(local_path, repo_id: str, token: str, prefix: str = DEFAULT_PREFIX, api=None) -> list[dict]:
    """Merge local attempts with the shared history, write them back locally and upload the merge."""
    local = load(local_path)
    remote = pull(repo_id, token, prefix, api)
    merged = dedupe(remote + local)
    save(merged, local_path)
    push(merged, repo_id, token, prefix, api)
    return merged


def usage_totals(records) -> dict:
    """Sum the Kaggle Benchmarks Usage fields recorded on each attempt, in nanodollars."""
    out = {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "cost_nanodollars": 0,
           "latency_ms": 0, "models": {}}
    for r in records:
        u = r.get("usage") or {}
        out["attempts"] += 1
        for k, dest in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                        ("total_cost_nanodollars", "cost_nanodollars"), ("total_backend_latency_ms", "latency_ms")):
            out[dest] += int(u.get(k) or 0)
        model = str(r.get("model") or "unknown")
        out["models"][model] = out["models"].get(model, 0) + 1
    out["cost_usd"] = round(out["cost_nanodollars"] / 1e9, 6)
    return out
