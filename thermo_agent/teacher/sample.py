"""Frontier teacher sampler over any OpenAI-compatible endpoint.

The teacher only proposes; the tool suite decides. Every attempt is written to a raw log with its timestamp,
elapsed time and verifier verdict; accepted answers become SFT rows, and rejected ones become hindsight-memory
notes in the shared memory directory, so the next prompt (and the student) can read them. Nothing is retried
into the training set: if the verifiers reject a sample, the rejection is the useful output.

    python -m thermo_agent.teacher.sample data/processed data/frontier --task repair --samples 2 \
        --base-url https://... --model <frontier-model> --api-key-env TEACHER_API_KEY \
        --memory data/memory --extra-data data/processed_arch

`--dry-run` builds and measures the prompts without touching the network, so the harness/memory injection can be
checked on a machine with no endpoint.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

from ..agent import harness
from ..core.schema import Plant
from . import memory, validate
from .kaggle import normalize_base_url

CHAT_KW = {"enable_thinking": False}


class Teacher:
    """Minimal OpenAI-compatible chat client (no extra dependency)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float = 0.8,
                 max_tokens: int = 800, retries: int = 3, timeout: int = 180):
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens, "chat_template_kwargs": CHAT_KW}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read().decode())
                return out["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"teacher request failed after {self.retries} attempts: {last}")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _plant(task: str, row: dict) -> Plant | None:
    for key in ("plant_broken", "plant"):
        if row.get(key):
            return Plant.model_validate_json(row[key])
    return None


def build_messages(task: str, row: dict, mode: str = "agents+skills", notes: list[memory.Note] | None = None,
                   k_memory: int = 3) -> list[dict]:
    from ..agent.tool_prompt import build_harness_tool_messages
    plant = _plant(task, row)
    # One harness text for every arm: tracegen uses the same compact builder as
    # the *_h training corpora and eval/GRPO. mode="none" keeps the raw prompt.
    msgs = (list(row["prompt"]) if mode == "none"
            else build_harness_tool_messages(row["prompt"], task, plant, mode=mode))
    if notes:
        query = row.get("fault_type", "") + " " + (plant.name if plant else "")
        msgs = memory.augment_with_memory(msgs, task, query, notes, k=k_memory, exclude_record=str(row.get("id", "")))
    return msgs


def load_rows(data_dir, task: str, split: str = "train", limit: int | None = None, extra=()) -> list[dict]:
    rows = []
    for d in [data_dir, *extra]:
        p = pathlib.Path(d) / f"{task}_{split}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.open(encoding="utf-8")]
    return rows[:limit] if limit else rows


def _attempt_counts(raw_path: pathlib.Path) -> collections.Counter:
    """How many samples each row already has in the raw log, so a resume never re-spends quota."""
    counts = collections.Counter()
    if not raw_path.exists():
        return counts
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                counts[json.loads(line)["id"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def sample(rows: list[dict], teacher: Teacher, task: str, samples: int, out_dir, notes_out=None,
           mode: str = "agents+skills", notes: list[memory.Note] | None = None, k_memory: int = 3,
           split: str = "train", resume: bool = True) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted_path = out / f"frontier_{task}_{split}.jsonl"
    raw_path = out / f"frontier_{task}_{split}_raw.jsonl"
    runs_path = out / f"frontier_{task}_{split}_runs.jsonl"
    notes_dir = pathlib.Path(notes_out) if notes_out else out
    notes_path = notes_dir / f"memory_{task}.jsonl"
    done = set()
    if resume and accepted_path.exists():
        done = {json.loads(l)["id"].split("-t-")[0] for l in accepted_path.open(encoding="utf-8")}
    attempted = _attempt_counts(raw_path) if resume else collections.Counter()
    stats = collections.Counter()
    new_notes: list[memory.Note] = []
    started = _now()
    with raw_path.open("a", encoding="utf-8") as raw, accepted_path.open("a", encoding="utf-8") as acc:
        for row in rows:
            if row["id"] in done:
                stats["skipped"] += 1
                continue
            if attempted[row["id"]] >= samples:
                stats["skipped"] += 1
                continue
            for j in range(samples - attempted[row["id"]]):
                msgs = build_messages(task, row, mode, notes, k_memory)
                prompt_chars = sum(len(m.get("content", "")) for m in msgs)
                t0 = time.time()
                completion = teacher.chat(msgs)
                verdict = validate.validate(task, completion, row)
                raw.write(json.dumps({"id": row["id"], "sample": j, "ts": _now(),
                                      "elapsed_s": round(time.time() - t0, 2), "task": task, "mode": mode,
                                      "prompt_chars": prompt_chars, "model": teacher.model,
                                      "completion": completion, "accepted": verdict["accepted"],
                                      "reasons": verdict["reasons"], "reward": verdict["reward"]}) + "\n")
                stats["attempts"] += 1
                if verdict["accepted"]:
                    acc.write(json.dumps(validate.teacher_row(task, completion, row, verdict, tag=f"s{j}")) + "\n")
                    stats["accepted"] += 1
                else:
                    new_notes.append(validate.rejection_note(task, row, verdict, source=f"frontier:{teacher.model}"))
                    stats["rejected"] += 1
    if new_notes:
        memory.save(memory.dedupe(memory.load(notes_path) + new_notes), notes_path)
    stats["memory_notes"] = len(new_notes)
    for key in ("attempts", "accepted", "rejected", "skipped", "memory_notes"):
        stats[key] += 0
    result = dict(stats)
    result["acceptance_rate"] = round(stats["accepted"] / stats["attempts"], 3) if stats["attempts"] else None
    record = {"task": task, "split": split, "model": teacher.model, "mode": mode, "samples": samples,
              "started": started, "finished": _now(), **result}
    with runs_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--task", default="repair", choices=list(validate.TASKS))
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", default="agents+skills", choices=list(harness.MODES))
    ap.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL", ""))
    ap.add_argument("--api-key-env", default="TEACHER_API_KEY")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--memory", help="JSONL dir of hindsight notes to retrieve from (and to write rejections into)")
    ap.add_argument("--k-memory", type=int, default=3)
    ap.add_argument("--extra-data", nargs="*", default=[], help="more processed dirs, e.g. the architecture corpus")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = load_rows(a.data, a.task, a.split, a.limit, a.extra_data)
    notes = memory.load(a.memory) if a.memory else []
    if a.dry_run:
        sizes = []
        for row in rows[:10]:
            msgs = build_messages(a.task, row, a.mode, notes, a.k_memory)
            sizes.append(sum(len(m["content"]) for m in msgs))
        print(json.dumps({"rows": len(rows), "notes": len(notes), "prompt_chars": sizes,
                          "mean_chars": (sum(sizes) / len(sizes)) if sizes else 0}, indent=1))
        return
    if not a.base_url or not a.model:
        raise SystemExit("need --base-url and --model (or TEACHER_BASE_URL / TEACHER_MODEL)")
    t = Teacher(a.base_url, a.model, api_key=os.environ.get(a.api_key_env), temperature=a.temperature,
                max_tokens=a.max_tokens)
    print(json.dumps(sample(rows, t, a.task, a.samples, a.out, a.memory, a.mode, notes, a.k_memory, a.split),
                     indent=1))


if __name__ == "__main__":
    main()
