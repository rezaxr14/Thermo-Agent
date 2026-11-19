#!/usr/bin/env python3
"""Real-tokenizer length audit of the harness-prompt corpora.

Builds on the measurement conventions already used in this repo (see
``scripts/dataset_stats.py``, ``thermo_agent/synth/combine.py`` and
``thermo_agent/train/token_audit.py``): the *local* cached Qwen3.5 tokenizer is
used, text is encoded with ``add_special_tokens=False``, and the training gate
adds ``TOKEN_RESERVE`` (96) for chat markers.  No model is downloaded unless
``--allow-download`` is passed explicitly.

Two measures are reported per row:

* ``rendered`` -- the task-requested measure: the full conversation
  (``row["prompt"] + row["completion"]``) rendered with
  ``apply_chat_template(..., add_generation_prompt=False)`` and then encoded.
  NOTE: this repo's Qwen3.5 chat template *drops* inline ``<tool_call>`` /
  ``<tool_result>`` blocks from an assistant message, so this measure can
  UNDER-count the real training sequence.  It is reported because it is the
  explicitly requested method.
* ``pipeline`` -- the gate actually used by ``build_corpus.ps1`` /
  ``token_audit.py`` for the 8192 limit: plain prompt text + plain completion
  text + ``TOKEN_RESERVE``.  This is the authoritative 8192 gate measure.

If neither tokenizer source is available, a char-length heuristic
(``ceil(chars / 3.5)``, the corpus' stored ``char_length`` convention) is used
and the report says so explicitly.

The corpora are only ever opened for reading.

    .venv/Scripts/python.exe scripts/audit_harness_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_stats as ds  # noqa: E402  (reuse quantiles / length_stats / _safe_output)
from thermo_agent.synth.combine import TOKENIZER_ENV, TOKEN_RESERVE, tokenizer_path  # noqa: E402

DEFAULT_DIRS = ("data/sft_tools_h", "data/grpo_tools_h")
DEFAULT_OUT = "data/provenance/harness_corpus_audit_2026-11-19.json"
GATE = 8192
MODEL_ID = "Qwen/Qwen3.5-4B"
CHAT_KW = {"enable_thinking": True}  # matches thermo_agent.train.render.CHAT_KW


# ---------------------------------------------------------------------------
# tokenizer loading: local cache first, never download unless asked
# ---------------------------------------------------------------------------
def load_chat_tokenizer(explicit: str | None, allow_download: bool):
    """Return (tok, info dict) or (None, info dict) to trigger the char fallback.

    ``tok`` is a ``transformers.AutoTokenizer`` (so it has ``apply_chat_template``)
    loaded from the locally cached tokenizer directory -- the same cache file
    that ``dataset_stats.py`` / ``combine.py`` use, just wrapped in the
    transformers interface so the chat template is available.
    """
    info: dict[str, Any] = {"name": MODEL_ID, "status": "not available",
                            "path": None, "method": None, "reason": None}
    candidates = [explicit, os.environ.get(TOKENIZER_ENV, ""),
                  str(tokenizer_path()) if tokenizer_path() else ""]
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        info["reason"] = f"transformers not installed: {exc}"
        return None, info

    local_dirs: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_dir():
            local_dirs.append(str(p.resolve()))
        elif p.is_file():
            local_dirs.append(str(p.parent.resolve()))
    # de-duplicate, keep order
    seen: set[str] = set()
    local_dirs = [d for d in local_dirs if not (d in seen or seen.add(d))]

    for d in local_dirs:
        try:
            tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        if getattr(tok, "chat_template", None):
            info.update(status="measured", path=d, method="local_auto_chat_template")
            return tok, info

    if allow_download:
        try:
            tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=False)
        except Exception as exc:  # noqa: BLE001 - network failure -> char fallback
            info["reason"] = f"hub download failed: {type(exc).__name__}: {exc}"
        else:
            info.update(status="measured", path=MODEL_ID, method="hub_auto_chat_template")
            return tok, info
    else:
        info["reason"] = ("no local chat-template tokenizer found; download not "
                          "attempted (pass --allow-download to try the hub)")
    return None, info


def encode_len(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def char_len(text: str) -> int:
    """Fallback token estimate: ~3.5 chars/token (the char-length heuristic)."""
    return int(math.ceil(len(text) / 3.5))


# ---------------------------------------------------------------------------
# core audit
# ---------------------------------------------------------------------------
def audit(root: Path, dirs: list[str], tok, gate: int, reserve: int) -> dict[str, Any]:
    files = ds.discover_files(root, dirs)
    status = "measured" if tok is not None else "char_length_heuristic"

    # per (dir, task) numeric collections
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"rendered": [], "pipeline": [], "char": [], "rows": 0}
    )
    over_rendered: list[dict[str, Any]] = []
    over_pipeline: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    generated_from: dict[str, Any] = {}

    for label, path in files:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        hasher = hashlib.sha256()
        file_rows = 0
        with path.open("rb") as fh:
            for raw in fh:
                hasher.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeDecodeError):
                    anomalies.append({"file": rel, "issue": "malformed_json"})
                    continue
                if not isinstance(row, dict):
                    anomalies.append({"file": rel, "issue": "row_not_object"})
                    continue
                file_rows += 1

                rid = row.get("id")
                task = ds.task_of(row, path)
                prompt = row.get("prompt")
                completion = row.get("completion")

                # structural checks requested by the caller
                if not isinstance(prompt, list):
                    anomalies.append({"file": rel, "id": rid, "issue": "prompt_not_list"})
                if not isinstance(completion, list):
                    anomalies.append({"file": rel, "id": rid, "issue": "completion_not_list"})
                if "prompt_variant" not in row:
                    anomalies.append({"file": rel, "id": rid, "issue": "missing_prompt_variant"})

                prompt_text = ds.text_of(prompt)
                completion_text = ds.text_of(completion)
                n_char = len(prompt_text) + len(completion_text)

                if tok is not None:
                    msgs = (prompt if isinstance(prompt, list) else []) + \
                           (completion if isinstance(completion, list) else [])
                    try:
                        rendered_text = tok.apply_chat_template(
                            msgs, add_generation_prompt=False, tokenize=False, **CHAT_KW
                        )
                        n_rendered = encode_len(tok, rendered_text)
                    except Exception as exc:  # noqa: BLE001
                        anomalies.append({"file": rel, "id": rid,
                                          "issue": "chat_render_failed",
                                          "detail": f"{type(exc).__name__}: {exc}"})
                        n_rendered = None
                    n_prompt = encode_len(tok, prompt_text)
                    n_completion = encode_len(tok, completion_text)
                    n_pipeline = n_prompt + n_completion + reserve
                else:
                    n_rendered = None
                    n_pipeline = char_len(prompt_text) + char_len(completion_text) + reserve

                b = buckets[(label, task)]
                b["rows"] += 1
                if n_rendered is not None:
                    b["rendered"].append(n_rendered)
                b["pipeline"].append(n_pipeline)
                b["char"].append(n_char)

                if n_rendered is not None and n_rendered > gate:
                    over_rendered.append({"id": rid, "task": task, "dir": label,
                                          "tokens": n_rendered})
                if n_pipeline > gate:
                    over_pipeline.append({"id": rid, "task": task, "dir": label,
                                          "tokens": n_pipeline})

        generated_from[rel] = {
            "sha256": hasher.hexdigest(),
            "rows": file_rows,
            "bytes": path.stat().st_size,
        }

    per_dir_task: dict[str, Any] = defaultdict(dict)
    for (label, task), b in buckets.items():
        per_dir_task[label][task] = {
            "n": b["rows"],
            "rendered": ds.length_stats(b["rendered"]) if b["rendered"] else None,
            "pipeline": ds.length_stats(b["pipeline"]),
            "chars": ds.length_stats(b["char"]),
        }

    # cross-corpus totals per task
    per_task: dict[str, Any] = {}
    task_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rendered": [], "pipeline": [], "char": [], "n": 0}
    )
    for (label, task), b in buckets.items():
        t = task_rows[task]
        t["n"] += b["rows"]
        t["rendered"].extend(b["rendered"])
        t["pipeline"].extend(b["pipeline"])
        t["char"].extend(b["char"])
    for task in sorted(task_rows):
        t = task_rows[task]
        per_task[task] = {
            "n": t["n"],
            "rendered": ds.length_stats(t["rendered"]) if t["rendered"] else None,
            "pipeline": ds.length_stats(t["pipeline"]),
            "chars": ds.length_stats(t["char"]),
        }

    all_rows = sum(b["rows"] for b in buckets.values())
    return {
        "schema_version": 1,
        "generated_by": "scripts/audit_harness_corpus.py",
        "note": ("counts and lengths only; no prompt/completion text. "
                 "rendered=chat-template(add_generation_prompt=False); "
                 "pipeline=plain prompt+completion+reserve (the build's 8192 gate)."),
        "gate": gate,
        "token_reserve": reserve,
        "tokenizer": status,
        "inputs": sorted({label for label, _ in files}),
        "generated_from": dict(sorted(generated_from.items())),
        "totals": {"rows": all_rows,
                   "malformed_or_anomalous": len(anomalies),
                   "over_8192_rendered": len(over_rendered),
                   "over_8192_pipeline": len(over_pipeline)},
        "per_dir_task": {d: dict(sorted(t.items())) for d, t in sorted(per_dir_task.items())},
        "per_task": dict(sorted(per_task.items())),
        "over_8192_rendered": sorted(over_rendered, key=lambda r: (-r["tokens"], str(r["id"]))),
        "over_8192_pipeline": sorted(over_pipeline, key=lambda r: (-r["tokens"], str(r["id"]))),
        "anomalies": anomalies,
    }


def print_report(stats: dict[str, Any], tok_info: dict[str, Any]) -> None:
    print(f"tokenizer: {tok_info['status']}  name={tok_info['name']}  "
          f"method={tok_info['method']}  path={tok_info['path']}")
    if tok_info.get("reason"):
        print(f"  reason: {tok_info['reason']}")
    print(f"gate: {stats['gate']}  reserve: {stats['token_reserve']}")
    hdr = (f"{'dir':>16} {'task':>8} {'n':>5} | "
           f"{'render p50':>10} {'p95':>7} {'max':>7} | "
           f"{'pipeline p50':>12} {'p95':>7} {'max':>7}")
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(stats["per_dir_task"]):
        for task, s in sorted(stats["per_dir_task"][d].items()):
            r = s["rendered"] or {"median": None, "p95": None, "max": None}
            p = s["pipeline"]
            print(f"{d:>16} {task:>8} {s['n']:>5} | "
                  f"{str(r['median']):>10} {str(r['p95']):>7} {str(r['max']):>7} | "
                  f"{str(p['median']):>12} {str(p['p95']):>7} {str(p['max']):>7}")
    print()
    print("per-task totals (both corpora):")
    for task, s in stats["per_task"].items():
        r = s["rendered"] or {"median": None, "p95": None, "max": None}
        p = s["pipeline"]
        print(f"  {task:>8} n={s['n']:>4}  rendered p50/p95/max="
              f"{r['median']}/{r['p95']}/{r['max']}  "
              f"pipeline p50/p95/max={p['median']}/{p['p95']}/{p['max']}")
    t = stats["totals"]
    print(f"\nTOTAL rows={t['rows']}  over8192(rendered)={t['over_8192_rendered']}  "
          f"over8192(pipeline)={t['over_8192_pipeline']}  anomalies={t['malformed_or_anomalous']}")
    if stats["over_8192_rendered"]:
        print("over-8192 (rendered) ids:")
        for r in stats["over_8192_rendered"]:
            print(f"  {r['id']}  {r['dir']}/{r['task']}  {r['tokens']} tokens")
    if stats["over_8192_pipeline"]:
        print("over-8192 (pipeline) ids:")
        for r in stats["over_8192_pipeline"]:
            print(f"  {r['id']}  {r['dir']}/{r['task']}  {r['tokens']} tokens")
    if stats["anomalies"]:
        print("anomalies:")
        for a in stats["anomalies"][:40]:
            print("  ", a)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="+", default=list(DEFAULT_DIRS))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--gate", type=int, default=GATE)
    ap.add_argument("--tokenizer", default=None,
                    help="explicit local chat-template tokenizer dir or tokenizer.json")
    ap.add_argument("--allow-download", action="store_true",
                    help="allow a hub download if no local tokenizer is found")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    out = ds._safe_output(ds._path(root, args.out))
    tok, tok_info = load_chat_tokenizer(args.tokenizer, args.allow_download)
    stats = audit(root, args.dirs, tok, args.gate, TOKEN_RESERVE)
    stats["tokenizer_info"] = tok_info

    payload = json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")

    print_report(stats, tok_info)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
