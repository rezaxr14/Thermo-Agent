#!/usr/bin/env python3
"""Scrub unusable ``call_id`` blobs from frontier tool traces (additive-only).

Real harness completions embed tool results like::

    <tool_result>ToolInvocationResult(name='inspect_plant', arguments={},
      call_id='call_...__mpthought__<hundreds of base64 chars>',
      output='{"ok": true, ...}', error=None)</tool_result>

The ``call_id`` is random, huge and untrainable (one trace is ~60k chars / ~15k
tokens).  This script rewrites ONLY the completion text: every wrapped
``<tool_result>`` that carries a ``call_id`` becomes::

    <tool_result>{"tool": "<name>", ...payload}</tool_result>

The tool name and the ``output=`` JSON payload survive; ``call_id=`` and the
``ToolInvocationResult(...)`` wrapper are dropped.  Blocks without a ``call_id``
(and everything outside the blobs, including ids and the final answer) are left
byte-for-byte unchanged.  The same blobs also occur as bare strings in
``trajectory.turns[].tool_results``; those are scrubbed too so the output dir has
zero ``call_id='`` occurrences.  A NEW output directory is written; an existing
non-empty one is refused.  The run fails loudly if any blob survives.

    .venv/Scripts/python.exe scripts/clean_frontier_traces.py --dry-run
    .venv/Scripts/python.exe scripts/clean_frontier_traces.py \
        --input data/frontier_tools --out data/frontier_tools_clean
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import re
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.synth import combine  # noqa: E402

DEFAULT_INPUT = "data/frontier_tools"
DEFAULT_OUTPUT = "data/frontier_tools_clean"

BLOCK_RE = re.compile(r"<tool_result>(.*?)</tool_result>", re.S)
_NAME_RE = re.compile(r"\bname='([^']*)'")
_OUTPUT_RE = re.compile(r"\boutput='(.*)',\s*error=", re.S)


def _parse(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(m.get("content", "")) for m in value if isinstance(m, dict))
    return str(value or "")


def _decode_payload(raw: str) -> tuple[dict | None, str]:
    """Decode an ``output=`` body into ``(dict_payload, decoded_text)``.

    Normally the body is literal JSON (the 127 clean blocks); that path is kept
    first and unchanged so their output stays byte-identical.  In at least one real
    row the body is the inside of a Python *repr* string -- it contains ``\\'`` and
    ``\\\\"`` escapes.  Re-wrapping it in quotes and using ``ast.literal_eval``
    recovers the original JSON text, which ``json.loads`` then parses.  If even
    that fails we still return the decoded text so the block can be rewritten
    without leaving a ``call_id`` behind.
    """
    # 1) literal JSON -- preserve the existing 127-block output exactly.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, raw
    except (ValueError, TypeError):
        pass
    # 2) Python repr string -> literal_eval -> json.loads.
    for quote in ("'", '"'):
        try:
            decoded = ast.literal_eval(quote + raw + quote)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(decoded, dict):
            return decoded, decoded
        if isinstance(decoded, str):
            try:
                obj = json.loads(decoded)
                if isinstance(obj, dict):
                    return obj, decoded
            except (ValueError, TypeError):
                pass
            return None, decoded
    return None, raw


def _rewrite_inner(inner: str) -> str | None:
    """New block body, or ``None`` when the block must be left unchanged.

    A wrapped block carrying ``call_id=`` is ALWAYS rewritten -- a payload that
    cannot be parsed is kept verbatim under an ``output`` key rather than silently
    leaving the unusable blob in the corpus.
    """
    if "ToolInvocationResult" not in inner or "call_id=" not in inner:
        return None
    m_name = _NAME_RE.search(inner)
    m_out = _OUTPUT_RE.search(inner)
    if not m_name:
        return None
    name = m_name.group(1)
    raw = m_out.group(1) if m_out else ""
    payload, decoded = _decode_payload(raw)
    if isinstance(payload, dict):
        obj = {"tool": name, **payload}
    else:
        obj = {"tool": name, "output": decoded}
    return json.dumps(obj, ensure_ascii=False)


def rewrite_completion(text: str) -> tuple[str, int]:
    """Rewrite every wrapped tool_result that has a call_id. Returns (text, count)."""
    count = [0]

    def repl(match: re.Match) -> str:
        new_inner = _rewrite_inner(match.group(1))
        if new_inner is None:
            return match.group(0)
        count[0] += 1
        return "<tool_result>" + new_inner + "</tool_result>"

    return BLOCK_RE.sub(repl, text), count[0]


def clean_line(raw: str) -> tuple[str, int]:
    """Clean one JSONL line. Returns (new_line, rewritten_block_count).

    Two kinds of blob are scrubbed, with the same payload-preserving parser:

    * wrapped ``<tool_result>`` blocks inside ``completion`` text;
    * bare ``ToolInvocationResult(...)`` strings in
      ``trajectory.turns[].tool_results`` (they carry the same call_id blobs).

    Everything else -- ids, plants, prompt, final answer, other trajectory fields --
    is preserved byte-for-byte by splicing the newly encoded strings back into the
    original raw line.  Only if a splice cannot be located do we re-dump the row.
    """
    try:
        row = json.loads(raw)
    except (ValueError, TypeError):
        return raw, 0
    if not isinstance(row, dict):
        return raw, 0

    pairs: list[tuple[str, str]] = []
    completion_updates: dict[int, str] = {}
    trajectory_updates: dict[tuple[int, int], str] = {}
    total = 0

    completion = row.get("completion")
    if isinstance(completion, list):
        for idx, msg in enumerate(completion):
            if not isinstance(msg, dict) or not isinstance(msg.get("content"), str):
                continue
            text = msg["content"]
            if "<tool_result>" not in text:
                continue
            new, count = rewrite_completion(text)
            if count:
                pairs.append((text, new))
                completion_updates[idx] = new
                total += count

    traj = row.get("trajectory")
    if isinstance(traj, dict) and isinstance(traj.get("turns"), list):
        for ti, turn in enumerate(traj["turns"]):
            if not isinstance(turn, dict) or not isinstance(turn.get("tool_results"), list):
                continue
            for ri, value in enumerate(turn["tool_results"]):
                if not isinstance(value, str) or "call_id=" not in value:
                    continue
                new = _rewrite_inner(value)
                if new is not None:
                    pairs.append((value, new))
                    trajectory_updates[(ti, ri)] = new
                    total += 1

    if not total:
        return raw, 0

    line = raw
    spliced = True
    for old, new in pairs:
        old_enc = json.dumps(old, ensure_ascii=False)
        new_enc = json.dumps(new, ensure_ascii=False)
        if old_enc in line:
            line = line.replace(old_enc, new_enc, 1)
        else:
            spliced = False
            break
    if spliced:
        return line, total

    # Fallback: re-dump the row with only the blobs changed.
    for idx, new in completion_updates.items():
        completion[idx] = dict(completion[idx], content=new)
    for (ti, ri), new in trajectory_updates.items():
        traj["turns"][ti]["tool_results"][ri] = new
    eol = "\r\n" if raw.endswith("\r\n") else ("\n" if raw.endswith("\n") else "")
    return json.dumps(row, ensure_ascii=False) + eol, total


def _lengths(row: dict, tok) -> tuple[int, int]:
    """(chars, tokens) for one row, prompt + completion, repo-standard measure."""
    prompt, comp = row.get("prompt"), row.get("completion")
    p_text, c_text = _parse(prompt), _parse(comp)
    chars = len(p_text) + len(c_text)
    if tok is not None:
        n = sum(len(tok.encode(str(m.get("content", "")), add_special_tokens=False).ids)
                for m in (prompt if isinstance(prompt, list) else [])
                if isinstance(m, dict))
        n += sum(len(tok.encode(str(m.get("content", "")), add_special_tokens=False).ids)
                 for m in (comp if isinstance(comp, list) else [])
                 if isinstance(m, dict))
        tokens = n + combine.TOKEN_RESERVE
    else:
        tokens = math.ceil(chars / 3.5)
    return chars, tokens


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]


def _stats(values: list[int]) -> dict:
    return {"n": len(values), "p50": _pct(values, 0.5),
            "p95": _pct(values, 0.95), "max": max(values) if values else None}


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path)
    return path if path.is_absolute() else ROOT / path


def clean_dir(input_dir: pathlib.Path | str, out_dir: pathlib.Path | str,
              dry_run: bool = False, manifest_name: str = "MANIFEST.json") -> dict:
    """Read every ``*.jsonl`` in ``input_dir`` and write a cleaned copy to ``out_dir``."""
    src, dest = _resolve(input_dir), _resolve(out_dir)
    if not src.is_dir():
        raise SystemExit(f"input dir does not exist: {src}")
    if dest.resolve() == src.resolve() or dest.resolve().is_relative_to(src.resolve()):
        raise SystemExit(f"refusing to write inside the input dir: {dest}")
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output dir: {dest}")

    files = sorted(p for p in src.glob("*.jsonl") if p.is_file())
    tok = combine.tokenizer()
    before_chars: list[int] = []
    before_tokens: list[int] = []
    after_chars: list[int] = []
    after_tokens: list[int] = []
    total_blocks = 0
    rows_changed = 0
    leftover_call_ids = 0
    per_file: dict[str, dict] = {}

    for path in files:
        with path.open(encoding="utf-8", newline="") as fh:
            raw_lines = fh.read().splitlines(keepends=True)
        f_blocks = 0
        f_changed = 0
        f_leftover = 0
        f_before_c: list[int] = []
        f_before_t: list[int] = []
        f_after_c: list[int] = []
        f_after_t: list[int] = []
        cleaned: list[str] = []
        for raw in raw_lines:
            if not raw.strip():
                cleaned.append(raw)
                continue
            new_line, count = clean_line(raw)
            f_blocks += count
            f_changed += bool(count)
            f_leftover += new_line.count("call_id='")
            try:
                old_row, new_row = json.loads(raw), json.loads(new_line)
                c0, t0 = _lengths(old_row, tok)
                c1, t1 = _lengths(new_row, tok)
            except (ValueError, TypeError):
                c0 = t0 = c1 = t1 = 0
            f_before_c.append(c0); f_before_t.append(t0)
            f_after_c.append(c1); f_after_t.append(t1)
            cleaned.append(new_line)

        before_chars += f_before_c; before_tokens += f_before_t
        after_chars += f_after_c; after_tokens += f_after_t
        total_blocks += f_blocks
        rows_changed += f_changed
        leftover_call_ids += f_leftover
        per_file[path.name] = {
            "rows": len(f_before_c), "rewritten_blocks": f_blocks, "rows_changed": f_changed,
            "leftover_call_ids": f_leftover,
            "before": {"chars": _stats(f_before_c), "tokens": _stats(f_before_t)},
            "after": {"chars": _stats(f_after_c), "tokens": _stats(f_after_t)},
        }
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / path.name).open("w", encoding="utf-8", newline="") as fh:
                fh.writelines(cleaned)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/clean_frontier_traces.py",
        "input": str(src.relative_to(ROOT)).replace("\\", "/") if src.is_relative_to(ROOT) else str(src),
        "output": str(dest.relative_to(ROOT)).replace("\\", "/") if dest.is_relative_to(ROOT) else str(dest),
        "dry_run": bool(dry_run),
        "tokenizer": "measured" if tok is not None else "char_length_heuristic",
        "files": len(files),
        "rows": len(before_chars),
        "rows_changed": rows_changed,
        "rewritten_blocks": total_blocks,
        "leftover_call_ids": leftover_call_ids,
        "before": {"chars": _stats(before_chars), "tokens": _stats(before_tokens)},
        "after": {"chars": _stats(after_chars), "tokens": _stats(after_tokens)},
        "per_file": per_file,
    }
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
    if leftover_call_ids:
        raise SystemExit(
            f"refusing a silent partial clean: {leftover_call_ids} 'call_id=' occurrences "
            f"remain in the output (see {dest / manifest_name})")
    return manifest


def _print_report(manifest: dict) -> None:
    b, a = manifest["before"], manifest["after"]
    print(f"input={manifest['input']}  output={manifest['output']}  "
          f"dry_run={manifest['dry_run']}  tokenizer={manifest['tokenizer']}")
    print(f"files={manifest['files']}  rows={manifest['rows']}  "
          f"rows_changed={manifest['rows_changed']}  rewritten_blocks={manifest['rewritten_blocks']}  "
          f"leftover_call_ids={manifest['leftover_call_ids']}")
    print(f"chars  before p50/p95/max={b['chars']['p50']}/{b['chars']['p95']}/{b['chars']['max']}  "
          f"after={a['chars']['p50']}/{a['chars']['p95']}/{a['chars']['max']}")
    print(f"tokens before p50/p95/max={b['tokens']['p50']}/{b['tokens']['p95']}/{b['tokens']['max']}  "
          f"after={a['tokens']['p50']}/{a['tokens']['p95']}/{a['tokens']['max']}")
    for name, f in manifest["per_file"].items():
        print(f"  {name}: rows={f['rows']} rewritten={f['rewritten_blocks']} "
              f"chars {f['before']['chars']['max']}->{f['after']['chars']['max']} "
              f"tokens {f['before']['tokens']['max']}->{f['after']['tokens']['max']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=DEFAULT_INPUT, help="frontier dir to read")
    ap.add_argument("--out", default=DEFAULT_OUTPUT,
                    help="NEW dir to write (default: data/frontier_tools_clean)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; do not create or write the output dir")
    args = ap.parse_args(argv)
    manifest = clean_dir(args.input, args.out, dry_run=args.dry_run)
    _print_report(manifest)
    if not args.dry_run:
        print(f"wrote {manifest['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
