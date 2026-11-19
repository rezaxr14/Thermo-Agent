"""Build the tool-using student corpora (Gate 4).

Reads ground-truth rows from ``data/processed*`` (never from the single-turn corpora),
turns each one into deterministic tool traces with ``synth.tool_traces.build_trace``,
and mixes in accepted frontier rows **only when their SFT text really contains tool
turns**.  Outputs go to new, additive directories ``data/sft_tools`` and
``data/grpo_tools``; ``data/sft_all``, ``data/grpo_all`` and ``data/frontier*`` are
never opened for writing.

    python scripts/build_tool_corpus.py --src data/processed --out data/sft_tools \
        --grpo-out data/grpo_tools --tasks repair,design,audit,ledger --split train \
        --limit 5

    # opt-in: prompts get AGENTS.md + skills + the exact eval tool contract;
    # written to new additive *_h directories (default: <out>_h / <grpo-out>_h)
    python scripts/build_tool_corpus.py --src data/processed --out data/sft_tools \
        --grpo-out data/grpo_tools --prompt-variant harness_tools

The SFT manifest records per-task/per-source counts, char-length statistics, the
over-8k drop count and the held-out contamination guard result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermo_agent.agent.tool_prompt import build_harness_tool_messages  # noqa: E402
from thermo_agent.core.schema import Plant  # noqa: E402
from thermo_agent.synth import tool_traces  # noqa: E402
from thermo_agent.teacher import validate  # noqa: E402

TASKS = ("repair", "design", "audit", "ledger")
FRONTIER_DIRS = ("data/frontier", "data/frontier_prov")
PROTECTED = ("data/sft_all", "data/grpo_all", "data/sft", "data/grpo",
             "data/frontier", "data/frontier_prov", "data/frontier_export")

# Same high-entropy identity fields as tests/test_contamination.py: a verbatim leak
# is found regardless of JSON field order or whitespace.
PLANT_FIELDS = ("plant", "plant_broken", "plant_original")
SITE_FIELDS = ("site",)
SPEC_FIELDS = ("ledger",)
IDENTITY_FIELDS = PLANT_FIELDS + SITE_FIELDS + SPEC_FIELDS


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read(path: pathlib.Path, limit: int = 0) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def _parse(value):
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, (dict, list)) and value:
        return value
    return None


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def signature_set(row: dict) -> set[tuple[str, str]]:
    """All ``(kind, sha256)`` identity signatures of one dataset row."""
    out: set[tuple[str, str]] = set()
    for name in IDENTITY_FIELDS:
        obj = _parse(row.get(name))
        if obj is not None:
            out.add((name, _digest(obj)))
    site, spec = _parse(row.get("site")), _parse(row.get("ref_spec"))
    if site is not None and spec is not None:
        out.add(("site+ref_spec", _digest({"site": site, "ref_spec": spec})))
    rid = row.get("id") or row.get("record_id")
    if isinstance(rid, str) and rid:
        out.add(("id", _digest(rid)))
    return out


def build_heldout_signatures(heldout_dir: pathlib.Path | str) -> tuple[set[tuple[str, str]], list[str]]:
    """Hash plants/sites/specs/ids from every frozen held-out ``*_test_*.jsonl``."""
    heldout_dir = pathlib.Path(heldout_dir)
    if not heldout_dir.is_absolute():
        heldout_dir = ROOT / heldout_dir
    files = sorted(heldout_dir.glob("*_test_*.jsonl"))
    if not files:
        files = sorted(ROOT / "tests" / "fixtures" / "heldout" / "*_test_*.jsonl")
    index: set[tuple[str, str]] = set()
    for path in files:
        for row in _read(path):
            index |= signature_set(row)
    return index, [str(p.relative_to(ROOT)) for p in files]


def scan_rows(rows, index: set[tuple[str, str]]) -> list[dict]:
    """Rows whose plant/site/spec/id signature appears in the held-out index."""
    leaks: list[dict] = []
    for row in rows:
        kinds = sorted(k for k, h in signature_set(row) if (k, h) in index)
        if kinds:
            leaks.append({"record_id": row.get("record_id", row.get("id")), "kinds": kinds})
    return leaks


def contamination_guard(rows, heldout_dir: pathlib.Path | str = "data/heldout") -> dict:
    """Assert zero overlap between the corpus rows and the frozen held-out set."""
    index, files = build_heldout_signatures(heldout_dir)
    leaks = scan_rows(rows, index)
    return {
        "heldout_files": files,
        "signatures": len(index),
        "rows_scanned": len(rows),
        "overlap": len(leaks),
        "leaks": leaks[:10],
        "passed": not leaks,
    }


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]


def _length_stats(values: list[int]) -> dict:
    return {"n": len(values), "p50": _pct(values, 0.50),
            "p95": _pct(values, 0.95), "max": max(values) if values else None}


def _aggregate_counts(entries: list[dict]) -> dict:
    """Sum kept rows by generator across all sources of one task."""
    out: dict[str, int] = {}
    for entry in entries:
        out[entry["generator"]] = out.get(entry["generator"], 0) + entry["kept"]
    return out


def _final_from_trace(text: str) -> str:
    """Final assistant answer of a frontier tool trace (tail after the last tool result)."""
    idx = text.rfind("</tool_result>")
    body = text[idx + len("</tool_result>"):] if idx >= 0 else text
    body = re.sub(r"<think>.*?</think>", "", body, flags=re.S)
    return body.strip()


def _source_hash(src: pathlib.Path) -> str:
    return hashlib.md5(str(src).replace("\\", "/").encode("utf-8")).hexdigest()[:4]


def _keep(row: dict, task: str) -> dict:
    return {k: row[k] for k in validate.KEEP.get(task, ()) if k in row}


def _plant_from_row(row: dict):
    """The row's plant, parsed in the same key order as ToolLoop._base_messages."""
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                return Plant.model_validate_json(row[key])
            except Exception:
                pass
    return None


def _check_destinations(out: pathlib.Path, grpo_out: pathlib.Path) -> None:
    """Refuse to write into any corpus we do not own (additive-only contract)."""
    for dest in (out, grpo_out):
        resolved = dest.resolve()
        for name in PROTECTED:
            if resolved == (ROOT / name).resolve():
                raise SystemExit(f"refusing to write into protected corpus: {dest}")
        if resolved in (ROOT.resolve(), (ROOT / "data").resolve()):
            raise SystemExit(f"refusing to write into the repo/data root: {dest}")


def _check_harness_destinations(out: pathlib.Path, grpo_out: pathlib.Path) -> None:
    """Harness corpora are additive-only: new *_h dirs, never an existing non-empty dir."""
    _check_destinations(out, grpo_out)
    for dest in (out, grpo_out):
        if not dest.name.endswith("_h"):
            raise SystemExit(f"harness corpus must be a new *_h directory: {dest}")
        if dest.exists() and any(dest.iterdir()):
            raise SystemExit(f"refusing to write into non-empty harness corpus: {dest}")


def _emit(task: str, row: dict, trace: dict, variant: str, generator: str,
          src: pathlib.Path, prompt: list, prompt_variant: str = "plain") -> tuple[dict, dict]:
    """One SFT row (full tool trace) and one GRPO row (final answer + task columns)."""
    harness = prompt_variant == "harness_tools"
    if harness:
        prompt = build_harness_tool_messages(prompt, task, _plant_from_row(row))
    gen_tag = "dt" if generator == "deterministic_tools" else "ft"
    provenance = {
        "generator": generator,
        "variant": variant,
        "tool_calls": trace["tool_calls"],
        "source_row_id": str(row.get("id", "")),
        "source_dir": str(src).replace("\\", "/"),
    }
    sft = {
        "id": f"{row.get('id', 'row')}-{gen_tag}-{variant}-{_source_hash(src)}",
        "record_id": str(row.get("id", "")),
        "split": row.get("split", "train"),
        "task": task,
        "source": generator,
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": trace["sft_completion"]}],
        "provenance": provenance,
        "harness": "agents+skills" if harness else "none",
        "memory_notes": [],
        "char_length": trace["char_length"],
    }
    if harness:
        sft["prompt_variant"] = "harness_tools"
    if trace["tokens"] is not None:
        sft["tokens"] = trace["tokens"]
    sft.update(_keep(row, task))

    verdict = trace["verdict"] if "verdict" in trace else validate.validate(task, trace["final_answer"], row)
    grpo = validate.teacher_row(task, trace["final_answer"], row, verdict,
                                tag=f"{gen_tag}-{variant}")
    grpo["id"] = f"{grpo['id']}-{_source_hash(src)}"
    grpo["source"] = generator
    grpo["record_id"] = str(row.get("id", ""))
    grpo["provenance"] = dict(provenance)
    if harness:
        grpo["prompt"] = prompt
        grpo["harness"] = "agents+skills"
        grpo["prompt_variant"] = "harness_tools"
    grpo["trajectory"] = {
        "final": trace["final_answer"],
        "sft_completion": trace["sft_completion"],
        "turns": trace["turns"],
        "tool_calls": trace["tool_calls"],
    }
    return sft, grpo


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(src: list[str], out: str, grpo_out: str, tasks=TASKS, split: str = "train",
          limit: int = 0, include_frontier: bool = True,
          max_tool_result_chars: int = tool_traces.DEFAULT_MAX_TOOL_RESULT_CHARS,
          max_tokens: int = tool_traces.DEFAULT_MAX_TOKENS,
          heldout_dir: str = "data/heldout", prompt_variant: str = "plain",
          out_harness: str | None = None, grpo_out_harness: str | None = None) -> dict:
    if prompt_variant not in ("plain", "harness_tools"):
        raise SystemExit(f"prompt_variant must be plain or harness_tools, got {prompt_variant!r}")
    out_dir, grpo_dir = pathlib.Path(out), pathlib.Path(grpo_out)
    if prompt_variant == "harness_tools":
        # Additive-only: default to new <out>_h / <grpo-out>_h directories.
        out_dir = pathlib.Path(out_harness) if out_harness else pathlib.Path(str(out) + "_h")
        grpo_dir = pathlib.Path(grpo_out_harness) if grpo_out_harness else pathlib.Path(str(grpo_out) + "_h")
        _check_harness_destinations(out_dir, grpo_dir)
    else:
        _check_destinations(out_dir, grpo_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grpo_dir.mkdir(parents=True, exist_ok=True)
    src_dirs = [pathlib.Path(s) for s in src]

    sft_rows: list[dict] = []
    grpo_rows: list[dict] = []
    char_lengths: list[int] = []
    token_lengths: list[int] = []
    sources: list[dict] = []
    dropped_over_8k = 0
    dropped_large_tools = 0
    dropped_invalid = 0
    frontier_stats = {"scanned": 0, "with_tool_turns": 0, "kept": 0,
                      "deduplicated": 0, "rejected": 0, "over_budget": 0}

    def add(task: str, row: dict, trace: dict, variant: str, generator: str, src_dir: pathlib.Path):
        nonlocal dropped_over_8k, dropped_large_tools, dropped_invalid
        if not trace["accepted"] or trace["tool_calls"] == 0:
            dropped_invalid += 1
            return False
        if trace["over_budget"]:
            dropped_over_8k += 1
            return False
        dropped_large_tools += len(trace["dropped_large_tools"])
        sft, grpo = _emit(task, row, trace, variant, generator, src_dir, row["prompt"], prompt_variant)
        sft_rows.append(sft)
        grpo_rows.append(grpo)
        char_lengths.append(trace["char_length"])
        if trace["tokens"] is not None:
            token_lengths.append(trace["tokens"])
        return True

    for task in tasks:
        for src_dir in src_dirs:
            rows = _read(src_dir / f"{task}_{split}.jsonl", limit)
            entry = {"task": task, "source_dir": str(src_dir).replace("\\", "/"),
                     "generator": "deterministic_tools", "rows_read": len(rows),
                     "variants": {}}
            for variant in ("direct", "recover"):
                kept = 0
                for row in rows:
                    trace = tool_traces.build_trace(
                        row, task, variant=variant, strict=False,
                        max_tool_result_chars=max_tool_result_chars, max_tokens=max_tokens)
                    kept += bool(add(task, row, trace, variant, "deterministic_tools", src_dir))
                entry["variants"][variant] = kept
            entry["kept"] = sum(entry["variants"].values())
            sources.append(entry)

        if include_frontier:
            seen: set[str] = set()
            fentry = {"task": task, "source_dir": "data/frontier*",
                      "generator": "frontier_tools", "rows_read": 0, "variants": {"tool_cot": 0}}
            local_tool_rows = 0
            for fd in FRONTIER_DIRS:
                for row in _read(pathlib.Path(fd) / f"frontier_{task}_{split}.jsonl"):
                    frontier_stats["scanned"] += 1
                    completion = row.get("completion")
                    text = (completion[0]["content"] if isinstance(completion, list) and completion
                            else str(completion or ""))
                    if "<tool_call" not in text and "<tool_result" not in text:
                        continue
                    frontier_stats["with_tool_turns"] += 1
                    local_tool_rows += 1
                    rid = str(row.get("record_id") or row.get("id", ""))
                    if rid in seen:
                        frontier_stats["deduplicated"] += 1
                        continue
                    seen.add(rid)
                    answer = _final_from_trace(text)
                    verdict = validate.validate(task, answer, row)
                    if not verdict["accepted"]:
                        frontier_stats["rejected"] += 1
                        continue
                    n_calls = text.count("<tool_call")
                    trace = {
                        "sft_completion": text, "turns": [], "tool_calls": n_calls,
                        "accepted": True, "final_answer": answer, "verdict": verdict,
                        "char_length": len(text), "tokens": None, "over_budget": False,
                        "dropped_large_tools": [],
                    }
                    tok = tool_traces.combine.tokenizer()
                    if tok is not None:
                        n_tok = len(tok.encode(text, add_special_tokens=False).ids)
                        n_prompt_tok = sum(len(tok.encode(m.get("content", ""), add_special_tokens=False).ids)
                                           for m in row.get("prompt", []))
                        trace["tokens"] = n_prompt_tok + n_tok + tool_traces.TOKEN_RESERVE
                        trace["over_budget"] = trace["tokens"] > max_tokens
                        if trace["over_budget"]:
                            frontier_stats["over_budget"] += 1
                            continue
                    if add(task, row, trace, "tool_cot", "frontier_tools", pathlib.Path("data/frontier")):
                        fentry["variants"]["tool_cot"] += 1
                        frontier_stats["kept"] += 1
            fentry["rows_read"] = local_tool_rows
            fentry["kept"] = fentry["variants"]["tool_cot"]
            sources.append(fentry)

    # write split files (only inside the two directories we own)
    for task in tasks:
        name = f"{task}_{split}.jsonl"
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for row in sft_rows:
                if row["task"] == task:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (grpo_dir / name).open("w", encoding="utf-8") as fh:
            for row in grpo_rows:
                if row["task"] == task:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    guard = contamination_guard(sft_rows, heldout_dir)
    if not guard["passed"]:
        raise SystemExit(f"contamination guard failed: {guard['leaks'][:5]}")

    by_task = {}
    for task in tasks:
        lengths = [r["char_length"] for r in sft_rows if r["task"] == task]
        by_task[task] = _length_stats(lengths)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "synth.tool_traces (deterministic) + accepted frontier tool traces",
        "config": {
            "src": [str(s).replace("\\", "/") for s in src], "out": str(out).replace("\\", "/"),
            "grpo_out": str(grpo_out).replace("\\", "/"), "tasks": list(tasks),
            "split": split, "limit": limit, "include_frontier": include_frontier,
            "max_tool_result_chars": max_tool_result_chars, "max_tokens": max_tokens,
            "prompt_variant": prompt_variant,
        },
        "totals": {"sft": len(sft_rows), "grpo": len(grpo_rows)},
        "counts": {
            task: _aggregate_counts([s for s in sources if s["task"] == task])
            for task in tasks
        },
        "sources": sources,
        "char_length": _length_stats(char_lengths),
        "char_length_by_task": by_task,
        "token_length": _length_stats(token_lengths),
        "dropped_over_8k": dropped_over_8k,
        "dropped_large_tool_results": dropped_large_tools,
        "dropped_invalid_or_no_tools": dropped_invalid,
        "frontier": frontier_stats,
        "contamination_guard": guard,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/processed", help="comma-separated processed data dirs")
    ap.add_argument("--out", default="data/sft_tools")
    ap.add_argument("--grpo-out", default="data/grpo_tools")
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task names")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="rows per source/task (0 = all)")
    ap.add_argument("--include-frontier", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-tool-result-chars", type=int, default=tool_traces.DEFAULT_MAX_TOOL_RESULT_CHARS)
    ap.add_argument("--max-tokens", type=int, default=tool_traces.DEFAULT_MAX_TOKENS)
    ap.add_argument("--heldout", default="data/heldout")
    ap.add_argument("--prompt-variant", choices=["plain", "harness_tools"], default="plain",
                    help="harness_tools adds AGENTS.md + skills + the eval tool contract to each prompt")
    ap.add_argument("--out-harness", default=None,
                    help="harness_tools SFT dir (default: <out>_h); must be a new *_h dir")
    ap.add_argument("--grpo-out-harness", default=None,
                    help="harness_tools GRPO dir (default: <grpo-out>_h); must be a new *_h dir")
    a = ap.parse_args()
    tasks = tuple(t.strip() for t in a.tasks.split(",") if t.strip())
    manifest = build(
        [s.strip() for s in a.src.split(",") if s.strip()],
        a.out, a.grpo_out, tasks, a.split, a.limit, a.include_frontier,
        a.max_tool_result_chars, a.max_tokens, a.heldout,
        a.prompt_variant, a.out_harness, a.grpo_out_harness)
    print(json.dumps({
        "totals": manifest["totals"], "counts": manifest["counts"],
        "char_length": manifest["char_length"],
        "dropped_over_8k": manifest["dropped_over_8k"],
        "contamination_guard": {"overlap": manifest["contamination_guard"]["overlap"],
                                "passed": manifest["contamination_guard"]["passed"]},
        "frontier": manifest["frontier"],
    }, indent=1))


if __name__ == "__main__":
    main()
