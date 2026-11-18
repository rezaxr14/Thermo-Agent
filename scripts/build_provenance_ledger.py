#!/usr/bin/env python3
"""Build per-trace provenance for the thermo_agent corpora.

One ledger record is emitted per UNIQUE reasoning trace (canonical id), not per
corpus row.  A trace is identified by its source record id, its model (kept as a
provenance suffix) and its harness mode when the answer was produced through a
tool/skill harness.  Paraphrase markers (``#vN``) never create a new trace: they
are collected as variants on the trace they paraphrase.

Sources are streamed, never materialised as a corpus:

  data/processed/*.jsonl, data/processed_arch/*.jsonl   synthetic ground truth
  data/processed/meta_*.json, data/processed_arch/...   generator + seed
  data/teacher_runs/*.jsonl          every journal, auto-discovered (any model)
  data/frontier/*.jsonl              accepted teacher rows + validation verdict
  data/memory/*.jsonl                rejected attempts / memory notes
  data/memory/failure_reasons.json   aggregate rejection reasons (stats only)
  data/sft/*.jsonl, data/grpo/*.jsonl        downstream corpus rows
  paper/assets/examples.json                  downstream paper examples
  reports/eval_*.json                         downstream eval arms (replay provenance)

Outputs (all derived, all reproducible):

  data/provenance/trace_ledger.jsonl
  data/provenance/stats.json
  docs/PROVENANCE.md

Run with:  python scripts/build_provenance_ledger.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = ROOT / "data" / "provenance" / "trace_ledger.jsonl"
STATS_PATH = ROOT / "data" / "provenance" / "stats.json"
DOCS_PATH = ROOT / "docs" / "PROVENANCE.md"

SCHEMA_VERSION = "1.0"
KIND_RANK = {"synthetic": 0, "frontier": 1, "harness": 2, "adversarial": 3}

VARIANT_RE = re.compile(r"#v\d+")
MODEL_SUFFIX_RE = re.compile(r"-t-([A-Za-z0-9][A-Za-z0-9._/-]*)$")
SPLIT_FROM_FILENAME_RE = re.compile(
    r"(?:^|_)(test_id|test_ood_[a-z_]+|test_[a-z_]+|train|val)$"
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def rel(path: Path | str) -> str:
    """Repo-relative, forward-slash path (stable across Windows/Linux)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def slugify(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(model).strip().lower()).strip("-")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream JSONL as (1-based line number, parsed object)."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield lineno, obj


def parse_id(raw: str) -> tuple[str, list[str], str | None]:
    """Return (base id, paraphrase tokens, model slug from -t- suffix)."""
    raw = str(raw or "")
    variants = VARIANT_RE.findall(raw)
    base = VARIANT_RE.sub("", raw)
    model_slug = None
    match = MODEL_SUFFIX_RE.search(base)
    if match:
        model_slug = match.group(1)
        base = base[: match.start()]
    return base, variants, model_slug


def trace_id_for(base: str, model_slug: str | None, harness: str | None) -> str:
    """Canonical trace id: base id + model suffix + non-default harness suffix."""
    tid = base
    if model_slug:
        tid += "@m=" + model_slug
    if harness and harness != "none":
        tid += "@h=" + harness
    return tid


def base_id_of(tid: str) -> str:
    return tid.split("@m=", 1)[0].split("@h=", 1)[0]


def split_from_filename(path: Path) -> str | None:
    match = SPLIT_FROM_FILENAME_RE.search(path.stem)
    return match.group(1) if match else None


def short_site_ref(site: Any) -> str | None:
    """Stable reference for a site payload (full payloads are left in source)."""
    if site is None:
        return None
    text = site if isinstance(site, str) else json.dumps(site, sort_keys=True)
    return "sha256:" + sha256(text)[:16]


def infer_task_from_id(base_id: str, prefix_map: dict[str, str]) -> str | None:
    prefix = base_id.split("-", 1)[0]
    return prefix_map.get(prefix)


def bump_kind(rec: dict[str, Any], kind: str) -> None:
    if KIND_RANK[kind] > KIND_RANK[rec["kind"]]:
        rec["kind"] = kind


def new_record(tid: str, base: str) -> dict[str, Any]:
    return {
        "trace_id": tid,
        "kind": "synthetic",
        "base_id": base,
        "task": None,
        "split": None,
        "topology": None,
        "site_ref": None,
        "topology_site_ref": None,
        "origin": None,
        "origin_summary": None,
        "journal": None,
        "journal_occurrences": [],
        "harness": None,
        "paraphrase_variants": [],
        "cost_nanodollars": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "completion_sha256": None,
        "completion_length": None,
        "finish_reason": None,
        "error": None,
        "tool_limit_exhausted": False,
        "acceptance": {
            "status": "unknown",
            "reason": None,
            "reward": None,
            "reasons": [],
            "source": None,
        },
        "downstream": [],
        "first_seen": None,
        "notes": [],
        "attributes": {},
        # scratch, removed on finalisation
        "_variants": set(),
        "_downstream": {},
    }


class Ledger:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def get(self, tid: str) -> dict[str, Any]:
        rec = self.records.get(tid)
        if rec is None:
            rec = new_record(tid, base_id_of(tid))
            self.records[tid] = rec
        return rec

    def see(
        self,
        rec: dict[str, Any],
        *,
        source: str,
        path: Path | str,
        lineno: int | None,
        timestamp: str | None = None,
    ) -> None:
        if rec["first_seen"] is None:
            rec["first_seen"] = {
                "source": source,
                "path": rel(path),
                "line": lineno,
                "timestamp": timestamp,
            }

    def add_downstream(
        self,
        rec: dict[str, Any],
        *,
        artifact: str,
        kind: str,
        row_id: str | None = None,
        count: int = 1,
        arm: str | None = None,
        replay_file: str | None = None,
        harness_mode: str | None = None,
        paraphrases: Iterable[str] = (),
    ) -> None:
        key = (artifact, kind, arm or "")
        entry = rec["_downstream"].get(key)
        if entry is None:
            entry = {
                "artifact": artifact,
                "kind": kind,
                "count": 0,
                "row_ids": set(),
                "harness_modes": set(),
                "paraphrase_variants": set(),
            }
            if arm:
                entry["arm"] = arm
            if replay_file:
                entry["replay_file"] = replay_file
            rec["_downstream"][key] = entry
        entry["count"] += count
        if row_id:
            entry["row_ids"].add(row_id)
        if harness_mode:
            entry["harness_modes"].add(harness_mode)
        entry["paraphrase_variants"].update(paraphrases)


# --------------------------------------------------------------------------- #
# pass 0: discovery
# --------------------------------------------------------------------------- #
def discover() -> dict[str, Any]:
    processed = sorted((ROOT / "data" / "processed").glob("*.jsonl"))
    arch = sorted((ROOT / "data" / "processed_arch").glob("*.jsonl"))
    big = sorted((ROOT / "data" / "processed_big").glob("*.jsonl"))
    journals = sorted((ROOT / "data" / "teacher_runs").glob("*.jsonl"))
    frontier = sorted((ROOT / "data" / "frontier").glob("*.jsonl"))
    memory = sorted((ROOT / "data" / "memory").glob("*.jsonl"))
    sft = sorted((ROOT / "data" / "sft").glob("*.jsonl"))
    sft_all = sorted((ROOT / "data" / "sft_all").glob("*.jsonl"))
    grpo = sorted((ROOT / "data" / "grpo").glob("*.jsonl"))
    grpo_all = sorted((ROOT / "data" / "grpo_all").glob("*.jsonl"))
    reports = sorted((ROOT / "reports").glob("eval_*.json"))
    return {
        "processed": processed,
        "processed_arch": arch,
        "processed_big": big,
        "journals": journals,
        "frontier": frontier,
        "memory": memory,
        "sft": sft,
        "sft_all": sft_all,
        "grpo": grpo,
        "grpo_all": grpo_all,
        "reports": reports,
        "metas": sorted((ROOT / "data" / "processed").glob("meta_*.json"))
        + sorted((ROOT / "data" / "processed_arch").glob("meta_*.json"))
        + sorted((ROOT / "data" / "processed_big").glob("meta_*.json")),
        "failure_reasons": ROOT / "data" / "memory" / "failure_reasons.json",
        "examples": ROOT / "paper" / "assets" / "examples.json",
    }


def load_metas(discovered: dict[str, Any]) -> list[dict[str, Any]]:
    metas = []
    for path in discovered["metas"]:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta["_path"] = path
        metas.append(meta)
    return metas


def meta_for(directory: str, filename: str, metas: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a processed file to its meta_*.json (e.g. repair -> meta_repair.json)."""
    task_hint = filename.split("_", 1)[0]
    candidates = [m for m in metas if str(m.get("task", "")).startswith(task_hint)]
    if directory == "processed_arch":
        arch = [m for m in candidates if "arch" in str(m.get("task", ""))]
        if arch:
            candidates = arch
    else:
        non_arch = [m for m in candidates if "arch" not in str(m.get("task", ""))]
        if non_arch:
            candidates = non_arch
    return max(candidates, key=lambda m: len(str(m.get("task", "")))) if candidates else None


def build_model_map(discovered: dict[str, Any]) -> dict[str, str]:
    """Map exact model ids and id suffixes to exact model ids, from the data itself."""
    mapping: dict[str, str] = {}
    for path in discovered["journals"]:
        for _, row in iter_jsonl(path):
            model = row.get("model")
            if model:
                mapping[slugify(model)] = str(model)
    for path in discovered["frontier"]:
        for _, row in iter_jsonl(path):
            _, _, suffix = parse_id(row.get("id", ""))
            if suffix and suffix not in mapping:
                mapping[suffix] = suffix
    for path in discovered["memory"]:
        for _, row in iter_jsonl(path):
            source = str(row.get("source", ""))
            if source.startswith("frontier:"):
                suffix = source.split(":", 1)[1]
                if suffix not in mapping:
                    mapping[suffix] = suffix
    return mapping


def build_prefix_map(discovered: dict[str, Any]) -> dict[str, str]:
    """Infer task from an id prefix by observing the generated corpus."""
    mapping: dict[str, str] = {}
    for group in ("processed", "processed_arch", "processed_big"):
        for path in discovered[group]:
            task = path.stem.split("_", 1)[0]
            for _, row in iter_jsonl(path):
                prefix = str(row.get("id", "")).split("-", 1)[0]
                if prefix:
                    mapping[prefix] = task
    return mapping


# --------------------------------------------------------------------------- #
# pass 1: origins
# --------------------------------------------------------------------------- #
def add_synthetic(
    ledger: Ledger, discovered: dict[str, Any], metas: list[dict[str, Any]]
) -> None:
    for group in ("processed", "processed_arch", "processed_big"):
        directory = ROOT / "data" / group
        for path in discovered[group]:
            meta = meta_for(group, path.stem, metas)
            params = (meta or {}).get("params", {}) or {}
            generator = "datagen:" + str((meta or {}).get("task", path.stem.split("_", 1)[0]))
            seed = params.get("seed")
            created = (meta or {}).get("created_utc")
            meta_rel = rel(meta["_path"]) if meta else None
            task = path.stem.split("_", 1)[0]
            split = split_from_filename(path)
            for lineno, row in iter_jsonl(path):
                base, variants, model_slug = parse_id(row.get("id", ""))
                tid = trace_id_for(base, model_slug, None)
                rec = ledger.get(tid)
                rec["task"] = rec["task"] or task or row.get("task")
                rec["split"] = rec["split"] or split or row.get("split")
                rec["topology"] = rec["topology"] or row.get("topology")
                site = row.get("site")
                rec["site_ref"] = rec["site_ref"] or short_site_ref(site)
                rec["topology_site_ref"] = rec["topology_site_ref"] or (
                    str(row.get("topology") or rec["site_ref"] or "")
                )
                rec["origin"] = {
                    "type": "generator",
                    "generator": generator,
                    "seed": seed,
                    "meta_file": meta_rel,
                    "task": (meta or {}).get("task"),
                    "schema_version": (meta or {}).get("schema_version"),
                    "created_utc": created,
                }
                rec["origin_summary"] = f"{generator} seed={seed} ({meta_rel})"
                rec["_variants"].update(variants)
                if row.get("paraphrase"):
                    para = row["paraphrase"]
                    if isinstance(para, dict) and para.get("variant") is not None:
                        rec["_variants"].add("#v" + str(para["variant"]))
                ledger.see(
                    rec,
                    source=group,
                    path=directory / path.name,
                    lineno=lineno,
                    timestamp=created,
                )
                for key in ("fault_type", "fault_types", "min_edits", "detected", "deducible"):
                    if key in row and key not in rec["attributes"]:
                        rec["attributes"][key] = row[key]
                if not rec["notes"]:
                    rec["notes"].append(f"synthetic trace from {group}/{path.name}")


def add_journals(ledger: Ledger, discovered: dict[str, Any], model_map: dict[str, str],
                 prefix_map: dict[str, str]) -> None:
    for path in discovered["journals"]:
        for lineno, row in iter_jsonl(path):
            base, variants, slug = parse_id(row.get("id", ""))
            harness = str(row.get("harness") or "none")
            model = str(row.get("model") or "") or model_map.get(slug or "", slug or "")
            model_slug = slugify(model) if model else slug
            if model and model_slug:
                model_map.setdefault(model_slug, model)
            tid = trace_id_for(base, model_slug, harness)
            rec = ledger.get(tid)
            bump_kind(rec, "harness" if harness != "none" else "frontier")
            rec["harness"] = harness
            rec["task"] = rec["task"] or row.get("task") or infer_task_from_id(base, prefix_map)
            rec["split"] = rec["split"] or row.get("split") or split_from_filename(path)
            rec["topology"] = rec["topology"] or row.get("topology")
            rec["_variants"].update(variants)
            rec["origin"] = {
                "type": "model",
                "model": model or model_slug,
                "model_slug": model_slug,
                "harness": harness,
                "journal": rel(path),
            }
            rec["origin_summary"] = f"{model or model_slug} [harness={harness}]"
            usage = row.get("usage") or {}
            cost = _num(usage.get("total_cost_nanodollars", row.get("total_cost_nanodollars")))
            in_tok = _num(usage.get("input_tokens", row.get("prompt_tokens")))
            out_tok = _num(usage.get("output_tokens", row.get("completion_tokens")))
            completion = row.get("completion")
            if isinstance(completion, list):
                completion_text = "".join(
                    str(block.get("content", "")) for block in completion
                    if isinstance(block, dict)
                )
            else:
                completion_text = str(completion or "")
            finish = usage.get("finish_reason") or row.get("finish_reason") or row.get("finish")
            error = row.get("error") or usage.get("error") or row.get("error_reason")
            exhausted = bool(row.get("tool_limit_exhausted", False))
            occurrence = {
                "path": rel(path),
                "line": lineno,
                "timestamp": row.get("ts"),
                "completion_sha256": sha256(completion_text) if completion_text else None,
                "completion_length": len(completion_text),
                "finish_reason": finish,
                "error": error,
                "model": model or model_slug,
                "harness": harness,
                "cost_nanodollars": cost,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "tool_limit_exhausted": exhausted,
            }
            rec["journal_occurrences"].append(occurrence)
            if rec["journal"] is None:
                rec["journal"] = {"path": rel(path), "line": lineno}
                # A trace can appear in several journals (probe, export, replay).
                # Tokens/cost are taken from the first occurrence so re-exports
                # never double-count spend; every occurrence is still recorded.
                rec["cost_nanodollars"] = cost
                rec["input_tokens"] = in_tok
                rec["output_tokens"] = out_tok
                rec["completion_sha256"] = occurrence["completion_sha256"]
                rec["completion_length"] = occurrence["completion_length"]
                rec["finish_reason"] = finish
                rec["error"] = error
                rec["tool_limit_exhausted"] = exhausted
            if exhausted:
                rec["tool_limit_exhausted"] = True
            if row.get("tool_calls") is not None and "tool_calls" not in rec["attributes"]:
                rec["attributes"]["tool_calls"] = len(row["tool_calls"]) if isinstance(row["tool_calls"], list) else 1
            if row.get("tools") is not None and "tools" not in rec["attributes"]:
                rec["attributes"]["tools"] = row["tools"]
            ledger.see(
                rec,
                source="teacher_runs",
                path=path,
                lineno=lineno,
                timestamp=row.get("ts"),
            )


def add_frontier(ledger: Ledger, discovered: dict[str, Any], model_map: dict[str, str],
                 prefix_map: dict[str, str]) -> None:
    for path in discovered["frontier"]:
        for lineno, row in iter_jsonl(path):
            base, variants, slug = parse_id(row.get("id", ""))
            if not slug and row.get("record_id"):
                _, _, slug = parse_id(row.get("record_id", ""))
            model = model_map.get(slug or "", slug or "")
            tid = trace_id_for(base, slug, None)
            rec = ledger.get(tid)
            bump_kind(rec, "frontier")
            rec["_variants"].update(variants)
            rec["task"] = rec["task"] or row.get("task") or infer_task_from_id(base, prefix_map)
            rec["split"] = rec["split"] or row.get("split")
            rec["topology"] = rec["topology"] or row.get("topology")
            site = row.get("site")
            rec["site_ref"] = rec["site_ref"] or short_site_ref(site)
            rec["topology_site_ref"] = rec["topology_site_ref"] or (
                str(row.get("topology") or rec["site_ref"] or "")
            )
            if slug:
                rec["origin"] = {
                    "type": "model",
                    "model": model,
                    "model_slug": slug,
                    "harness": rec["harness"] or "none",
                    "frontier_file": rel(path),
                }
                rec["origin_summary"] = f"{model} [harness={rec['harness'] or 'none'}]"
            validation = row.get("validation") or {}
            if not isinstance(validation, dict):
                validation = {}
            reward = validation.get("reward")
            reasons = validation.get("reasons") or []
            if isinstance(reasons, str):
                reasons = [reasons]
            rejected = validation.get("rejected") or validation.get("rejection_reasons")
            status = "accepted"
            if rejected or (reward is not None and _num(reward) <= 0):
                status = "rejected"
            rec["acceptance"] = {
                "status": status,
                "reason": (reasons[0] if reasons else None),
                "reward": reward,
                "reasons": list(reasons),
                "source": rel(path),
            }
            if rejected:
                rec["acceptance"]["rejection_reasons"] = (
                    list(rejected) if isinstance(rejected, (list, tuple)) else [rejected]
                )
            ledger.see(rec, source="frontier", path=path, lineno=lineno)
            rec["notes"].append(
                f"frontier validation {status} reward={reward} reasons={list(reasons)}"
            )
            for key in ("fault_type", "fault_types", "min_edits", "deducible"):
                if key in row and key not in rec["attributes"]:
                    rec["attributes"][key] = row[key]


def add_memory(ledger: Ledger, discovered: dict[str, Any], model_map: dict[str, str],
               prefix_map: dict[str, str]) -> None:
    for path in discovered["memory"]:
        for lineno, row in iter_jsonl(path):
            note_id = str(row.get("id") or "")
            base, variants, slug = parse_id(note_id)
            source = str(row.get("source") or "")
            symptoms = row.get("symptoms")
            lesson = row.get("lesson")
            reasons = [str(x) for x in (symptoms, lesson) if x]
            # The rejected attempt is its own trace: the note id (minus any #vN
            # marker) identifies the one attempt, and the trailing hash keeps
            # different adversarial fixes on the same record distinct.
            attempt_tid = VARIANT_RE.sub("", note_id) or f"memory-line-{lineno}"
            attempt = ledger.get(attempt_tid)
            bump_kind(attempt, "adversarial")
            attempt["task"] = row.get("task") or infer_task_from_id(
                str(row.get("record_id") or base), prefix_map
            )
            attempt["topology"] = row.get("architecture")
            attempt["topology_site_ref"] = str(row.get("architecture") or "")
            attempt["_variants"].update(variants)
            attempt["acceptance"] = {
                "status": "rejected" if row.get("accepted") is False else "unknown",
                "reason": reasons[0] if reasons else None,
                "reward": None,
                "reasons": reasons,
                "source": source or rel(path),
            }
            attempt["origin"] = {
                "type": "memory",
                "source": source,
                "record_id": row.get("record_id"),
                "memory_file": rel(path),
            }
            attempt["origin_summary"] = f"memory:{source}" if source else f"memory:{rel(path)}"
            attempt["notes"].append(f"memory note {note_id}: {lesson}" if lesson else f"memory note {note_id}")
            ledger.see(attempt, source="memory", path=path, lineno=lineno)
            attempt["attributes"]["record_id"] = row.get("record_id")

            # The note is ALSO a downstream artifact of the trace it comments on.
            record_base, _, record_slug = parse_id(str(row.get("record_id") or ""))
            if not record_slug and source.startswith("frontier:"):
                record_slug = source.split(":", 1)[1]
            if record_slug:
                record_slug = slugify(model_map.get(record_slug, record_slug))
            parent_tid = trace_id_for(record_base, record_slug, None)
            if record_base and parent_tid in ledger.records:
                ledger.add_downstream(
                    ledger.records[parent_tid],
                    artifact=rel(path),
                    kind="memory_note",
                    row_id=note_id,
                    harness_mode=None,
                    paraphrases=variants,
                )


# --------------------------------------------------------------------------- #
# pass 2: downstream usage
# --------------------------------------------------------------------------- #
def tid_from_row(row: dict[str, Any], model_map: dict[str, str]) -> tuple[str | None, str, list[str]]:
    raw = str(row.get("id") or "")
    base, variants, slug = parse_id(raw)
    if not base:
        return None, raw, variants
    if slug:
        slug = slugify(model_map.get(slug, slug))
    return trace_id_for(base, slug, None), raw, variants


def add_corpus_rows(ledger: Ledger, discovered: dict[str, Any], model_map: dict[str, str]) -> Counter:
    expansions: Counter = Counter()
    for group, artifact_kind in (("sft", "sft_row"), ("grpo", "grpo_row")):
        for path in discovered[group]:
            artifact = rel(path)
            for _, row in iter_jsonl(path):
                tid, raw_id, variants = tid_from_row(row, model_map)
                if tid is None:
                    continue
                rec = ledger.records.get(tid)
                if rec is None:
                    # Downstream rows may reference traces from a journal that
                    # was not present at build time; keep the reference anyway.
                    rec = ledger.get(tid)
                    rec["notes"].append("referenced by downstream row but no origin record found")
                row_variants = set(variants)
                para = row.get("paraphrase")
                if isinstance(para, dict) and para.get("variant") is not None:
                    row_variants.add("#v" + str(para["variant"]))
                rec["_variants"].update(row_variants)
                ledger.add_downstream(
                    rec,
                    artifact=artifact,
                    kind=artifact_kind,
                    row_id=raw_id,
                    harness_mode=row.get("harness"),
                    paraphrases=row_variants,
                )
                if variants or row.get("paraphrase"):
                    expansions["rows_with_variant"] += 1
                    expansions[f"by_task:{row.get('task')}"] += 1
                    expansions["distinct"] += 1
    return expansions


def add_paper_examples(ledger: Ledger, discovered: dict[str, Any], model_map: dict[str, str]) -> None:
    path = discovered["examples"]
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    artifact = rel(path)
    for example in examples:
        if not isinstance(example, dict):
            continue
        tid, raw_id, variants = tid_from_row(example, model_map)
        if tid is None:
            continue
        rec = ledger.records.get(tid) or ledger.get(tid)
        rec["_variants"].update(variants)
        ledger.add_downstream(
            rec,
            artifact=artifact,
            kind="paper_example",
            row_id=raw_id,
            paraphrases=variants,
        )
        rec["attributes"].setdefault("paper_file", example.get("file"))


def _walk_examples(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "id" in node and any(k in node for k in ("success", "exact", "solves", "reward")):
            yield node
        for value in node.values():
            yield from _walk_examples(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_examples(value)


def add_eval_reports(ledger: Ledger, discovered: dict[str, Any], model_map: dict[str, str]) -> None:
    """Attribute eval arms through their replay files, then the example ids."""
    replay_index: dict[str, set[str]] = {}
    for path in discovered["journals"]:
        ids: set[str] = set()
        for _, row in iter_jsonl(path):
            base, _, suffix_slug = parse_id(str(row.get("id") or ""))
            model = str(row.get("model") or "")
            slug = suffix_slug or (slugify(model) if model else None)
            if slug:
                slug = slugify(model_map.get(slug, slug))
            harness = str(row.get("harness") or "none")
            tid = trace_id_for(base, slug, harness)
            if tid:
                ids.add(tid)
        replay_index[rel(path)] = ids

    for path in discovered["reports"]:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        meta = report.get("meta") or {}
        arm = str(meta.get("model") or meta.get("display_name") or path.stem)
        replay_file = meta.get("replay_file")
        replay_rel = rel(replay_file) if replay_file else None
        replay_ids = replay_index.get(replay_rel, set()) if replay_rel else set()
        example_ids: dict[str, str] = {}
        for example in _walk_examples(report):
            example_base, _, example_slug = parse_id(example.get("id", ""))
            if example_base:
                example_ids[example_base] = str(example["id"])
        artifact = rel(path)
        for tid in replay_ids:
            rec = ledger.records.get(tid) or ledger.get(tid)
            row_id = next(
                (rid for base, rid in example_ids.items() if base_id_of(tid) == base),
                None,
            )
            ledger.add_downstream(
                rec,
                artifact=artifact,
                kind="eval_arm",
                row_id=row_id,
                arm=arm,
                replay_file=replay_rel,
                harness_mode=rec.get("harness") or "none",
            )
            if tid not in ledger.records:
                rec["notes"].append("eval replay trace has no origin record")


# --------------------------------------------------------------------------- #
# finalisation + stats
# --------------------------------------------------------------------------- #
def _num(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _finalise_downstream(entry: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in entry.items()
        if key not in ("row_ids", "harness_modes", "paraphrase_variants")
    }
    row_ids = sorted(entry.get("row_ids", set()))
    harness_modes = sorted(entry.get("harness_modes", set()))
    variants = sorted(entry.get("paraphrase_variants", set()))
    if row_ids:
        out["row_ids"] = row_ids
    if harness_modes:
        out["harness_modes"] = harness_modes
    if variants:
        out["paraphrase_variants"] = variants
    return out


def finalise_records(ledger: Ledger) -> list[dict[str, Any]]:
    records = []
    for tid in sorted(ledger.records):
        rec = ledger.records[tid]
        rec["paraphrase_variants"] = sorted(rec.pop("_variants"))
        downstream_map = rec.pop("_downstream")
        rec["downstream"] = sorted(
            (_finalise_downstream(entry) for entry in downstream_map.values()),
            key=lambda e: (e["artifact"], e["kind"], e.get("arm", "")),
        )
        if rec["topology_site_ref"] is None:
            rec["topology_site_ref"] = str(rec["topology"] or rec["site_ref"] or "")
        if rec["origin_summary"] is None:
            rec["origin_summary"] = str(rec["origin"])
        records.append(rec)
    return records


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 4) if den else 0.0


def build_stats(records: list[dict[str, Any]], discovered: dict[str, Any],
                expansions: Counter, ledger_bytes: int, ledger_sha: str,
                failure_reasons: dict[str, Any]) -> dict[str, Any]:
    total = len(records)
    by_kind: Counter = Counter(r["kind"] for r in records)
    by_source: Counter = Counter(
        (r["first_seen"] or {}).get("source", "unknown") for r in records
    )
    by_file: Counter = Counter(
        (r["first_seen"] or {}).get("path", "unknown") for r in records
    )
    by_task: Counter = Counter(r["task"] or "(unknown)" for r in records)
    by_split: Counter = Counter(r["split"] or "(none)" for r in records)
    def model_of(rec: dict[str, Any]) -> str:
        origin = rec["origin"] or {}
        if origin.get("model"):
            return str(origin["model"])
        return "(adversarial/memory)" if rec["kind"] == "adversarial" else "(synthetic)"

    by_model: Counter = Counter(model_of(r) for r in records)
    harness_modes: Counter = Counter(r["harness"] or "none" for r in records)
    acceptance: Counter = Counter(r["acceptance"]["status"] for r in records)

    model_stats: dict[str, Any] = {}
    for model in sorted(by_model):
        subset = [r for r in records if model_of(r) == model]
        acc = sum(1 for r in subset if r["acceptance"]["status"] == "accepted")
        rej = sum(1 for r in subset if r["acceptance"]["status"] == "rejected")
        cost = sum(r["cost_nanodollars"] for r in subset)
        per = round(cost / acc, 2) if acc else None
        model_stats[model] = {
            "traces": len(subset),
            "accepted": acc,
            "rejected": rej,
            "unknown": len(subset) - acc - rej,
            "acceptance_rate_of_known_pct": _pct(acc, acc + rej),
            "accepted_share_of_all_pct": _pct(acc, len(subset)),
            "cost_nanodollars": cost,
            "input_tokens": sum(r["input_tokens"] for r in subset),
            "output_tokens": sum(r["output_tokens"] for r in subset),
            "cost_per_accepted_nanodollars": per,
            "cost_per_accepted_usd": round(per / 1e9, 9) if per is not None else None,
        }

    task_stats: dict[str, Any] = {}
    for task in sorted(by_task):
        subset = [r for r in records if (r["task"] or "(unknown)") == task]
        task_stats[task] = {
            "traces": len(subset),
            "accepted": sum(1 for r in subset if r["acceptance"]["status"] == "accepted"),
            "rejected": sum(1 for r in subset if r["acceptance"]["status"] == "rejected"),
            "cost_nanodollars": sum(r["cost_nanodollars"] for r in subset),
        }

    artifact_stats: dict[str, Any] = {}
    downstream_kind_counts: Counter = Counter()
    downstream_rows: Counter = Counter()
    for rec in records:
        for entry in rec["downstream"]:
            artifact = entry["artifact"]
            info = artifact_stats.setdefault(
                artifact,
                {"artifact": artifact, "kind": entry["kind"], "unique_traces": 0, "rows": 0},
            )
            info["unique_traces"] += 1
            info["rows"] += entry["count"]
            downstream_kind_counts[entry["kind"]] += 1
            downstream_rows[entry["kind"]] += entry["count"]

    with_downstream = sum(1 for r in records if r["downstream"])
    coverage_by_kind: dict[str, Any] = {}
    for kind in sorted(by_kind):
        subset = [r for r in records if r["kind"] == kind]
        hit = sum(1 for r in subset if r["downstream"])
        coverage_by_kind[kind] = {
            "traces": len(subset),
            "with_downstream": hit,
            "coverage_pct": _pct(hit, len(subset)),
        }
    coverage_by_source: dict[str, Any] = {}
    for source in sorted(by_source):
        subset = [r for r in records if (r["first_seen"] or {}).get("source", "unknown") == source]
        hit = sum(1 for r in subset if r["downstream"])
        coverage_by_source[source] = {
            "traces": len(subset),
            "with_downstream": hit,
            "coverage_pct": _pct(hit, len(subset)),
        }

    variants_per_trace = sum(len(r["paraphrase_variants"]) for r in records)
    traces_with_variants = sum(1 for r in records if r["paraphrase_variants"])
    known_den = acceptance["accepted"] + acceptance["rejected"]
    total_cost = sum(r["cost_nanodollars"] for r in records)
    total_accepted = acceptance["accepted"]
    latest_ts = max(
        [str(x) for x in [o.get("timestamp") for r in records for o in r["journal_occurrences"]] if x]
        + [str(m.get("created_utc")) for m in discovered.get("_metas", []) if m.get("created_utc")]
        or ["(none)"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/build_provenance_ledger.py",
        "reproducible": True,
        "source_latest_timestamp": latest_ts,
        "ledger": {
            "path": rel(LEDGER_PATH),
            "records": total,
            "bytes": ledger_bytes,
            "sha256": ledger_sha,
        },
        "totals": {
            "traces": total,
            "by_kind": dict(sorted(by_kind.items())),
            "by_first_seen_source": dict(sorted(by_source.items())),
            "downstream_entries": sum(len(r["downstream"]) for r in records),
            "journal_occurrences": sum(len(r["journal_occurrences"]) for r in records),
        },
        "by_first_seen_file": dict(sorted(by_file.items())),
        "by_task": task_stats,
        "by_split": dict(sorted(by_split.items())),
        "by_model": model_stats,
        "acceptance": {
            "counts": dict(sorted(acceptance.items())),
            "known_traces": known_den,
            "acceptance_rate_of_known_pct": _pct(acceptance["accepted"], known_den),
            "accepted_share_of_all_pct": _pct(acceptance["accepted"], total),
        },
        "cost": {
            "total_nanodollars": total_cost,
            "total_usd": round(total_cost / 1e9, 6),
            "accepted_traces": total_accepted,
            "nanodollars_per_accepted_trace": round(total_cost / total_accepted, 2) if total_accepted else None,
            "by_model": {
                model: {
                    "cost_nanodollars": info["cost_nanodollars"],
                    "input_tokens": info["input_tokens"],
                    "output_tokens": info["output_tokens"],
                    "cost_per_accepted_nanodollars": info["cost_per_accepted_nanodollars"],
                    "accepted_traces": info["accepted"],
                }
                for model, info in model_stats.items()
            },
        },
        "harness_modes": {
            mode: {"traces": count, "share_pct": _pct(count, total)}
            for mode, count in sorted(harness_modes.items())
        },
        "paraphrase_expansion": {
            "traces_with_variants": traces_with_variants,
            "variant_tokens_total": variants_per_trace,
            "mean_variants_per_trace": round(variants_per_trace / traces_with_variants, 4) if traces_with_variants else 0.0,
            "downstream_rows_with_variant": expansions.get("rows_with_variant", 0),
            "by_task": {
                key.split("by_task:", 1)[1]: value
                for key, value in sorted(expansions.items())
                if key.startswith("by_task:")
            },
        },
        "downstream": {
            "artifacts": sorted(artifact_stats.values(), key=lambda x: x["artifact"]),
            "entries_by_kind": dict(sorted(downstream_kind_counts.items())),
            "rows_by_kind": dict(sorted(downstream_rows.items())),
        },
        "coverage": {
            "traces_with_downstream": with_downstream,
            "traces_total": total,
            "coverage_pct": _pct(with_downstream, total),
            "by_kind": coverage_by_kind,
            "by_source": coverage_by_source,
        },
        "memory_failure_reasons": failure_reasons,
        "sources_discovered": {
            key: [rel(p) for p in value]
            for key, value in discovered.items()
            if isinstance(value, list) and not key.startswith("_")
        },
        "notes": [
            "One ledger record per canonical trace id; paraphrase #vN markers are variants, not traces.",
            "Canonical id = <base id> + @m=<model> + @h=<harness> for non-default harness modes.",
            "Costs and tokens come from the first journal occurrence; duplicate journal exports are listed under journal_occurrences and never double-counted.",
            "Old Gemini and GPT-nano traces remain in the ledger; future big-model journals are appended additively by re-running the script (no code edit).",
        ],
    }


# --------------------------------------------------------------------------- #
# docs (generated from stats.json so the numbers can never drift)
# --------------------------------------------------------------------------- #
def render_docs(stats: dict[str, Any]) -> str:
    def table(headers: list[str], rows: list[list[Any]]) -> str:
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        return "\n".join(out)

    s = stats
    kind_rows = [
        [k, v["traces"], v["with_downstream"], f'{v["coverage_pct"]}%']
        for k, v in s["coverage"]["by_kind"].items()
    ]
    source_rows = [
        [k, v["traces"], v["with_downstream"], f'{v["coverage_pct"]}%']
        for k, v in s["coverage"]["by_source"].items()
    ]
    model_rows = [
        [
            m,
            v["traces"],
            v["accepted"],
            v["rejected"],
            f'{v["acceptance_rate_of_known_pct"]}%',
            v["cost_nanodollars"],
            v["input_tokens"],
            v["output_tokens"],
            v["cost_per_accepted_nanodollars"]
            if v["cost_per_accepted_nanodollars"] is not None
            else "n/a",
        ]
        for m, v in s["by_model"].items()
    ]
    task_rows = [
        [k, v["traces"], v["accepted"], v["rejected"], v["cost_nanodollars"]]
        for k, v in s["by_task"].items()
    ]
    split_rows = [[k, v] for k, v in s["by_split"].items()]
    harness_rows = [[k, v["traces"], f'{v["share_pct"]}%'] for k, v in s["harness_modes"].items()]
    artifact_rows = [
        [a["artifact"], a["kind"], a["unique_traces"], a["rows"]]
        for a in s["downstream"]["artifacts"]
    ]
    fr = s["memory_failure_reasons"]
    top_lessons = fr.get("top_lessons", []) if isinstance(fr, dict) else []
    lesson_rows = [[t.get("lesson", ""), t.get("count", 0)] for t in top_lessons[:8]]
    p = s["paraphrase_expansion"]
    fr_by_task = json.dumps(fr.get("by_task", {}), ensure_ascii=False) if isinstance(fr, dict) else "{}"
    fr_by_fault = json.dumps(fr.get("by_fault_type", {}), ensure_ascii=False) if isinstance(fr, dict) else "{}"
    fr_n = fr.get("n", 0) if isinstance(fr, dict) else 0

    return f"""# Per-trace provenance

This document describes the per-trace provenance ledger for thermo_agent. The
ledger is **per reasoning trace, not per corpus row**: a synthetic ground-truth
record, a hosted-model answer, and a tool-harness answer each get their own
record, and every downstream use is attached to that record.

## Regenerate

```bash
python scripts/build_provenance_ledger.py
```

The script streams the source files (it loads no model and imports no training
code) and rewrites three derived artifacts:

| Output | Contents |
| --- | --- |
| `data/provenance/trace_ledger.jsonl` | one JSON object per canonical trace |
| `data/provenance/stats.json` | exact counts computed from the ledger records |
| `docs/PROVENANCE.md` | this file, including the generated stats tables |

`scripts/build_corpus.ps1` calls the script automatically after the swapped
corpora have been verified. All three outputs are fully derived and can be
deleted and regenerated at any time; no source data is modified. The large
`trace_ledger.jsonl` is gitignored (like its source corpora); `stats.json` and
this document are committed so the exact numbers remain reviewable.

## Canonical id and correlation

`trace_id` is `<base id>` plus `@m=<model>` when the trace came from a hosted
model plus `@h=<harness>` when the answer was produced through a non-default
harness (`tools`, `skills`, `agents+skills`). The model suffix is always kept as
provenance. `#vN` paraphrase markers are **stripped**: they are listed in
`paraphrase_variants` on the trace they rewrite, so 20 paraphrases of one repair
trace are one ledger record with 20 variants, not 20 traces. Correlation with
SFT, GRPO, paper and eval rows is done exclusively through these canonical ids
and the recorded `base_id`, never by counting rows.

When the same answer is exported to several journals (for example a probe and a
full run), it is still one trace: `journal` holds the first path + line and
`journal_occurrences` holds every path + line with its own hash, token count and
cost. Cost and token fields on the record come from the first occurrence, so
re-exports never double-count spend.

## Ledger record schema

| Field | Meaning |
| --- | --- |
| `trace_id` | canonical trace id (see above) |
| `base_id` | source record id with `#vN`, model and harness suffixes removed; this is the correlation key shared with `memory.record_id`, paper examples and eval replay rows |
| `kind` | `synthetic` (datagen), `frontier` (accepted/hosted answer), `harness` (tool/skill run), `adversarial` (rejected memory attempt) |
| `task`, `split` | task and split, inferred from the source path/row and id prefix |
| `topology`, `site_ref`, `topology_site_ref` | plant topology and/or a stable `sha256:` reference to the site payload |
| `origin` | generator + seed from `meta_*.json`, or the exact model id (with slug and journal path) |
| `journal` | `{{"path", "line"}}` of the first journal occurrence (null for synthetic traces) |
| `journal_occurrences` | every journal hit: path, line, timestamp, completion hash + length, finish/error, tokens, cost, harness |
| `harness` | `none`, `tools`, `skills`, `agents+skills`, … |
| `paraphrase_variants` | sorted `#vN` tokens observed for this trace |
| `cost_nanodollars`, `input_tokens`, `output_tokens` | usage from the first journal occurrence |
| `completion_sha256`, `completion_length`, `finish_reason`, `error`, `tool_limit_exhausted` | completion fingerprint and outcome |
| `acceptance` | `status` (`accepted`/`rejected`/`unknown`), verdict `reason`, `reward`, `reasons`, `source`; rejection reasons where present |
| `downstream` | structured list of `{{artifact, kind, count, row_ids, harness_modes, paraphrase_variants}}` plus `arm`/`replay_file` for eval arms |
| `first_seen` | `{{source, path, line, timestamp}}` where the trace was first observed |
| `notes`, `attributes` | human-readable provenance notes and auxiliary source fields |

`acceptance` records an actual verdict: frontier validation verdicts (accepted),
memory-note rejections (rejected), and no verdict at all (unknown). Synthetic
ground-truth traces have no teacher verdict, so they are `unknown` even though
they are valid training targets; do not read `unknown` as a failure.

## Adding a new model journal

No code edit is needed. Drop the journal into `data/teacher_runs/` as JSONL with
one answer per line (`id`, `model`, `harness`, `usage`, `completion`, optional
`ts`), then re-run `python scripts/build_provenance_ledger.py`. The script
auto-discovers every `*.jsonl` in that directory, slugifies the exact `model`
id, and matches it to the `-t-<model>` suffix used in `data/frontier`,
`data/sft` and `data/grpo`. A new harness mode automatically becomes a new
`@h=<mode>` trace id. Eval arms are attributed through each report's
`meta.replay_file`, so a replay journal for a future model also needs no code
change.

Old Gemini (`google/gemini-3.1-flash-lite-preview`) and GPT-nano
(`openai/gpt-5.4-nano-2026-03-17`) traces remain in the ledger unchanged.
Future bigger-model traces are **appended additively**: re-running the builder
adds their records and downstream links without rewriting the existing ones
(other than refreshing the regenerated tables below).

## Current exact stats (generated from `data/provenance/stats.json`)

Generated from {s["ledger"]["records"]} ledger records
(`sha256:{s["ledger"]["sha256"][:16]}`, {s["ledger"]["bytes"]} bytes).

### Totals

{table(["Metric", "Value"], [
    ["Traces", s["totals"]["traces"]],
    ["Downstream entries", s["totals"]["downstream_entries"]],
    ["Journal occurrences", s["totals"]["journal_occurrences"]],
    ["Accepted / rejected / unknown", "{accepted} / {rejected} / {unknown}".format(**s["acceptance"]["counts"])],
    ["Acceptance rate of known traces", f'{s["acceptance"]["acceptance_rate_of_known_pct"]}%'],
    ["Total cost (nanodollars)", s["cost"]["total_nanodollars"]],
    ["Total cost (USD)", s["cost"]["total_usd"]],
    ["Nanodollars per accepted trace", s["cost"]["nanodollars_per_accepted_trace"]],
])}

### Coverage by kind

{table(["Kind", "Traces", "With downstream", "Coverage"], kind_rows)}

### Coverage by first-seen source

{table(["Source", "Traces", "With downstream", "Coverage"], source_rows)}

### By model

{table(["Model", "Traces", "Accepted", "Rejected", "Acc. rate (known)", "Cost (n$)", "Input tok.", "Output tok.", "Cost/accepted (n$)"], model_rows)}

### By task

{table(["Task", "Traces", "Accepted", "Rejected", "Cost (n$)"], task_rows)}

### By split

{table(["Split", "Traces"], split_rows)}

### Harness-mode shares

{table(["Harness", "Traces", "Share"], harness_rows)}

### Paraphrase expansion

{table(["Metric", "Value"], [
    ["Traces with variants", p["traces_with_variants"]],
    ["Variant tokens total", p["variant_tokens_total"]],
    ["Mean variants per trace", p["mean_variants_per_trace"]],
    ["Downstream rows with a variant", p["downstream_rows_with_variant"]],
])}

By task: {p["by_task"] or "(none)"}

### Downstream artifacts

{table(["Artifact", "Kind", "Unique traces", "Rows"], artifact_rows)}

### Aggregate memory rejection reasons

`data/memory/failure_reasons.json` has no per-trace ids, so it is reported here
and in `stats.json` only (it is never turned into synthetic ledger records).

Total recorded failures: **{fr_n}**.
By task: {fr_by_task}.
By fault type: {fr_by_fault}.

{table(["Top lesson", "Count"], lesson_rows)}
"""


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    discovered = discover()
    metas = load_metas(discovered)
    discovered["_metas"] = metas
    model_map = build_model_map(discovered)
    prefix_map = build_prefix_map(discovered)

    ledger = Ledger()
    add_synthetic(ledger, discovered, metas)
    add_journals(ledger, discovered, model_map, prefix_map)
    add_frontier(ledger, discovered, model_map, prefix_map)
    add_memory(ledger, discovered, model_map, prefix_map)

    expansions = add_corpus_rows(ledger, discovered, model_map)
    add_paper_examples(ledger, discovered, model_map)
    add_eval_reports(ledger, discovered, model_map)

    records = finalise_records(ledger)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n")
    ledger_bytes = LEDGER_PATH.stat().st_size
    ledger_sha = sha256(LEDGER_PATH.read_text(encoding="utf-8"))

    failure_reasons: dict[str, Any] = {}
    if discovered["failure_reasons"].exists():
        try:
            loaded = json.loads(discovered["failure_reasons"].read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                failure_reasons = loaded
        except (OSError, json.JSONDecodeError):
            pass

    stats = build_stats(
        records, discovered, expansions, ledger_bytes, ledger_sha, failure_reasons
    )
    STATS_PATH.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    DOCS_PATH.write_text(render_docs(stats), encoding="utf-8", newline="\n")

    print(f"wrote {len(records)} provenance records -> {rel(LEDGER_PATH)}")
    print(f"wrote exact stats -> {rel(STATS_PATH)}")
    print(f"wrote generated documentation -> {rel(DOCS_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
