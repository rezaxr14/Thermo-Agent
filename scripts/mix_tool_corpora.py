"""Mix the deterministic Gate-4 tool corpora with accepted frontier tool rows.

Additive-only Gate-4 preparation step: it reads (never writes) the already-built
deterministic corpora ``data/sft_tools`` / ``data/grpo_tools`` plus the teacher
frontier exports ``data/frontier``, and writes a *new* deterministic +
accepted-frontier mix into ``data/sft_mix`` / ``data/grpo_mix``.  The protected
corpora (``data/sft_tools``, ``data/grpo_tools``, ``data/sft_all``,
``data/grpo_all``, ``data/frontier*``) are only ever opened for reading.

    python scripts/mix_tool_corpora.py --help
    python scripts/mix_tool_corpora.py --stats          # print table, write nothing
    python scripts/mix_tool_corpora.py                  # deterministic-only today

Selection rules
---------------
A frontier row is eligible only when BOTH hold:

1. It is verifier-accepted.  We honour an explicit ``accepted: true`` (top level
   or ``verdict.accepted``), ``validation.accepted``, or -- the convention the
   frontier files actually use -- a positive ``validation.reward`` with no
   ``validation.rejected`` / ``rejection_reasons`` marker.  This mirrors
   ``scripts/build_provenance_ledger.py`` (frontier rows carry a validation
   verdict, not a top-level ``accepted`` flag) and ``teacher.validate``.
2. Its completion / trajectory contains at least one ``<tool_call`` AND one
   ``<tool_result`` turn.  Rows with zero tool turns are rejected even when the
   verifier accepted their final answer.

Today all 668 rows in ``data/frontier`` are accepted but none carry tool turns,
so the selected count is 0 and the script says so loudly while still emitting a
valid deterministic-only corpus.

Frontier caps (precedence)
--------------------------
``--frontier-per-task N`` (N > 0) caps eligible frontier rows to N per task and
takes precedence over ``--frontier-ratio``.  Only when ``--frontier-per-task``
is 0 (the default) is ``--frontier-ratio R`` (R > 0) used, capping to
``floor(R * deterministic_sft_rows_for_task)`` per task.  With both at 0 there
is no cap: every eligible row is taken.  A positive cap applies a seeded
shuffle, so the chosen ids are reproducible from ``--seed``.

Rows are passed through unchanged (no re-rendering, no truncation, no field
drops).  Ids are deduplicated within each source; an id shared between sources
in the same output stream is a hard error.

Plain A-slice (decision E2)
---------------------------
``--a-slice <dir>`` adds the plain-prompt single-turn rows built by
``scripts/build_a_slice.py`` to the SFT mix (``data/sft_mix_h`` = harness tool
rows + plain A rows).  A rows keep ``prompt_variant="plain"`` / ``harness="none"``
and are never re-prompted by ``--harness-prompt``, which applies only to
frontier tool rows.  ``--a-slice-ratio R`` (when R > 0) caps A rows per task to
``round(R * deterministic_rows_for_that_task)``; R = 0 (the default) takes every
row in the slice dir, which is already task-balanced at build time.  Omitting
``--a-slice`` leaves the deterministic/frontier mix byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ("repair", "design", "audit", "ledger")
SCRIPT_TAG = "mix_tool_corpora.py"
MANIFEST_NAME = "MANIFEST.json"
HARNESS_MARKER = "MULTI-TURN TOOL ENVIRONMENT"

# Never write into any of these (additive-only contract; see build_tool_corpus.py).
PROTECTED = (
    "data/sft_tools", "data/grpo_tools", "data/sft_tools_h", "data/grpo_tools_h",
    "data/sft_all", "data/grpo_all", "data/sft", "data/grpo",
    "data/frontier", "data/frontier_prov", "data/a_slice", "data/heldout",
    "data/frontier_export", "data/frontier_eval_export",
)

TOOL_CALL = "<tool_call"
TOOL_RESULT = "<tool_result"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _resolve(value: pathlib.Path | str) -> pathlib.Path:
    """Absolute path; relative paths are anchored at the repo root."""
    path = pathlib.Path(value)
    return path if path.is_absolute() else (ROOT / path)


def _rel(path: pathlib.Path) -> str:
    """Repo-relative path when possible, absolute otherwise (for manifests)."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_jsonl(path: pathlib.Path) -> list[dict]:
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
    return rows


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover(directory: pathlib.Path, task: str, prefix: str = "") -> list[pathlib.Path]:
    """All ``<prefix><task>_*.jsonl`` files in a directory, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{prefix}{task}_*.jsonl"))


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None
    return None


def _plant_from_row(row: dict):
    """The row's plant, parsed exactly as scripts/build_tool_corpus.py does."""
    from thermo_agent.core.schema import Plant

    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                return Plant.model_validate_json(row[key])
            except Exception:
                pass
    return None


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(m.get("content", "")) for m in prompt if isinstance(m, dict))
    return ""


def to_harness_prompt(row: dict, task: str) -> dict:
    """Copy a row with the exact `ours_harness` system prompt.

    A frontier trace was generated under the teacher's LangGraph prompt, while the
    student is evaluated in `ours_harness` with AGENTS.md + skills + the shared
    `<tool_call>` contract. Training on the teacher's prompt would teach the model a
    different contract from the one it is scored with, so every frontier row that joins
    a harness mix is re-prompted here with the shared builder. The completion (the
    teacher's real tool trajectory) is never touched.
    """
    from thermo_agent.agent.tool_prompt import build_harness_tool_messages

    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"row {row.get('id')!r} has no message-list prompt")
    out = dict(row)
    out["prompt"] = build_harness_tool_messages(prompt, task, _plant_from_row(row))
    out["harness"] = "agents+skills"
    out["prompt_variant"] = "harness_tools"
    return out


def is_harness_prompt(row: dict) -> bool:
    return (row.get("prompt_variant") == "harness_tools"
            or HARNESS_MARKER in _prompt_text(row.get("prompt")))


# --------------------------------------------------------------------------- #
# selection predicates
# --------------------------------------------------------------------------- #
def is_verifier_accepted(row: dict) -> bool:
    """True when a frontier row carries an actual verifier-acceptance signal.

    Precedence mirrors the repo's journal classifier (teacher_campaign_stats)
    and frontier provenance ledger: explicit top-level / verdict flags first,
    then the ``validation`` verdict that real ``data/frontier`` rows carry.
    """
    if row.get("accepted") is True:
        return True
    if row.get("accepted") is False:
        return False
    verdict = row.get("verdict")
    if isinstance(verdict, dict):
        if verdict.get("graded") is False:
            return False
        if "accepted" in verdict:
            return verdict["accepted"] is True
        if str(verdict.get("status", "")).lower() == "accepted":
            return True
    validation = row.get("validation")
    if isinstance(validation, dict):
        if validation.get("accepted") is True:
            return True
        if validation.get("accepted") is False:
            return False
        if validation.get("rejected") or validation.get("rejection_reasons"):
            return False
        reward = _number(validation.get("reward"))
        if reward is not None:
            return reward > 0
    return False


def _text_from_completion(row: dict) -> str:
    value = row.get("completion")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _text_from_trajectory(row: dict) -> str:
    value = row.get("trajectory")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        sft = value.get("sft_completion")
        if isinstance(sft, str):
            parts.append(sft)
        try:
            parts.append(json.dumps(value, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
        return "\n".join(parts)
    return ""


def has_tool_turns(row: dict) -> bool:
    """True only when a completion/trajectory has BOTH a call and a result turn."""
    blob = _text_from_completion(row) + "\n" + _text_from_trajectory(row)
    return TOOL_CALL in blob and TOOL_RESULT in blob


def _cap(per_task: int, ratio: float, deterministic_count: int) -> tuple[int | None, str]:
    """Resolve the per-task frontier cap and report which knob won."""
    if per_task and per_task > 0:
        return per_task, "per_task"
    if ratio and ratio > 0:
        return max(0, int(deterministic_count * ratio)), "ratio"
    return None, "unlimited"


def _dedupe(rows: list[dict], label: str, stats: dict) -> list[dict]:
    """Keep the first row per id, counting drops under ``stats[label]``."""
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        rid = str(row.get("id") or "")
        if rid in seen:
            stats[f"deduplicated_{label}"] += 1
            if "deduplicated" in stats:
                stats["deduplicated"] += 1
            continue
        seen.add(rid)
        kept.append(row)
    return kept


# --------------------------------------------------------------------------- #
# planning (read-only: used by both --stats and the writer)
# --------------------------------------------------------------------------- #
def build_mix(deterministic_sft: pathlib.Path | str,
              deterministic_grpo: pathlib.Path | str,
              frontier: pathlib.Path | str,
              tasks=TASKS, seed: int = 0, frontier_per_task: int = 0,
              frontier_ratio: float = 0.0, harness_prompt: bool = False,
              a_slice=None, a_slice_ratio: float = 0.0) -> dict:
    """Read all inputs and return the planned mix without writing anything.

    ``harness_prompt=True`` builds the ``ours_harness`` mix: the deterministic inputs
    must already be the ``*_h`` (harness-prompt) corpora, and every selected frontier
    row is re-prompted with the shared harness system prompt so teacher and student
    see the same contract. Its completion (the real tool trajectory) is untouched.

    ``a_slice`` adds plain rows to SFT only. They are never re-prompted, and keep
    ``prompt_variant="plain"`` / ``harness="none"`` next to the ``harness_tools``
    tool rows. ``a_slice_ratio > 0`` caps A rows per task to
    ``round(ratio * deterministic_rows)``; 0 takes every row in the slice dir.
    """
    det_sft_dir, det_grpo_dir, frontier_dir = (
        _resolve(deterministic_sft), _resolve(deterministic_grpo), _resolve(frontier))
    a_dir = _resolve(a_slice) if a_slice else None
    rng = random.Random(seed)

    sft_rows: list[dict] = []
    grpo_rows: list[dict] = []
    sft_owner: dict[str, str] = {}
    grpo_owner: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    per_source = {"deterministic_sft": {}, "deterministic_grpo": {}, "frontier": {},
                  "a_slice": {}}
    stream_counts = {"sft": {}, "grpo": {}}
    a_stats = {
        "enabled": a_dir is not None, "dir": _rel(a_dir) if a_dir else None,
        "ratio": a_slice_ratio, "scanned": {}, "selected": {},
        "skipped_over_cap": {}, "selected_ids": [], "deduplicated": 0,
        "rejected_non_plain": 0,
    }
    frontier_stats = {
        "scanned": 0, "accepted": 0, "with_tool_turns": 0, "eligible": 0,
        "selected_count": 0, "selected_ids": [], "skipped_over_cap": 0,
        "deduplicated": 0, "cap_mode": "unlimited", "cap_per_task": frontier_per_task,
        "cap_ratio": frontier_ratio, "reprompted": 0,
    }
    dedupe_stats = {"deduplicated_deterministic_sft": 0,
                    "deduplicated_deterministic_grpo": 0}
    collisions: list[dict] = []
    inputs_missing: list[str] = []

    for name, directory in (("deterministic_sft", det_sft_dir),
                            ("deterministic_grpo", det_grpo_dir),
                            ("frontier", frontier_dir)) + (
                                (("a_slice", a_dir),) if a_dir is not None else ()):
        if not directory.is_dir():
            inputs_missing.append(_rel(directory))

    for task in tasks:
        # ---- deterministic SFT -------------------------------------------- #
        dsft: list[dict] = []
        for path in _discover(det_sft_dir, task):
            input_hashes[_rel(path)] = _sha256_file(path)
            dsft.extend(_read_jsonl(path))
        dsft = _dedupe(dsft, "deterministic_sft", dedupe_stats)
        for row in dsft:
            rid = str(row.get("id") or "")
            sft_owner[rid] = "deterministic_sft"
        sft_rows.extend(dsft)
        per_source["deterministic_sft"][task] = len(dsft)

        # ---- deterministic GRPO ------------------------------------------ #
        dgrpo: list[dict] = []
        for path in _discover(det_grpo_dir, task):
            input_hashes[_rel(path)] = _sha256_file(path)
            dgrpo.extend(_read_jsonl(path))
        dgrpo = _dedupe(dgrpo, "deterministic_grpo", dedupe_stats)
        for row in dgrpo:
            rid = str(row.get("id") or "")
            grpo_owner[rid] = "deterministic_grpo"
        grpo_rows.extend(dgrpo)
        per_source["deterministic_grpo"][task] = len(dgrpo)

        if harness_prompt:
            plain = [str(r.get("id") or "") for r in (dsft + dgrpo) if not is_harness_prompt(r)]
            if plain:
                raise SystemExit(
                    f"--harness-prompt needs the *_h deterministic corpora: {len(plain)} "
                    f"{task} row(s) still carry the plain prompt (e.g. {plain[:3]}). Pass "
                    "--deterministic-sft data/sft_tools_h --deterministic-grpo data/grpo_tools_h.")

        # ---- frontier candidates ----------------------------------------- #
        candidates: list[dict] = []
        for path in _discover(frontier_dir, task, prefix="frontier_"):
            input_hashes[_rel(path)] = _sha256_file(path)
            for row in _read_jsonl(path):
                frontier_stats["scanned"] += 1
                if not is_verifier_accepted(row):
                    continue
                frontier_stats["accepted"] += 1
                if not has_tool_turns(row):
                    continue
                frontier_stats["with_tool_turns"] += 1
                candidates.append(row)
        candidates = _dedupe(candidates, "frontier", frontier_stats)
        frontier_stats["eligible"] += len(candidates)

        cap, cap_mode = _cap(frontier_per_task, frontier_ratio, len(dsft))
        if task == tasks[0]:
            frontier_stats["cap_mode"] = cap_mode
        if cap is not None and len(candidates) > cap:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            chosen = shuffled[:cap]
            frontier_stats["skipped_over_cap"] += len(candidates) - cap
        else:
            chosen = list(candidates)

        if harness_prompt and chosen:
            chosen = [to_harness_prompt(row, task) for row in chosen]
            frontier_stats["reprompted"] += len(chosen)

        for row in chosen:
            rid = str(row.get("id") or "")
            for stream, owner in (("sft", sft_owner), ("grpo", grpo_owner)):
                if rid in owner and owner[rid] != "frontier":
                    collisions.append({"id": rid, "stream": stream,
                                       "sources": [owner[rid], "frontier"]})
            if rid not in sft_owner:
                sft_owner[rid] = "frontier"
                sft_rows.append(row)
            if rid not in grpo_owner:
                grpo_owner[rid] = "frontier"
                grpo_rows.append(row)

        per_source["frontier"][task] = sum(
            1 for r in chosen if str(r.get("id") or "") in sft_owner
            and sft_owner[str(r.get("id") or "")] == "frontier"
        )

        # ---- plain A-slice rows (SFT only, never re-prompted) -------------- #
        a_rows: list[dict] = []
        if a_dir is not None:
            for path in _discover(a_dir, task):
                input_hashes[_rel(path)] = _sha256_file(path)
                a_rows.extend(_read_jsonl(path))
            a_stats["scanned"][task] = len(a_rows)
            a_dedupe = {"deduplicated_a_slice": 0}
            a_rows = _dedupe(a_rows, "a_slice", a_dedupe)
            a_stats["deduplicated"] += a_dedupe["deduplicated_a_slice"]
            plain = [r for r in a_rows
                     if (r.get("prompt_variant") in (None, "", "plain")
                         and r.get("harness") in (None, "", "none")
                         and not is_harness_prompt(r))]
            a_stats["rejected_non_plain"] += len(a_rows) - len(plain)
            if len(plain) != len(a_rows):
                raise SystemExit(
                    f"--a-slice must contain plain single-turn rows: {len(a_rows) - len(plain)} "
                    f"{task} row(s) carry a harness/tool prompt (e.g. "
                    f"{[str(r.get('id')) for r in a_rows if r not in plain][:3]})")
            a_rows = plain
            if a_slice_ratio > 0:
                cap_a = round(a_slice_ratio * len(dsft))
                if len(a_rows) > cap_a:
                    shuffled_a = list(a_rows)
                    rng.shuffle(shuffled_a)
                    a_stats["skipped_over_cap"][task] = len(a_rows) - cap_a
                    a_rows = shuffled_a[:cap_a]
            for row in a_rows:
                rid = str(row.get("id") or "")
                if rid in sft_owner and sft_owner[rid] != "a_slice":
                    collisions.append({"id": rid, "stream": "sft",
                                       "sources": [sft_owner[rid], "a_slice"]})
                if rid not in sft_owner:
                    sft_owner[rid] = "a_slice"
                    out = dict(row)
                    out["prompt_variant"] = "plain"
                    out["harness"] = "none"
                    # A few upstream A-slice rows are labelled `frontier` because
                    # their *plant* came from a hosted run. In this mix they are
                    # plain single-turn ground-truth paraphrases, not frontier tool
                    # traces, so use the stream's source label and keep the original
                    # in provenance rather than dropping the row.
                    if out.get("source") == "frontier":
                        out["source"] = "ground_truth"
                        prov = dict(out.get("provenance") or {})
                        prov["a_slice_source"] = "frontier"
                        out["provenance"] = prov
                    sft_rows.append(out)
                    a_stats["selected_ids"].append(rid)
            per_source["a_slice"][task] = len(a_rows)
            a_stats["selected"][task] = len(a_rows)

        # Count by the ACTUAL per-row `source`, not by input directory: the v2
        # deterministic corpora already contain the frontier traces, so a mix that
        # passes `--frontier data/frontier` (zero tool rows) still has frontier rows
        # in its output. Counting by input dir used to report frontier=0 wrongly.
        def _actual(rows: list[dict], owners: dict[str, str]) -> collections.Counter:
            out: collections.Counter = collections.Counter()
            for row in rows:
                src = row.get("source")
                if src not in ("deterministic_tools", "frontier_tools", "ground_truth"):
                    src = {"frontier": "frontier_tools",
                           "a_slice": "ground_truth"}.get(
                               owners.get(str(row.get("id") or "")), "deterministic_tools")
                out[src] += 1
            return out

        sft_actual = _actual([r for r in sft_rows if r.get("task") == task], sft_owner)
        grpo_actual = _actual([r for r in grpo_rows if r.get("task") == task], grpo_owner)
        stream_counts["sft"][task] = {
            "deterministic": sft_actual["deterministic_tools"],
            "frontier": sft_actual["frontier_tools"],
            "total": sum(sft_actual.values()),
        }
        if a_dir is not None:
            stream_counts["sft"][task]["plain_a"] = sft_actual["ground_truth"]
        stream_counts["grpo"][task] = {
            "deterministic": grpo_actual["deterministic_tools"],
            "frontier": grpo_actual["frontier_tools"],
            "total": sum(grpo_actual.values()),
        }

    # Sum the frontier rows actually appended (dedup across tasks would be a
    # collision, already rejected above, so this equals the sum of per-task).
    frontier_stats["selected_count"] = len(
        [1 for rid, owner in sft_owner.items() if owner == "frontier"])
    frontier_stats["selected_ids"] = sorted(
        rid for rid, owner in sft_owner.items() if owner == "frontier")

    # Reza's directive (DECISIONS section 4): the FINAL training mix must carry a
    # real `tier` column in EXACT equal thirds per task. The trainer consumes rows,
    # not plant groups, so the final mix is tiered per emitted row (the deterministic
    # corpora themselves keep their documented grouped tiering). Crucially SFT and
    # GRPO are tiered as SEPARATE streams: each written file must independently have
    # floor(n/3)/floor(n/3)/remainder-to-hard counts, which tiering their union
    # together does not guarantee.
    from thermo_agent.synth import tiers as tier_mod

    def _tier_stream(task: str, stream_rows: list[dict]):
        scoped = [r for r in stream_rows if r.get("task") == task]
        # `assign_tiers` keys its result by record_id; several emitted rows share a
        # plant's record_id (variants) and frontier/A ids carry a `#v...` suffix.
        # Use unique, #v-free proxy ids so every row gets its own tier and
        # `tier_ranges` can look it up again.
        proxies = []
        for i, row in enumerate(scoped):
            proxy = dict(row)
            proxy["record_id"] = f"row{i:06d}"
            proxies.append(proxy)
        mapping = tier_mod.assign_tiers(task, proxies)
        for row, proxy in zip(scoped, proxies):
            if proxy["record_id"] in mapping:
                row["tier"] = mapping[proxy["record_id"]]
        return tier_mod.tier_ranges(task, proxies, mapping)

    tier_summary: dict[str, dict] = {}
    for task in tasks:
        sft_ranges = _tier_stream(task, sft_rows)
        grpo_ranges = _tier_stream(task, grpo_rows)
        tier_summary[task] = {
            "counts": {t: sft_ranges[t]["n"] for t in tier_mod.TIERS},
            "counts_sft": {t: sft_ranges[t]["n"] for t in tier_mod.TIERS},
            "counts_grpo": {t: grpo_ranges[t]["n"] for t in tier_mod.TIERS},
            "ranges": {t: sft_ranges[t] for t in tier_mod.TIERS},
            "ranges_sft": {t: sft_ranges[t] for t in tier_mod.TIERS},
            "ranges_grpo": {t: grpo_ranges[t] for t in tier_mod.TIERS},
            "unscored": sft_ranges["unscored"] + grpo_ranges["unscored"],
            "rule": "exact equal thirds per emitted row by documented per-task "
                    "component-count measure, sha256(row id) tie-break, remainder "
                    "to hard; SFT and GRPO tiered as separate streams; "
                    "deterministic, frontier and plain-A rows tiered together",
        }

    def _provenance_counts(rows: list[dict]) -> dict:
        out: dict[str, int] = {}
        for row in rows:
            family = (row.get("provenance") or {}).get("source_family") or (
                "ai_generated" if row.get("source") == "frontier_tools" else "deterministic")
            model = (row.get("provenance") or {}).get("model") or row.get("model") or "none"
            out[f"{family}:{model}"] = out.get(f"{family}:{model}", 0) + 1
        return dict(sorted(out.items()))

    provenance_summary = {
        "sft": _provenance_counts(sft_rows),
        "grpo": _provenance_counts(grpo_rows),
    }

    return {
        "tasks": tuple(tasks), "seed": seed,
        "sft_rows": sft_rows, "grpo_rows": grpo_rows,
        "tiers": tier_summary, "provenance": provenance_summary,
        "counts": stream_counts, "per_source": per_source,
        "frontier": frontier_stats, "dedupe": dedupe_stats, "a_slice": a_stats,
        "collisions": collisions, "input_hashes": input_hashes,
        "inputs_missing": inputs_missing,
        "frontier_per_task": frontier_per_task, "frontier_ratio": frontier_ratio,
        "harness_prompt": harness_prompt, "a_slice_ratio": a_slice_ratio,
        "dirs": {"deterministic_sft": det_sft_dir, "deterministic_grpo": det_grpo_dir,
                 "frontier": frontier_dir, "a_slice": a_dir},
    }


# --------------------------------------------------------------------------- #
# stats / manifest / writing
# --------------------------------------------------------------------------- #
def print_stats(plan: dict, stream=None) -> None:
    """Print the per-task table (no writes). With --a-slice the three SFT
    sources (tool-deterministic, tool-frontier, plain-A) are shown per task."""
    if stream is None:
        stream = sys.stdout
    tasks = plan["tasks"]
    a_on = plan.get("a_slice", {}).get("enabled", False)
    header = (f"{'task':<10} {'det-sft':>8} {'det-grpo':>9} {'frontier':>9} "
              + (f"{'plain-A':>9} " if a_on else "")
              + f"{'sft-total':>10} {'grpo-total':>11}")
    print(header, file=stream)
    for task in tasks:
        s = plan["counts"]["sft"][task]
        g = plan["counts"]["grpo"][task]
        line = (f"{task:<10} {s['deterministic']:>8} {g['deterministic']:>9} "
                f"{s['frontier']:>9}")
        if a_on:
            line += f"{s.get('plain_a', 0):>9} "
        print(line + f"{s['total']:>10} {g['total']:>11}", file=stream)
    dsft = sum(plan["counts"]["sft"][t]["deterministic"] for t in tasks)
    dgrpo = sum(plan["counts"]["grpo"][t]["deterministic"] for t in tasks)
    fsel = sum(plan["counts"]["sft"][t]["frontier"] for t in tasks)
    asel = sum(plan["counts"]["sft"][t].get("plain_a", 0) for t in tasks)
    total_line = (f"{'TOTAL':<10} {dsft:>8} {dgrpo:>9} {fsel:>9}"
                  + (f"{asel:>9} " if a_on else "")
                  + f"{dsft + fsel + asel:>10} {dgrpo + fsel:>11}")
    print(total_line, file=stream)
    print(f"seed={plan['seed']} frontier cap mode={plan['frontier']['cap_mode']} "
          f"(per-task={plan['frontier_per_task']}, ratio={plan['frontier_ratio']})",
          file=stream)
    if plan.get("tiers"):
        for task in tasks:
            for label, key in (("tiers-sft", "counts_sft"), ("tiers-grpo", "counts_grpo")):
                counts = plan["tiers"][task].get(key, plan["tiers"][task]["counts"])
                print(f"{label} {task:<8} easy={counts['easy']:>5} medium={counts['medium']:>5} "
                      f"hard={counts['hard']:>5}", file=stream)
    if a_on:
        print(f"plain A-slice: {asel} row(s) from {plan['a_slice']['dir']} "
              f"(ratio={plan['a_slice']['ratio']})", file=stream)


def _loud_frontier_warning(plan: dict) -> None:
    f = plan["frontier"]
    if f["selected_count"]:
        print(f"Selected {f['selected_count']} verifier-accepted frontier tool row(s): "
              f"{f['selected_ids']}", file=sys.stderr)
        return
    suffix = ("The tool part of the output is deterministic-only."
              if plan.get("a_slice", {}).get("enabled") else
              "The output is a deterministic-only corpus.")
    print("WARNING: 0 verifier-accepted frontier rows containing BOTH <tool_call> "
          "and <tool_result> were selected "
          f"(scanned={f['scanned']}, accepted={f['accepted']}, "
          f"with tool turns={f['with_tool_turns']}). " + suffix, file=sys.stderr)


def _check_destinations(out_sft: pathlib.Path, out_grpo: pathlib.Path,
                        inputs: dict[str, pathlib.Path], manifest: pathlib.Path) -> None:
    """Refuse to write into protected corpora, input dirs, or the repo/data root."""
    targets: dict[str, pathlib.Path] = {}
    for name in PROTECTED:
        targets[f"protected:{name}"] = _resolve(name)
    for name, path in inputs.items():
        targets[f"input:{name}"] = path
    for label, path in targets.items():
        resolved = path.resolve()
        for dest in (out_sft.resolve(), out_grpo.resolve(), manifest.resolve()):
            if dest == resolved:
                raise SystemExit(f"refusing to write into {label}: {path}")
    for dest in (out_sft.resolve(), out_grpo.resolve()):
        if dest in (ROOT.resolve(), (ROOT / "data").resolve()):
            raise SystemExit(f"refusing to write into the repo/data root: {dest}")


def _owned_from_manifest(manifest: pathlib.Path, out_sft: pathlib.Path,
                         out_grpo: pathlib.Path) -> set[pathlib.Path]:
    """Files this script previously wrote, as recorded in an existing manifest."""
    if not manifest.is_file():
        for directory in (out_sft, out_grpo):
            if directory.is_dir() and any(directory.iterdir()):
                raise SystemExit(
                    f"refusing to reuse existing output dir {directory}: no manifest "
                    f"at {manifest}; use fresh dirs or --force with a manifest written "
                    f"by {SCRIPT_TAG}")
        return set()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"cannot read existing manifest {manifest}: {exc}") from exc
    if not isinstance(data, dict) or data.get("generator") != SCRIPT_TAG:
        raise SystemExit(f"refusing --force: {manifest} was not written by {SCRIPT_TAG}")
    owned = set()
    for value in data.get("owned_files", []):
        owned.add(_resolve(value).resolve())
    return owned


def prepare_dirs(out_sft: pathlib.Path, out_grpo: pathlib.Path,
                 manifest: pathlib.Path, force: bool) -> None:
    """Create the two output dirs, never silently overwriting foreign files."""
    if not force:
        for directory in (out_sft, out_grpo):
            if directory.exists():
                raise SystemExit(
                    f"output dir already exists: {directory}; choose a fresh path or "
                    f"pass --force to replace only files owned by {SCRIPT_TAG}")
        out_sft.mkdir(parents=True, exist_ok=False)
        out_grpo.mkdir(parents=True, exist_ok=False)
        return

    owned = _owned_from_manifest(manifest, out_sft, out_grpo)
    for directory in (out_sft, out_grpo):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.resolve() not in owned:
                raise SystemExit(
                    f"refusing --force: unrecognized file {path} in output dir "
                    f"{directory} (not listed in {manifest})")
    for path in owned:
        if path.is_file():
            path.unlink()
    out_sft.mkdir(parents=True, exist_ok=True)
    out_grpo.mkdir(parents=True, exist_ok=True)


def _group_by_file(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        task = str(row.get("task") or "")
        split = str(row.get("split") or "train")
        grouped.setdefault(f"{task}_{split}.jsonl", []).append(row)
    return grouped


def write_mix(plan: dict, out_sft: pathlib.Path, out_grpo: pathlib.Path,
              manifest_path: pathlib.Path) -> dict:
    """Write the two corpora plus the manifest; return the manifest dict."""
    owned: list[str] = []
    sft_files = _group_by_file(plan["sft_rows"])
    grpo_files = _group_by_file(plan["grpo_rows"])
    for name in sorted(set(sft_files) | set(grpo_files)):
        for directory, grouped in ((out_sft, sft_files), (out_grpo, grpo_files)):
            path = directory / name
            with path.open("w", encoding="utf-8") as fh:
                for row in grouped.get(name, []):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            owned.append(_rel(path))

    manifest = {
        "generator": SCRIPT_TAG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "deterministic_sft": _rel(plan["dirs"]["deterministic_sft"]),
            "deterministic_grpo": _rel(plan["dirs"]["deterministic_grpo"]),
            "frontier": _rel(plan["dirs"]["frontier"]),
            "out_sft": _rel(out_sft), "out_grpo": _rel(out_grpo),
            "tasks": list(plan["tasks"]), "seed": plan["seed"],
            "frontier_per_task": plan["frontier_per_task"],
            "frontier_ratio": plan["frontier_ratio"],
            "a_slice": plan["a_slice"]["dir"],
            "a_slice_ratio": plan["a_slice_ratio"],
            "prompt_variant": "harness_tools" if plan["harness_prompt"] else "plain",
            "cap_precedence": "--frontier-per-task wins when > 0; otherwise "
                              "--frontier-ratio; 0/0 means no cap",
        },
        "totals": {"sft": len(plan["sft_rows"]), "grpo": len(plan["grpo_rows"])},
        "counts": plan["counts"],
        "per_source": plan["per_source"],
        "tiers": plan["tiers"],
        "provenance": plan["provenance"],
        "frontier": plan["frontier"],
        "a_slice": plan["a_slice"],
        "frontier_reprompted": plan["frontier"]["reprompted"],
        "deduplicated": plan["dedupe"],
        "input_file_hashes": plan["input_hashes"],
        "inputs_missing": plan["inputs_missing"],
        "seed": plan["seed"],
        "owned_files": owned + [_rel(manifest_path)],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deterministic-sft", default="data/sft_tools",
                    help="input deterministic SFT corpus dir (default: data/sft_tools)")
    ap.add_argument("--deterministic-grpo", default="data/grpo_tools",
                    help="input deterministic GRPO corpus dir (default: data/grpo_tools)")
    ap.add_argument("--frontier", default="data/frontier",
                    help="input teacher frontier dir (default: data/frontier)")
    ap.add_argument("--out-sft", default="data/sft_mix",
                    help="new SFT mix output dir (default: data/sft_mix)")
    ap.add_argument("--out-grpo", default="data/grpo_mix",
                    help="new GRPO mix output dir (default: data/grpo_mix)")
    ap.add_argument("--frontier-per-task", type=int, default=0,
                    help="max accepted tool-using frontier rows per task; 0 = no cap "
                         "(default: 0). Takes precedence over --frontier-ratio.")
    ap.add_argument("--frontier-ratio", type=float, default=0.0,
                    help="alternative cap: fraction of deterministic SFT rows per task; "
                         "used only when --frontier-per-task is 0 (default: 0)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS),
                    help=f"tasks to mix (default: {' '.join(TASKS)})")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the capped frontier selection shuffle (default: 0)")
    ap.add_argument("--manifest", default="data/sft_mix/MANIFEST.json",
                    help="manifest path (default: data/sft_mix/MANIFEST.json)")
    ap.add_argument("--stats", action="store_true",
                    help="print the per-task table and exit without writing anything")
    ap.add_argument("--force", action="store_true",
                    help="reuse existing output dirs, deleting only files listed in an "
                         "existing manifest written by this script; foreign files abort")
    ap.add_argument("--harness-prompt", action="store_true",
                    help="build the ours_harness mix: deterministic inputs must be the "
                         "*_h corpora and every selected frontier row is re-prompted with "
                         "the shared harness system prompt (completion untouched)")
    ap.add_argument("--a-slice", default=None,
                    help="dir of plain SFT rows from scripts/build_a_slice.py; those rows "
                         "are added to SFT only, keeping prompt_variant=plain")
    ap.add_argument("--a-slice-ratio", type=float, default=0.0,
                    help="cap A rows per task to round(ratio * deterministic rows); "
                         "0 (default) takes every row in --a-slice")
    args = ap.parse_args(argv)

    tasks = tuple(t.strip() for t in args.tasks if t.strip())
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        ap.error(f"unknown task(s): {', '.join(unknown)}; choose from {' '.join(TASKS)}")
    if args.frontier_per_task < 0:
        ap.error("--frontier-per-task must be >= 0")
    if args.frontier_ratio < 0:
        ap.error("--frontier-ratio must be >= 0")
    if args.a_slice_ratio < 0:
        ap.error("--a-slice-ratio must be >= 0")

    out_sft, out_grpo = _resolve(args.out_sft), _resolve(args.out_grpo)
    manifest_path = _resolve(args.manifest)
    inputs = {"deterministic_sft": _resolve(args.deterministic_sft),
              "deterministic_grpo": _resolve(args.deterministic_grpo),
              "frontier": _resolve(args.frontier)}
    if args.a_slice is not None:
        inputs["a_slice"] = _resolve(args.a_slice)

    if not args.stats:
        _check_destinations(out_sft, out_grpo, inputs, manifest_path)

    plan = build_mix(args.deterministic_sft, args.deterministic_grpo, args.frontier,
                     tasks=tasks, seed=args.seed,
                     frontier_per_task=args.frontier_per_task,
                     frontier_ratio=args.frontier_ratio,
                     harness_prompt=args.harness_prompt,
                     a_slice=args.a_slice, a_slice_ratio=args.a_slice_ratio)

    if plan["collisions"]:
        raise SystemExit(f"id collision between sources: {plan['collisions'][:5]}")

    print_stats(plan)
    _loud_frontier_warning(plan)
    if plan["inputs_missing"]:
        print(f"WARNING: missing input dirs: {plan['inputs_missing']}", file=sys.stderr)

    if args.stats:
        return 0

    prepare_dirs(out_sft, out_grpo, manifest_path, args.force)
    manifest = write_mix(plan, out_sft, out_grpo, manifest_path)
    print(json.dumps({
        "totals": manifest["totals"], "counts": manifest["counts"],
        "frontier_selected": manifest["frontier"]["selected_count"],
        "frontier_selected_ids": manifest["frontier"]["selected_ids"],
        "plain_a_selected": manifest["a_slice"]["selected"],
        "input_files_hashed": len(manifest["input_file_hashes"]),
        "manifest": _rel(manifest_path),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
