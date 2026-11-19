"""Hindsight memory: verifier-rejected failure modes AND verifier-accepted working patterns.

Every attempt made against the physics tool suite is recorded, accepted or rejected. Rejected attempts become
short notes: the symptom the verifier reported, the mistake class, and the correct pattern. Accepted attempts
become equally short "what worked" notes. At training and inference time the notes are retrieved by token
overlap with the current plant and appended to the system prompt, so the model sees both the failure modes and
the working patterns that are plausible for this plant before it answers.

Notes describe *classes*, never the answer to the current row: retrieval is by plant tokens and task, and any
note mined from the same record id is excluded. Accepted notes ("OK") can never crowd out the failure notes
("FAIL"): retrieval reserves at least one failure slot whenever failures are relevant.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import re
from dataclasses import asdict, dataclass, field, replace

TOKEN = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_QUERY_WEIGHT = 0.5


@dataclass(frozen=True)
class Note:
    id: str
    task: str
    fault_type: str
    architecture: str
    symptoms: str
    lesson: str
    fix: str
    source: str
    accepted: bool = False  # legacy notes predate this flag; missing => treated as a failure
    record_id: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        return asdict(self)


def _accepted(note: Note) -> bool:
    """Backwards compatible accepted check: legacy rows without the flag are failures."""
    return bool(getattr(note, "accepted", False))


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text or "")}


def note_text(note: Note) -> str:
    return " ".join([note.fault_type, note.architecture, note.symptoms, note.lesson])


def score(query_tokens: set[str], note: Note) -> float:
    """Cosine-like overlap, with a small boost when the query names the fault class or architecture."""
    nt = tokens(note_text(note))
    if not nt or not query_tokens:
        return 0.0
    overlap = len(query_tokens & nt)
    base = overlap / math.sqrt(len(query_tokens) * len(nt))
    if note.fault_type and note.fault_type.lower() in query_tokens:
        base += 0.25
    if note.architecture and note.architecture.lower() in query_tokens:
        base += 0.15
    return base


def _is_seed(note: Note) -> bool:
    """Hand-written method notes (source ``seed:*``) are guidance, not verifier evidence."""
    return bool((getattr(note, "extra", None) or {}).get("seed")) or str(
        getattr(note, "source", "")).startswith("seed:")


def retrieve(query: str, notes: list[Note], k: int = 3, task: str | None = None,
             exclude_record: str = "", min_score: float = 0.02) -> list[Note]:
    q = tokens(query)
    scored = []
    for n in notes:
        if task and n.task != task:
            continue
        if exclude_record and n.record_id == exclude_record:
            continue
        s = score(q, n)
        if s >= min_score:
            scored.append((s, n))
    scored.sort(key=lambda sn: (-sn[0], sn[1].id))
    top = scored[:k]
    # Seed method notes are useful as a recap, but verifier-mined notes are evidence: serve at most
    # ONE seed note per block and backfill its slots with the next-best mined note, so generic
    # guidance cannot crowd out the plant-specific successes/failures the tools actually produced.
    seed_positions = [i for i, (_, n) in enumerate(top) if _is_seed(n)]
    if len(seed_positions) > 1:
        keep_index = seed_positions[0]
        kept = [top[keep_index]]
        mined = [sn for i, sn in enumerate(top) if i != keep_index and not _is_seed(sn[1])]
        backfill = [sn for sn in scored[k:] if not _is_seed(sn[1])]
        top = mined + backfill[:max(0, k - 1 - len(mined))]
        top.append(kept[0])
        top.sort(key=lambda sn: (-sn[0], sn[1].id))
        top = top[:k]
    # Working patterns must not crowd out the failure lessons: reserve at least one slot for the best-ranked
    # failure (min(1, k//2) == 1 for every useful k >= 2) whenever a failure is relevant at all.
    reserved = min(1, k // 2)
    if reserved and len(top) == k and not any(not _accepted(n) for _, n in top):
        for candidate in scored[k:]:
            if not _accepted(candidate[1]):
                top[-1] = candidate
                break
    return [n for _, n in top]


def format_notes(notes: list[Note]) -> str:
    if not notes:
        return ""
    lines = ["KNOWN FAILURE MODES AND WORKING PATTERNS (retrieved from earlier verifier-rejected attempts and "
             "verifier-accepted solutions; check each against this plant before answering):"]
    for n in notes:
        marker = "OK  " if _accepted(n) else "FAIL"
        label = f"{n.task}/{n.fault_type}" if n.fault_type else n.task
        fix = f" Fix: {n.fix}" if n.fix else ""
        lines.append(f"- {marker} {label}: Lesson: {n.lesson}{fix}")
    return "\n".join(lines)


def augment_with_memory(messages: list[dict], task: str, query: str, notes: list[Note],
                        k: int = 3, exclude_record: str = "") -> list[dict]:
    """Append retrieved failure notes to the system message. Returns a new list; the input is not modified."""
    block = format_notes(retrieve(query, notes, k=k, task=task, exclude_record=exclude_record))
    if not block:
        return messages
    out = [dict(m) for m in messages]
    out[0]["content"] = out[0]["content"] + "\n\n" + block
    return out


def save(notes: list[Note], path) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for n in notes:
            fh.write(json.dumps(n.to_dict(), sort_keys=True) + "\n")


def load(path) -> list[Note]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Note(**d))
    return out


def dedupe(notes: list[Note]) -> list[Note]:
    """One note per (task, fault type, lesson, fix, kind); keep the first id for determinism.

    `accepted` is part of the key so a working pattern is never deduplicated away by a failure with identical
    text (or vice versa).
    """
    seen, out = set(), []
    for n in sorted(notes, key=lambda x: x.id):
        key = (n.task, n.fault_type, n.lesson, n.fix, _accepted(n))
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def dedupe_lessons(notes: list[Note]) -> list[Note]:
    """One note per (task, fault type, lesson, kind), ignoring the plant-specific fix.

    Memory teaches classes, not one plant's stream names, so we keep a single example per lesson -- but in two
    separate classes: an accepted note and a rejected note with the same lesson text are both kept.
    """
    seen, out = set(), []
    for n in sorted(notes, key=lambda x: x.id):
        key = (n.task, n.fault_type, n.lesson, _accepted(n))
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def summarize(notes: list[Note], top: int = 12) -> dict:
    """Count the common failure reasons across graded rejections, by task and lesson.

    Written next to the notes so the retrieval layer and the paper can quote how often each mistake occurred,
    instead of only seeing one deduplicated example of it.
    """
    by_task: dict[str, int] = collections.Counter()
    by_lesson: dict[str, int] = collections.Counter()
    by_fault: dict[str, int] = collections.Counter()
    for n in notes:
        by_task[n.task] += 1
        by_lesson[n.lesson] += 1
        by_fault[n.fault_type] += 1
    ranked = sorted(by_lesson.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"n": len(notes), "by_task": dict(by_task), "by_fault_type": dict(by_fault),
            "top_lessons": [{"count": c, "lesson": lesson} for lesson, c in ranked[:top]]}


def frequency_notes(notes: list[Note], top: int = 8) -> list[Note]:
    """A small set of notes whose lesson states how often that failure occurred.

    These are retrieved alongside the concrete examples, so a common mistake is more likely to be shown.
    """
    counts = collections.Counter((n.task, n.lesson) for n in notes)
    out = []
    for (task, lesson), count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:top]:
        if count < 2:
            continue
        sample = next(n for n in notes if n.task == task and n.lesson == lesson)
        out.append(replace(sample, id=f"freq-{task}-{hashlib.md5(lesson.encode()).hexdigest()[:8]}",
                           lesson=f"Seen {count} times: {lesson}", record_id=""))
    return out
