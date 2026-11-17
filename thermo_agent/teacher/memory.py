"""Hindsight failure memory.

Every attempt made against the physics tool suite is recorded, accepted or rejected. Rejected attempts become
short notes: the symptom the verifier reported, the mistake class, and the correct pattern from ground truth.
At training and inference time the notes are retrieved by token overlap with the current plant and appended to
the system prompt, so the model reads the failure modes that are plausible for this plant before it answers.

Notes describe *classes* of mistakes, never the answer to the current row: retrieval is by plant tokens and
task, and any note mined from the same record id is excluded.
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
    accepted: bool
    record_id: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict:
        return asdict(self)


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
    return [n for _, n in scored[:k]]


def format_notes(notes: list[Note]) -> str:
    if not notes:
        return ""
    lines = ["KNOWN FAILURE MODES (retrieved from earlier attempts that the verifiers rejected; check each against this plant before answering):"]
    for n in notes:
        fix = f" Correct pattern: {n.fix}" if n.fix else ""
        lines.append(f"- {n.fault_type or n.task}: {n.lesson}{fix}")
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
    """One note per (task, fault type, lesson, fix); keep the first id for determinism."""
    seen, out = set(), []
    for n in sorted(notes, key=lambda x: x.id):
        key = (n.task, n.fault_type, n.lesson, n.fix)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def dedupe_lessons(notes: list[Note]) -> list[Note]:
    """One note per (task, fault type, lesson), ignoring the plant-specific fix.

    Failure memory should teach the class of mistake, not one plant's stream names, so the mined corpus keeps a
    single example per lesson.
    """
    seen, out = set(), []
    for n in sorted(notes, key=lambda x: x.id):
        key = (n.task, n.fault_type, n.lesson)
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
