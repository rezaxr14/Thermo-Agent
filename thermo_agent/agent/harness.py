"""Harness text for the ablation: AGENTS.md, and skills selected by the components present in the plant.
Modes: none | agents | skills | agents+skills."""
from __future__ import annotations
import re
from pathlib import Path
from ..core.schema import Plant
from .build_harness import frontmatter

ROOT = Path(__file__).parent
MODES = ("none", "agents", "skills", "agents+skills")
TASK_SKILLS = {"design": ["design-catalog"], "audit": ["exergy-audit"], "repair": ["repair-diagnosis"], "ledger": ["exergy-ledger"]}

# The verbose harness (``augment`` / ``compact=False``) is a before/after comparison path,
# NOT a production arm: the compact builder in ``tool_prompt`` selects exactly one task
# skill and carries the J2/J3 <= 2,500-token contract. Bound the *rendered* verbose text
# so growing the skill library cannot silently blow its own budget: with 27 skills, a
# plant matching every component skill produced ~31k chars / ~8.3k tokens, over both the
# ~4,096-token SFT budget (see combine.HARNESS_MODES) and the corpus' 22k-char prefilter,
# so every such row was silently dropped instead of trained on. 15k chars (~4k tokens)
# leaves AGENTS.md + the task skill + a few plant-relevant component skills.
VERBOSE_HARNESS_CHARS = 15000


def agents_md() -> str:
    return (ROOT / "AGENTS.md").read_text()


def skill_names() -> list[str]:
    return sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))


def read_skill(name: str) -> str:
    text = (ROOT / "skills" / name / "SKILL.md").read_text()
    return f"SKILL {name}: {frontmatter(text).get('description', '')}\n" + re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)


def skills_for(plant: Plant | None, task: str) -> list[str]:
    """The task's own skill plus the component skills whose components appear in the plant.

    When there is no plant yet (Design), only the task skill is included; the task skill
    is always first so the compact builder keeps serving exactly one skill. Matching
    component skills are ordered deterministically, most-targeted first (fewest entries
    in ``components:`` frontmatter, then by name), so the bounded verbose selection in
    :func:`_bounded_skill_names` keeps the most specific guidance. This function stays
    unbounded: its selection semantics are exercised directly by the tests; only the
    *rendered* verbose text is capped in :func:`augment`.
    """
    out = list(TASK_SKILLS.get(task, []))
    if plant is None:
        return out
    plant_types = {c.type for c in plant.components}
    matched: list[tuple[int, str]] = []
    for n in skill_names():
        comps = frontmatter((ROOT / "skills" / n / "SKILL.md").read_text()).get("components")
        if not comps: continue
        if any(c in plant_types for c in comps): matched.append((len(comps), n))
    out.extend(n for _, n in sorted(matched))
    return out


def _bounded_skill_names(messages: list[dict], mode: str, task: str,
                         plant: Plant | None) -> list[str]:
    """The skill names :func:`augment` will actually render, capped by ``VERBOSE_HARNESS_CHARS``.

    The task skill (first in ``skills_for``) is always kept; component skills are added
    in priority order only while the whole augmented system message still fits. Bounding
    here rather than in ``skills_for`` keeps the two callers honest: the selection API is
    unchanged, and the comparison-only verbose text can never silently exceed its budget
    when the skill library grows.
    """
    names = skills_for(plant, task)
    if not names:
        return []
    used = sum(len(m.get("content", "")) for m in messages)
    n_parts = 1  # the base system message is joined to the first appended part
    if "agents" in mode:
        used += len(agents_md())
        n_parts += 1
    used += 2 * n_parts  # the "\n\n" separators
    budget = VERBOSE_HARNESS_CHARS - used
    kept = [names[0]]                       # the task skill is unconditional
    size = len(read_skill(names[0]))
    for n in names[1:]:
        s = len(read_skill(n))
        if size + s + 2 > budget:
            continue
        kept.append(n)
        size += s + 2
    return kept


def augment(messages: list[dict], mode: str, task: str, plant: Plant | None = None) -> list[dict]:
    """Append harness text to the system message. The input list is not modified.

    The verbose skill block is bounded by :func:`_bounded_skill_names` so it stays inside
    the SFT prompt budget as the skill library grows. The compact production path
    (``tool_prompt.build_compact_harness_messages``) is unaffected.
    """
    if mode not in MODES: raise ValueError(f"mode must be one of {MODES}")
    if mode == "none": return messages
    parts = []
    if "agents" in mode: parts.append(agents_md())
    if "skills" in mode: parts += [read_skill(n) for n in _bounded_skill_names(messages, mode, task, plant)]
    out = [dict(m) for m in messages]
    out[0]["content"] = out[0]["content"] + "\n\n" + "\n\n".join(parts)
    return out

def augment_with_memory(messages: list[dict], mode: str, task: str, plant: Plant | None = None,
                        notes=None, k: int = 3, exclude_record: str = "") -> list[dict]:
    """`augment` plus short success/failure recaps retrieved for this plant.

    Kept separate from `augment` so the harness ablation (`MODES`) is unchanged; memory is an extra layer that
    can be switched on independently in training and evaluation. Only the reasoning-carrying recap is served:
    never the note id, record/id suffix, source model, timestamp, cost/token/latency fields or the raw journal.
    """
    from ..teacher import memory
    from ..teacher.compact_memory import compact_notes
    from .dsl import plant_to_text
    out = augment(messages, mode, task, plant)
    if not notes:
        return out
    # A plant gives the best query; without one (Design, or a text-only caller)
    # fall back to the task text carried in the messages so memory is still served.
    query = (plant_to_text(plant) if plant is not None else
             " ".join(str(m.get("content", "")) for m in messages if m.get("role") != "system"))
    retrieved = memory.retrieve(query, notes, k=k, task=task, exclude_record=exclude_record)
    block = compact_notes(retrieved, k=k)
    if not block:
        return out
    result = [dict(m) for m in out]
    result[0]["content"] = result[0]["content"] + "\n\n" + block
    return result
