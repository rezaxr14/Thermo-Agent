"""Harness text for the ablation: AGENTS.md, and skills selected by the components present in the plant.
Modes: none | agents | skills | agents+skills."""
from __future__ import annotations
import re
from pathlib import Path
from ..core.schema import Plant
from .build_harness import frontmatter

ROOT = Path(__file__).parent
MODES = ("none", "agents", "skills", "agents+skills")
TASK_SKILLS = {"design": ["design-catalog"], "audit": ["exergy-audit"], "repair": [], "ledger": ["exergy-ledger"]}


def agents_md() -> str:
    return (ROOT / "AGENTS.md").read_text()


def skill_names() -> list[str]:
    return sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))


def read_skill(name: str) -> str:
    text = (ROOT / "skills" / name / "SKILL.md").read_text()
    return f"SKILL {name}: {frontmatter(text).get('description', '')}\n" + re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)


def skills_for(plant: Plant | None, task: str) -> list[str]:
    """The task's own skill plus the component skills whose components appear in the plant.

    When there is no plant yet (Design), only the task skill is included. Every component skill at once adds
    about 12k characters, which pushes a design or ledger prompt past the 4096-token training limit; the
    design-catalog skill already lists the components and their parameters.
    """
    out = list(TASK_SKILLS.get(task, []))
    if plant is None:
        return out
    for n in skill_names():
        comps = frontmatter((ROOT / "skills" / n / "SKILL.md").read_text()).get("components")
        if not comps: continue
        if any(c.type in comps for c in plant.components): out.append(n)
    return out


def augment(messages: list[dict], mode: str, task: str, plant: Plant | None = None) -> list[dict]:
    """Append harness text to the system message. The input list is not modified."""
    if mode not in MODES: raise ValueError(f"mode must be one of {MODES}")
    if mode == "none": return messages
    parts = []
    if "agents" in mode: parts.append(agents_md())
    if "skills" in mode: parts += [read_skill(n) for n in skills_for(plant, task)]
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
