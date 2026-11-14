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
    """Component skills whose components appear in the plant (all of them when there is no plant yet, as in Design),
    plus the task's own skills."""
    out = list(TASK_SKILLS.get(task, []))
    for n in skill_names():
        comps = frontmatter((ROOT / "skills" / n / "SKILL.md").read_text()).get("components")
        if not comps: continue
        if plant is None or any(c.type in comps for c in plant.components): out.append(n)
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
