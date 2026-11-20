"""Contract tests for the thermodynamics + component skills library.

Every ``SKILL.md`` is checked for parseable frontmatter, consistent metadata and
size.  The stricter body contract (Objective: plus at least three ``##``
sections) applies to the skills added by this library; the eight pre-existing
skills were written in an older format and are deliberately left untouched, so
they are only required to keep parsing and to keep their metadata valid.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from thermo_agent.core.components import REQUIRED_PORTS

SKILLS_DIR = Path(__file__).resolve().parents[1] / "thermo_agent" / "agent" / "skills"
VALID_TASKS = {"repair", "design", "audit", "ledger"}

# The 16 component skills, 5 cross-cutting skills and the tool-guide skill.
# Three of these names (gas-engine, absorption-chiller, heat-rejector) already
# existed and are intentionally not overwritten; the library grows by addition.
REQUIRED_NAMES = {
    "compressor", "turbine", "pump", "combustor", "recuperator", "hrsg", "orc",
    "condenser", "evaporator", "heat-rejector", "electric-chiller",
    "absorption-chiller", "heat-pump", "gas-engine", "dc-load", "district-heat",
    "thermo-properties", "exergy-accounting", "pinch-analysis",
    "repair-diagnosis", "design-selection", "thermo-tools",
}
PREEXISTING_NAMES = {
    "absorption-chiller", "design-catalog", "exergy-audit", "exergy-ledger",
    "gas-engine", "heat-pump-district-heat", "heat-rejector", "orc-bottoming",
}
NEW_NAMES = REQUIRED_NAMES - PREEXISTING_NAMES

_FM = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def _parse(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    m = _FM.match(text)
    assert m, f"{path}: missing --- delimited frontmatter"
    raw: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if line.strip():
            key, _, value = line.partition(":")
            raw[key.strip()] = value.strip()
    return raw, text[m.end():]


def _list_field(raw: dict[str, str], key: str) -> list[str] | None:
    if key not in raw:
        return None
    value = raw[key].strip()
    assert value.startswith("[") and value.endswith("]"), f"{key} must be a bracketed list"
    items = [x.strip() for x in value[1:-1].split(",") if x.strip()]
    assert all(isinstance(x, str) for x in items), f"{key} entries must be strings"
    return items


FILES = _skill_files()
IDS = [p.parent.name for p in FILES]


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_frontmatter_metadata_and_size(path):
    """Every skill, new or old, parses and carries valid metadata."""
    raw, _ = _parse(path)
    assert raw.get("name") == path.parent.name
    assert len(raw.get("description", "")) >= 20
    assert len(path.read_text(encoding="utf-8")) < 6000


@pytest.mark.parametrize("name", sorted(NEW_NAMES))
def test_new_skill_body_format(name):
    """The new library skills follow the documented body contract and length."""
    path = SKILLS_DIR / name / "SKILL.md"
    assert path.is_file(), f"missing new skill {name}"
    _, body = _parse(path)
    assert "Objective:" in body
    assert len(re.findall(r"^## ", body, re.M)) >= 3
    assert 25 <= len(path.read_text(encoding="utf-8").splitlines()) <= 45


def test_required_names_exist():
    present = {p.parent.name for p in FILES}
    assert REQUIRED_NAMES <= present


def test_existing_skills_still_parse():
    """The eight original skills survive unchanged and still parse."""
    present = {p.parent.name for p in FILES}
    assert PREEXISTING_NAMES <= present
    for name in PREEXISTING_NAMES:
        raw, _ = _parse(SKILLS_DIR / name / "SKILL.md")
        assert raw.get("name") == name


def test_library_is_not_pinned_to_a_fixed_count():
    assert len(FILES) >= len(REQUIRED_NAMES)


def test_no_duplicate_names():
    names = [_parse(p)[0].get("name") for p in FILES]
    assert len(names) == len(set(names))


def test_every_component_type_is_covered():
    covered: set[str] = set()
    for path in FILES:
        covered.update(_list_field(_parse(path)[0], "components") or [])
    assert set(REQUIRED_PORTS) <= covered


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_list_fields_and_task_values(path):
    raw, _ = _parse(path)
    comps = _list_field(raw, "components")
    if comps is not None:
        assert comps, "components: must not be empty"
        assert set(comps) <= set(REQUIRED_PORTS), "no invented component types"
    variants = _list_field(raw, "variants")
    if variants is not None:
        assert all(v for v in variants)
    task = raw.get("task")
    if task is not None:
        assert task in VALID_TASKS


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_no_offset_temperature_units(path):
    text = path.read_text(encoding="utf-8")
    assert "degC" not in text
    assert "Celsius" not in text
