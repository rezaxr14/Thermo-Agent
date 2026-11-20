"""J2/J3 gate: one compact harness text, <= 2,500 real Qwen3.5 tokens, for every arm.

The harness block is the text ``build_harness_tool_messages`` appends to the
system message (compact AGENTS.md + the one selected skill + the five-tool
contract). Every arm -- tracegen, the ``*_h`` training corpora and eval/GRPO --
builds it through that one function, so the same (task, plant) input must always
produce the same text (and the same hash).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from thermo_agent.agent import harness as H
from thermo_agent.agent.tool_prompt import (
    FINAL_TAGS,
    build_compact_harness_messages,
    build_harness_tool_messages,
    compact_tool_contract,
)
from thermo_agent.core.schema import Plant
from thermo_agent.synth.combine import tokenizer
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES
from thermo_agent.teacher.toolkit import make_tools

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("repair", "design", "audit", "ledger")
TOOLS = ("inspect_plant", "first_law_check", "second_law_check",
         "validity_check", "solve_plant")
TOKEN_CEILING = 2500


def _row(task: str) -> dict:
    path = ROOT / "data" / "processed" / f"{task}_train.jsonl"
    if not path.is_file():
        pytest.skip(f"no real training row at {path}")
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def _plant(row: dict) -> Plant | None:
    for key in ("plant_broken", "plant", "plant_original"):
        if row.get(key):
            try:
                return Plant.model_validate_json(row[key])
            except Exception:
                pass
    return None


def _system(task: str, compact: bool = True) -> str:
    row = _row(task)
    msgs = build_harness_tool_messages(list(row["prompt"]), task, _plant(row), compact=compact)
    return msgs[0]["content"]


def _block(task: str, compact: bool = True) -> str:
    """Just the appended harness block, without the row's original system text."""
    row = _row(task)
    sysmsg = _system(task, compact=compact)
    base = row["prompt"][0]["content"]
    return sysmsg[len(base):] if sysmsg.startswith(base) else sysmsg


def _tok():
    tok = tokenizer()
    if tok is None:
        pytest.skip("no local Qwen3.5 tokenizer: set THERMO_TOKENIZER or cache "
                    "~/.cache/thermo/qwen3.5-4b/tokenizer.json (never silently pass)")
    return tok


def _ntok(tok, text: str) -> int:
    return len(tok.encode(text).ids)


# --------------------------------------------------------------------------- #
# 1. the hard gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_compact_harness_token_ceiling(task):
    tok = _tok()
    block = _block(task)
    n = _ntok(tok, block)
    assert n <= TOKEN_CEILING, (
        f"{task}: compact harness is {n} tokens (chars={len(block)}), "
        f"over the {TOKEN_CEILING}-token ceiling")


@pytest.mark.parametrize("task", TASKS)
def test_compact_harness_char_report(task):
    """A cheap, always-on sanity bound (and a char figure to report)."""
    block = _block(task)
    assert 0 < len(block) < 9000, f"{task}: unexpected compact harness size {len(block)} chars"


# --------------------------------------------------------------------------- #
# 2. one text for every arm -> one hash per (task, plant)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_single_shared_hash_for_task_and_plant(task):
    row = _row(task)
    plant = _plant(row)
    prompt = list(row["prompt"])

    # training / eval tool arm: the shared entry point (compact by default)
    via_entry = build_harness_tool_messages(prompt, task, plant)[0]["content"]
    # the compact builder itself (what tracegen and the Z arm route through)
    via_compact = build_compact_harness_messages(prompt, task, plant)[0]["content"]
    # explicit default limits match the builder's defaults
    via_explicit = build_harness_tool_messages(prompt, task, plant, max_turns=6, max_tool_calls=8)[0]["content"]

    digest = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    assert digest(via_entry) == digest(via_compact) == digest(via_explicit), (
        f"{task}: arms disagree on the compact harness text")

    # the verbosity toggle really produces a different (old) text.
    assert digest(_system(task, compact=False)) != digest(via_entry)


# --------------------------------------------------------------------------- #
# 3. content contract (no tool semantics changed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_five_tools_present(task):
    system = _system(task)
    for name in TOOLS:
        assert name in system, f"{task}: missing tool {name}"


@pytest.mark.parametrize("task", TASKS)
def test_final_tags_present(task):
    system = _system(task)
    for tag in FINAL_TAGS[task]:
        assert tag in system, f"{task}: missing final tag {tag}"


@pytest.mark.parametrize("task", TASKS)
def test_no_non_selected_skills(task):
    system = _system(task)
    selected = set(H.TASK_SKILLS.get(task, []))
    for name in H.skill_names():
        if name in selected:
            assert f"SKILL {name}" in system, f"{task}: selected skill {name} missing"
        else:
            assert f"SKILL {name}" not in system, f"{task}: non-selected skill {name} leaked"


def test_tool_call_and_result_format_unchanged():
    system = _system("repair")
    assert '<tool_call name="TOOL_NAME">' in system
    assert '{"candidate": "..."}' in system
    assert "<tool_result>" in system and "</tool_result>" in system
    assert "answer-oracle" in system


# --------------------------------------------------------------------------- #
# 4. opt-in dynamic tools: named only when included, still under the token gate
# --------------------------------------------------------------------------- #
def _dynamic_system(task: str) -> str:
    row = _row(task)
    tools = make_tools(task, row, include_dynamic=True)
    return build_harness_tool_messages(list(row["prompt"]), task, _plant(row),
                                        tools=tools)[0]["content"]


@pytest.mark.parametrize("task", TASKS)
def test_dynamic_tools_named_only_when_included(task):
    # Test the contract itself: the row/skill text may legitimately mention a
    # component (e.g. "combustor"), so the whole harness block is too coarse.
    row = _row(task)
    default = compact_tool_contract(task, 6, 8)
    assert all(name not in default for name in DYNAMIC_TOOL_NAMES)
    dynamic = compact_tool_contract(
        task, 6, 8, tools=make_tools(task, row, include_dynamic=True))
    assert all(name in dynamic for name in DYNAMIC_TOOL_NAMES)
    assert "JSON object string" in dynamic


@pytest.mark.parametrize("task", TASKS)
def test_dynamic_compact_harness_stays_under_token_ceiling(task):
    tok = _tok()
    row = _row(task)
    base = row["prompt"][0]["content"]
    system = _dynamic_system(task)
    block = system[len(base):] if system.startswith(base) else system
    n = _ntok(tok, block)
    assert n <= TOKEN_CEILING, f"{task}: dynamic compact harness is {n} tokens"
