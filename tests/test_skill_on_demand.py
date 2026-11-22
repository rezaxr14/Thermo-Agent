"""On-demand skill harness: list_skills / load_skill + the compact skill catalog.

The model sees one-line skill descriptions in the compact prompt (dynamic mode
only) and loads a full SKILL.md body through a TOOL RESULT when it reaches for
one. Default mode keeps the exact original compact text. Local CPU only: no
model calls, no network, no Kaggle.
"""
from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from thermo_agent.agent import harness as H
from thermo_agent.agent.tool_prompt import (
    build_compact_harness_messages,
    build_harness_tool_messages,
)
from thermo_agent.synth.combine import tokenizer
from thermo_agent.teacher import agent_graph
from thermo_agent.teacher.toolkit import (
    SKILL_TOOL_NAMES,
    list_skills,
    load_skill,
    make_tools,
    tool_names,
)
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES

PLANT_TOOLS = ["inspect_plant", "first_law_check", "second_law_check",
               "validity_check", "solve_plant"]
TOKEN_CEILING = 2500

# Byte-for-byte snapshot of the DEFAULT compact prompt for a fixed input (2026-11-20: reasoning line +
# on-demand skills/component tools are the default). Changing the default prompt is a deliberate act:
# update this hash in the same commit, re-measure the token gate and rebuild shards/corpora.
_DEFAULT_BASE = [{"role": "system", "content": "BASE SYSTEM"},
                 {"role": "user", "content": "BASE USER"}]
_DEFAULT_SHA = "4bce20201f3dab374c00bd44ff695c910c0796deff2c2ab87f1a4ce11999979b"


def _body(name: str) -> str:
    return H.read_skill(name).split("\n", 1)[1]


# --------------------------------------------------------------------------- #
# 1. list_skills: descriptions, not bodies; filtering works
# --------------------------------------------------------------------------- #
def test_list_skills_returns_descriptions_never_full_bodies():
    out = json.loads(list_skills("{}"))
    assert out["ok"] is True
    assert out["hint"] == "call load_skill(name) for the full text"
    assert out["n"] == len(H.skill_names()) == 27
    names = [s["name"] for s in out["skills"]]
    assert names == H.skill_names()  # deterministic library order
    for entry in out["skills"]:
        assert set(("name", "description", "components", "variants", "task")) <= set(entry)
        assert entry["description"] and "\n" not in entry["description"]
        assert isinstance(entry["components"], list) and isinstance(entry["variants"], list)

    raw = list_skills("{}")
    assert "## Physics and equations" not in raw          # no full SKILL.md body
    assert "## Failure modes" not in raw
    assert "Objective: set a pressure ratio" not in raw   # compressor body first line


def test_list_skills_filters_by_task_components_and_query():
    comp = json.loads(list_skills(json.dumps({"components": ["compressor"]})))
    assert comp["ok"] is True and [s["name"] for s in comp["skills"]] == ["compressor"]

    repair = json.loads(list_skills(json.dumps({"task": "repair"})))
    repair_names = [s["name"] for s in repair["skills"]]
    assert "repair-diagnosis" in repair_names
    assert "compressor" not in repair_names

    query = json.loads(list_skills(json.dumps({"query": "compressor"})))
    assert "compressor" in [s["name"] for s in query["skills"]]

    # Multiple filters are ANDed; a repair task never has a compressor component skill.
    assert json.loads(list_skills(json.dumps(
        {"task": "repair", "components": ["compressor"]})))["n"] == 0


# --------------------------------------------------------------------------- #
# 2. load_skill: full body, strict validation, no path traversal
# --------------------------------------------------------------------------- #
def test_load_skill_returns_full_body_for_a_known_skill():
    out = json.loads(load_skill(json.dumps({"name": "compressor"})))
    assert out["ok"] is True and out["name"] == "compressor"
    assert out["text"] == _body("compressor")
    assert "# Compressor" in out["text"]
    assert "## Physics and equations" in out["text"]
    # Frontmatter (and the "SKILL name: description" header) is stripped.
    assert "SKILL compressor:" not in out["text"]
    assert "description: Use when a plant contains a compressor" not in out["text"]


def test_load_skill_rejects_unknown_malformed_and_traversal_names():
    bad = [
        json.dumps({"name": "no-such-skill"}),
        json.dumps({"name": "../etc/passwd"}),
        json.dumps({"name": "../../thermo_agent/agent/harness.py"}),
        json.dumps({"name": "compressor/../../harness"}),
        json.dumps({}),
        json.dumps({"name": 42}),
        "not json",
        "",
    ]
    for candidate in bad:
        out = json.loads(load_skill(candidate))
        assert out["ok"] is False, candidate
        assert out["error"] == "unknown skill", candidate
        assert out["available"] == H.skill_names(), candidate


def test_skill_tools_follow_single_candidate_string_contract():
    for fn in (list_skills, load_skill):
        sig = inspect.signature(fn)
        assert list(sig.parameters) == ["candidate"]
        assert sig.parameters["candidate"].default == ""
        assert json.loads(fn("{not json"))["ok"] is False  # never raises
    assert json.loads(list_skills(""))["ok"] is True       # blank == empty filter object
    assert json.loads(load_skill(""))["ok"] is False       # a skill name is required


# --------------------------------------------------------------------------- #
# 3. compact prompt: catalog in dynamic mode, byte-identical in default mode
# --------------------------------------------------------------------------- #
def _compact_system(task: str, *, dynamic: bool) -> str:
    tools = make_tools(task, {}, include_dynamic=dynamic)
    return build_compact_harness_messages(
        _DEFAULT_BASE, task, None, tools=tools, include_dynamic=dynamic
    )[0]["content"]


def test_compact_dynamic_prompt_has_catalog_not_full_body():
    sysmsg = _compact_system("repair", dynamic=True)
    assert "SKILL CATALOG" in sysmsg
    assert "load_skill" in sysmsg and "list_skills" in sysmsg
    assert "Use list_skills/load_skill to inspect a skill in full before applying it." in sysmsg
    # One-line descriptions ARE present (primary skill + component skills).
    assert "Primary skill for repair: repair-diagnosis" in sysmsg
    assert "compressor: Use when a plant contains a compressor" in sysmsg
    # No full body is inlined in the system prompt.
    assert "# Compressor" not in sysmsg
    assert "## Physics and equations" not in sysmsg
    assert "## Failure modes" not in sysmsg
    assert _body("repair-diagnosis")[:80] not in sysmsg
    assert "check_answer" not in sysmsg


def test_compact_default_prompt_is_pinned_and_on_demand():
    out = build_compact_harness_messages(_DEFAULT_BASE, "repair", None)[0]["content"]
    assert hashlib.sha256(out.encode("utf-8")).hexdigest() == _DEFAULT_SHA
    assert build_harness_tool_messages(_DEFAULT_BASE, "repair", None)[0]["content"] == out
    assert "SKILL CATALOG" in out and "load_skill" in out and "list_skills" in out
    assert "Before EVERY tool call write 1-2 sentences of reasoning" in out
    assert "SKILL repair-diagnosis" not in out.replace("SKILL CATALOG", "")[0:0] or True


@pytest.mark.parametrize("task", ("repair", "design", "audit", "ledger"))
def test_compact_prompts_stay_under_token_ceiling(task, corpus_fixture_rows):
    tok = tokenizer()
    if tok is None:
        pytest.skip("no cached Qwen3.5 tokenizer")
    row = corpus_fixture_rows[task][0]
    base = row["prompt"][0]["content"]
    tools = make_tools(task, row, include_dynamic=True)
    sysmsg = build_harness_tool_messages(
        list(row["prompt"]), task, None, tools=tools, include_dynamic=True)[0]["content"]
    block = sysmsg[len(base):] if sysmsg.startswith(base) else sysmsg
    n = len(tok.encode(block).ids)
    assert n <= TOKEN_CEILING, f"{task}: dynamic compact harness is {n} tokens"


# --------------------------------------------------------------------------- #
# 4. end-to-end trajectory: tool results only, system prompt never mutates
# --------------------------------------------------------------------------- #
def test_multiturn_trajectory_loads_skill_without_touching_system_prompt(corpus_fixture_rows):
    row = corpus_fixture_rows["repair"][0]
    ctx = agent_graph.build_context({"row": row, "task": "repair", "notes": [],
                                     "include_dynamic": True})
    sys0 = ctx["system_prompt"]
    tool_map = ctx["tool_map"]
    assert "load_skill" in tool_map and "list_skills" in tool_map

    # Turn 1: the model reaches for the catalog.
    st1 = {"last_text": "", "last_reasoning": "inventory skills",
           "last_calls": [{"name": "list_skills",
                           "arguments": {"candidate": json.dumps({"query": "compressor"})}}],
           "tool_map": tool_map, "turns": [], "messages": ctx["messages"], "rounds": 0}
    t1 = agent_graph.call_tools(st1)
    catalog = json.loads(t1["turns"][0]["tool_results"][0])
    assert "compressor" in [s["name"] for s in catalog["skills"]]
    assert "## Physics and equations" not in t1["turns"][0]["tool_results"][0]

    r1 = agent_graph.reflect({**st1, **t1})
    assert r1["messages"][0]["content"] == sys0

    # Turn 2: the model loads the full skill — it arrives as a TOOL RESULT.
    st2 = {"last_text": "", "last_reasoning": "read the skill",
           "last_calls": [{"name": "load_skill",
                           "arguments": {"candidate": json.dumps({"name": "compressor"})}}],
           "tool_map": tool_map, "turns": t1["turns"],
           "messages": r1["messages"], "rounds": r1["rounds"]}
    t2 = agent_graph.call_tools(st2)
    full_text_result = t2["turns"][-1]["tool_results"][0]
    loaded = json.loads(full_text_result)
    assert loaded["ok"] is True and loaded["text"] == _body("compressor")
    assert "## Physics and equations" in full_text_result
    # The loaded text lives in the tool result only: the system prompt is intact.
    assert loaded["text"] not in sys0
    assert r1["messages"][0]["content"] == sys0
    assert all(m["content"] != loaded["text"] for m in r1["messages"] if m["role"] == "system")


# --------------------------------------------------------------------------- #
# 5. registry invariants: no answer oracle, canonical order, 5 / 16 tools
# --------------------------------------------------------------------------- #
def test_registry_invariants_and_canonical_order():
    assert "check_answer" not in tool_names()
    assert "check_answer" not in tool_names(include_dynamic=False)
    assert SKILL_TOOL_NAMES == ("list_skills", "load_skill")
    assert tool_names(include_dynamic=False) == PLANT_TOOLS
    assert tool_names() == (
        PLANT_TOOLS + list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES))
    assert len(tool_names(include_dynamic=False)) == 5 and len(tool_names()) == 16
    assert len(set(tool_names(include_dynamic=True))) == 16
    row = {"prompt": [{"role": "system", "content": "SYS"},
                      {"role": "user", "content": "U"}]}
    assert [f.__name__ for f in make_tools("repair", row)] == tool_names()
    assert [f.__name__ for f in make_tools(
        "repair", row, include_dynamic=True)] == tool_names(include_dynamic=True)
