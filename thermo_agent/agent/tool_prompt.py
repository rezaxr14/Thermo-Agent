"""Shared tool-using system prompt for the H eval arm and the SFT/GRPO corpora.

Training rows must see the *same* system prompt the model sees at eval time in
``ours_harness``: a compact AGENTS.md block + the one selected skill + the
multi-turn tool contract. Keeping the text in one module avoids eval/training
prompt drift.

J2/J3 (docs/DECISIONS.md): the compact harness is <= 2,500 real
Qwen3.5 tokens. ``build_compact_harness_messages`` is the single source of that
text and every arm (tracegen, training ``*_h`` corpora, eval/GRPO) calls it
through ``build_harness_tool_messages``. The original verbose builder is kept
callable with ``compact=False`` for before/after comparison; it is never used
by a production arm.
"""
from __future__ import annotations

import inspect as _inspect

from . import harness as H

# Final-answer XML contracts mirror agent/AGENTS.md ("Answer formats") and
# train/prompts.SYSTEM. Kept here as the single source for the tool contract.
FINAL_TAGS = {
    "repair": ("<diagnosis>", "<edits>"),
    "design": ("<rationale>", "<design>"),
    "audit": ("<analysis>", "<ranking>", "<change>", "<prediction>"),
    "ledger": ("<diagnosis>", "<fix>"),
}

# One line per task: the exact answer format the model must emit. These are the
# only task-specific sentences in the compact harness.
_TASK_ANSWER_FORMAT = {
    "repair": ("<diagnosis> citing only shown numbers, then <edits> with "
               "`SET <component>.<param> = <number>`, `SET @<stream>.<field> = <number>`, "
               "`REWIRE @<stream>.<to|from> = <component>.<port>`, `FLUID @<stream> = <fluid>`; "
               "fewest edits, no add/remove components."),
    "design": ("<rationale> then <design> with `key = value` lines, exactly the keys "
               "your choices require."),
    "audit": ("<analysis>, <ranking>A > B > C</ranking>, <change>SET ...</change>, "
              "<prediction>fuel_change_mw = ...</prediction>."),
    "ledger": ("<diagnosis> citing only shown MW numbers, then <fix> with "
               "`FIX T0 = <expr>`, `FIX E[<stream>] = <expr>`, `FIX ED[<component>] = <expr>`."),
}

# One-line semantics for the five physics tools (name -> contract). Keeping
# these here replaces the verbose toolkit docstrings in the compact harness.
_COMPACT_TOOL_LINE = {
    "inspect_plant": ('inspect_plant(candidate=""): solver status, failed checks and key '
                      'residuals for the plant/ledger AS GIVEN.'),
    "first_law_check": ('first_law_check(candidate): per-line/per-component mass (kg/s) and '
                        'energy (W) residuals, heat/power/fuel balance, convergence, failed checks.'),
    "second_law_check": ('second_law_check(candidate): exergy destruction (MW), closure, '
                         'eta_I/eta_II, Carnot/COP limits, failed checks.'),
    "validity_check": ('validity_check(candidate): unit/range violations, site constraints, '
                       'pinch approach (K), verifier verdict.'),
    "solve_plant": ('solve_plant(candidate): full stream state table (T K, P kPa, h/s kJ/kg, '
                    'exergy).'),
}

# One-line semantics for the nine continuous-input component tools (opt-in hosted mode).
# These are deliberately short: do NOT duplicate the long SKILL.md tool-guide text here.
_COMPACT_TOOL_LINE.update({
    "fluid_state": ('fluid_state(candidate): real-fluid/gas state from any two properties '
                    '(T_K, P_kPa, h_kJ_kg, s_kJ_kgK, q); T, P, h, s, rho, cp, phase.'),
    "saturation_state": ('saturation_state(candidate): T_sat/P_sat, h_fg, s_fg, v_f/v_g at T '
                         'or P and quality.'),
    "turbomachine": ('turbomachine(candidate): compressor/turbine/pump/fan outlet state, '
                     'specific work, power, exergy destruction from inlet state and eta_s.'),
    "heat_exchanger": ('heat_exchanger(candidate): two-stream duty, both outlets, min approach, '
                       'LMTD and UA from one outlet temperature, duty or effectiveness.'),
    "combustor": ('combustor(candidate): fuel/air/flue flows, heat release, AFR, adiabatic '
                  'flame temperature and CO2 flow.'),
    "cycle_performance": ('cycle_performance(candidate): Carnot/Lorenz limit and real '
                          'efficiency or COP for Rankine, ORC, heat-pump, chiller, absorption.'),
    "flow_network": ('flow_network(candidate): mixer/splitter/valve/pipe outlets, mass and '
                     'energy residuals, entropy generation and exergy destruction.'),
    "psychrometrics": ('psychrometrics(candidate): humid-air humidity ratio, enthalpy, dew and '
                       'wet bulb, density and cooling-tower approach.'),
    "exergy_stream": ('exergy_stream(candidate): physical + chemical flow exergy of one stream '
                      'vs the dead state, plus h and s.'),
})

# The two on-demand skill tools (opt-in, dynamic mode only). Skill bodies are
# delivered as tool results, never embedded in this prompt.
_COMPACT_TOOL_LINE.update({
    "list_skills": ('list_skills(candidate): compact JSON catalog of relevant skills '
                    '(name, description, components/variants/task) with a hint to call '
                    'load_skill; candidate is a JSON object {"task", "components", "query"} '
                    '(all optional); never returns full skill bodies.'),
    "load_skill": ('load_skill(candidate): candidate is {"name": "..."}; returns the FULL '
                   'SKILL.md text as this tool result only — call list_skills first to find '
                   'names; the system prompt is never changed.'),
})

_PLANT_TOOL_NAMES = frozenset({
    "inspect_plant", "first_law_check", "second_law_check", "validity_check", "solve_plant"})


def tool_contract(tools, task, max_turns, max_tool_calls) -> str:
    """The multi-turn limits and final-answer contract as text.

    The call syntax and the tool definitions are owned by the chat template:
    ``apply_chat_template(tools=...)`` injects them. This text must not duplicate
    either, so the model sees exactly one protocol.
    """
    tags = FINAL_TAGS.get(task, ("the task's XML tags",))
    lines = [
        "MULTI-TURN TOOL ENVIRONMENT (physics only; there is NO answer-oracle tool).",
        f"Limits: at most {max_turns} assistant turns and {max_tool_calls} tool calls.",
        "Tool results are environment output; read the numbers and revise.",
        "When done, STOP calling tools and emit ONLY the final answer XML tags: " + " + ".join(tags),
    ]
    return "\n".join(lines)


# 2026-11-20: reasoning prose is REQUIRED for every tool turn (the old traces had none and were archived).
RATIONALE_LINE = ("Before EVERY tool call write 1-2 sentences of reasoning (what you are checking and why); "
                  "after a result, one sentence on what the numbers imply. Cite only numbers shown.")


def compact_tool_contract(task, max_turns, max_tool_calls, tools=None) -> str:
    """The same multi-turn contract with one-line semantics (J2/J3 compact path).

    Only the rules stay in prompt text. The tool list and the native
    ``<tool_call><function=...>`` wire format are injected by
    ``apply_chat_template(tools=...)``; duplicating them here would create a second
    protocol.
    """
    tags = FINAL_TAGS.get(task, ("the task's XML tags",))
    lines = [
        "MULTI-TURN TOOL ENVIRONMENT (physics only; there is NO answer-oracle tool).",
        "The available functions are defined by this request; call them with the tool-call "
        "format the chat template specifies.",
        "Each tool result is environment output; read the numbers and revise.",
        RATIONALE_LINE,
        f"At most {max_turns} assistant turns and {max_tool_calls} tool calls. Call inspect_plant "
        "first (no answer needed); when residuals ~0 and checks pass, STOP and emit ONLY: "
        + " + ".join(tags) + ".",
    ]
    return "\n".join(lines)


def _compact_selected_skill(task: str) -> str:
    """The ONE task-selected skill, with duplicated sections removed.

    ``TASK_SKILLS`` designates at most one skill per task; component skills are
    deliberately *not* included (they are non-selected and duplicate AGENTS.md
    and each other). The skill's ``## Environment:`` section repeats the tool
    contract and final-answer format, so it is dropped.

    This is the DEFAULT-mode path: it is byte-for-byte the original compact
    text so the accepted traces / deterministic rows keep validating. Dynamic
    mode instead serves :func:`_compact_skill_catalog`.
    """
    names = H.TASK_SKILLS.get(task, [])
    if not names:
        return ""
    skill = H.read_skill(names[0])          # "SKILL <name>: <description>\n<body>"
    skill = skill.split("\n## Environment:")[0].rstrip()
    return skill


def _tools_include_dynamic(tools) -> bool:
    """Infer dynamic mode from the tool list when no explicit flag is passed.

    A caller that supplies the 11 opt-in tools (nine component + two skill tools)
    gets the on-demand catalog; a caller that supplies nothing or the five plant
    tools gets the original, byte-identical compact prompt.
    """
    if not tools:
        return True        # default harness = on-demand skills + component tools (2026-11-20)
    from ..teacher.thermo_tools import DYNAMIC_TOOL_NAMES
    from ..teacher.toolkit import SKILL_TOOL_NAMES
    names = {getattr(fn, "__name__", "") for fn in tools}
    return bool(names & (set(DYNAMIC_TOOL_NAMES) | set(SKILL_TOOL_NAMES)))


def _compact_skill_catalog(task: str) -> str:
    """On-demand skill catalog for the COMPACT dynamic-mode prompt.

    One line per skill (frontmatter description only): the task's primary skill
    plus every component skill in the library. No SKILL.md body is ever inlined;
    the model calls ``list_skills``/``load_skill`` to inspect a skill in full.
    """
    from ..teacher.toolkit import all_skill_metadata
    metas = all_skill_metadata()
    by_name = {m["name"]: m for m in metas}
    lines = ["SKILL CATALOG (descriptions only; full text arrives as a tool result):"]
    primary = (H.TASK_SKILLS.get(task) or [None])[0]
    if primary and primary in by_name:
        m = by_name[primary]
        lines.append(f"Primary skill for {task}: {m['name']}: {m['description']}")
    lines.append("Use list_skills/load_skill to inspect a skill in full before applying it.")
    lines.append("Available component skills (call load_skill(name) for the full text):")
    for m in metas:
        if m["components"]:
            lines.append(f"- {m['name']}: {m['description']}")
    return "\n".join(lines)


def compact_agents_md(task: str) -> str:
    """Compact AGENTS.md: conventions, failure checklist, this task's answer format.

    The ``Conventions``/``Components``/``Tools``/``Answer formats`` prose in the
    full AGENTS.md is duplicated by the contract and skill blocks, so only the
    lines that carry unique physics/format information are kept.
    """
    answer = _TASK_ANSWER_FORMAT.get(task, "the task's XML answer tags.")
    return "\n".join([
        "# AGENTS.md: energy-plant tasks (compact)",
        "Simulated plants for AI data centres; a solver and physics verifiers judge every "
        "answer, so cite only numbers the solver reproduces.",
        "Conventions: SI units (K, Pa, W, kg/s); eps/dp/PR are fractions, never percent; "
        "temperature in K, never degC. Plant graph: `C <id> <type> {params}` is a component; "
        "`S <id> <from> -> <to> <fluid> {extras}` is a stream; endpoints are `<comp>.<port>` "
        "or `ext:<name>`. Gas fluids are CGAM_AIR/FLUE/CH4; others are CoolProp liquids "
        "(Water, R245fa, Cyclopentane, Toluene); a port accepts only its fluid class. A `fixed` "
        "stream produced by a component is a tear closing a loop: change both sides together. "
        "Flows a component solves itself must not be specified (engine air/fuel, chiller "
        "gen/rejection, IT coolant, heat-pump sink).",
        "Failure checklist (before any candidate): unit mismatch (MW vs W, kJ vs J), fraction "
        "vs percent, K vs C; sign flip; swapped ports; wrong fluid class; efficiency > 1; "
        "out-of-range parameter; unconverged tear. Locate the failing check/stream first, fix "
        "it with the fewest edits, and never invent numbers.",
        f"Answer format ({task}): {answer}",
    ])


def _default_tools(dynamic: bool = True):
    """The physics tools in canonical order (plus the component and on-demand skill tools), for prompt building."""
    from .env import ThermoEnv
    from ..teacher import toolkit
    base = [ThermoEnv.inspect_plant, ThermoEnv.first_law_check, ThermoEnv.second_law_check,
            ThermoEnv.validity_check, ThermoEnv.solve_plant]
    return base + toolkit.dynamic_tools() + toolkit.skill_tools() if dynamic else base


def build_compact_harness_messages(messages, task, plant, mode="agents+skills",
                                   max_turns=6, max_tool_calls=8, tools=None,
                                   include_contract=True,
                                   include_dynamic: bool | None = None) -> list[dict]:
    """A deep copy of ``messages`` with the J2/J3 compact harness appended.

    The appended block is a single, task-determined text: compact AGENTS.md +
    the selected skill (default mode) OR the on-demand skill catalog (dynamic
    mode) + the compact tool contract. It is the source of truth shared by
    tracegen, the ``*_h`` training corpora and eval/GRPO.
    ``mode="none"`` adds the contract only (matching the original builder).
    ``include_contract=False`` is the no-tools (Z) eval arm: compact AGENTS +
    skill but no tool contract. The input list and its dicts are never mutated.

    ``include_dynamic`` defaults to ``None`` (infer from ``tools``). DEFAULT mode
    (no opt-in tools) is byte-for-byte the original text; only dynamic mode
    serves descriptions + ``load_skill`` instead of an inlined full skill body.
    """
    out = [dict(m) for m in messages]
    dynamic = _tools_include_dynamic(tools) if include_dynamic is None else bool(include_dynamic)
    parts = []
    if mode != "none":
        parts.append(compact_agents_md(task))
        if dynamic:
            parts.append(_compact_skill_catalog(task))
        else:
            skill = _compact_selected_skill(task)
            if skill:
                parts.append(skill)
    if include_contract:
        parts.append(compact_tool_contract(
            task, max_turns, max_tool_calls,
            tools if tools is not None else _default_tools(dynamic)))
    if not parts:
        return out
    block = "\n\n".join(parts)
    if out and out[0].get("role") == "system":
        out[0]["content"] = out[0]["content"] + "\n\n" + block
    else:
        out.insert(0, {"role": "system", "content": block})
    return out


def build_harness_tool_messages(messages, task, plant, mode="agents+skills",
                                max_turns=6, max_tool_calls=8, tools=None,
                                compact=True,
                                include_dynamic: bool | None = None) -> list[dict]:
    """A deep copy of ``messages`` with the harness and tool contract appended.

    ``compact=True`` (the default) delegates to :func:`build_compact_harness_messages`
    so every arm sees one identical harness text. ``compact=False`` keeps the
    original verbose behaviour (full AGENTS.md + all selected component skills
    + toolkit docstrings) for before/after comparison only.

    ``include_dynamic`` is forwarded to the compact builder; ``None`` infers it
    from ``tools``. The default path is unchanged.
    """
    if compact:
        return build_compact_harness_messages(
            messages, task, plant, mode=mode, max_turns=max_turns,
            max_tool_calls=max_tool_calls, tools=tools,
            include_dynamic=include_dynamic)
    augmented = [dict(m) for m in H.augment(list(messages), mode, task, plant)]
    contract = tool_contract(tools if tools is not None else _default_tools(),
                             task, max_turns, max_tool_calls)
    if augmented and augmented[0].get("role") == "system":
        augmented[0]["content"] = augmented[0]["content"] + "\n\n" + contract
    else:
        augmented.insert(0, {"role": "system", "content": contract})
    return augmented
