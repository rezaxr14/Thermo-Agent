import random, pytest
from thermo_agent.agent import harness as H, build_harness as B
from thermo_agent.core import components as C
from thermo_agent.synth.repair_dataset import sample_plant
from thermo_agent.synth.topologies import ORC_T_MAX

MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]

def test_generated_sections_are_up_to_date():
    assert B.main(check=True) == 0

def test_skills_reference_real_component_types_and_have_descriptions():
    for n in H.skill_names():
        fm = B.frontmatter((H.ROOT / "skills" / n / "SKILL.md").read_text())
        assert fm["name"] == n and fm["description"]
        assert all(c in C.REQUIRED_PORTS for c in fm.get("components", []))

def test_skill_numbers_match_the_code():
    orc = H.read_skill("orc-bottoming")
    for fluid, t in ORC_T_MAX.items(): assert f"{fluid} {t:.0f} K" in orc
    assert C.PARAMS["gas_engine"]["P_exh"][1] == 1.1e5 and "110 kPa" in H.read_skill("gas-engine")
    assert "COP_max" in H.read_skill("absorption-chiller") and "0.8" in H.read_skill("absorption-chiller")

def test_augment_modes_and_immutability():
    assert H.augment(MSGS, "none", "repair") is MSGS
    a = H.augment(MSGS, "agents", "repair"); assert "AGENTS.md" in a[0]["content"] and MSGS[0]["content"] == "SYS" and a[1] == MSGS[1]
    with pytest.raises(ValueError): H.augment(MSGS, "everything", "repair")

def test_gas_engine_skill_is_selected_only_for_engine_plants():
    rng = random.Random(1)
    eng, _, _ = sample_plant(rng, topology="dc_engine_electric"); turb, _, _ = sample_plant(rng, topology="dc_electric")
    assert "gas-engine" in H.skills_for(eng, "repair") and "gas-engine" not in H.skills_for(turb, "repair")
    text = H.augment(MSGS, "skills", "repair", eng)[0]["content"]
    assert "SKILL gas-engine" in text and "AGENTS.md" not in text
    assert "SKILL exergy-audit" in H.augment(MSGS, "skills", "audit", turb)[0]["content"]

def test_design_without_a_plant_gets_only_its_catalog_skill():
    s = H.skills_for(None, "design")
    assert s == ["design-catalog"], "all component skills at once do not fit the 4096-token prompt budget"
    assert H.skills_for(None, "ledger") == ["exergy-ledger"]


def test_repair_skill_mapped_and_component_skills_selected_from_frontmatter():
    # Repair traces must carry a real diagnosis skill instead of an empty map.
    assert H.TASK_SKILLS["repair"] == ["repair-diagnosis"]
    rng = random.Random(7)
    eng, _, _ = sample_plant(rng, topology="dc_engine_electric")
    selected = H.skills_for(eng, "repair")
    assert "repair-diagnosis" in selected

    comp_types = {c.type for c in eng.components}
    expected = set()
    for n in H.skill_names():
        comps = B.frontmatter((H.ROOT / "skills" / n / "SKILL.md").read_text()).get("components")
        if comps and any(c in comp_types for c in comps):
            expected.add(n)
    assert len(expected) >= 3, "the fixture plant should exercise several component skills"
    assert expected <= set(selected), "component skills are selected via `components:` frontmatter"
    # The task skill stays first so the compact builder keeps serving exactly one skill.
    assert selected[0] == "repair-diagnosis"

def test_harness_size_is_bounded():
    # AGENTS.md plus one task skill must leave room for the plant text inside max_length 4096.
    assert len(H.augment(MSGS, "agents+skills", "design")[0]["content"]) < 9000


def test_verbose_harness_stays_bounded_for_a_19_component_plant():
    """A plant matching many component skills must not silently blow the verbose harness budget.

    This is the regression for the 27-skill library: with every matching component skill
    rendered, the same plant produced ~31k chars / ~8.3k tokens, over both the corpus'
    22k-char prefilter and the ~4k-token SFT budget, so `combine` silently dropped every
    such row. The bounded verbose path must stay under both gates and keep the task skill.
    """
    from thermo_agent.core.schema import Component, Plant
    from thermo_agent.synth.combine import TOKEN_RESERVE, tokenizer

    types = [
        "compressor", "combustor", "turbine", "recuperator", "hrsg", "orc_evaporator",
        "orc_turbine", "pump", "condenser", "absorption_chiller", "heat_pump",
        "electric_chiller", "dc_load", "mixer", "splitter", "valve", "heat_rejector",
        "gas_engine", "compressor",  # 19 components, 14 distinct types -> many skills
    ]
    plant = Plant(name="kitchen-sink", streams=[],
                  components=[Component(id=f"c{i}", type=t) for i, t in enumerate(types)])

    tok = tokenizer()
    for task, skill in H.TASK_SKILLS.items():
        # skills_for itself stays unbounded (selection semantics are unchanged);
        # only the rendered verbose text is capped.
        assert len(H.skills_for(plant, task)) > 5
        text = H.augment(MSGS, "agents+skills", task, plant)[0]["content"]
        assert f"SKILL {skill[0]}" in text, f"{task}: task skill was not kept"
        assert len(text) < 22000, f"{task}: verbose harness is {len(text)} chars"
        if tok is not None:
            n = len(tok.encode(text, add_special_tokens=False).ids) + TOKEN_RESERVE
            assert n <= 4000, f"{task}: verbose harness is {n} tokens"


def test_compact_path_still_selects_exactly_one_task_skill():
    """The bounded verbose path must not change the compact production contract."""
    from thermo_agent.agent.tool_prompt import _compact_selected_skill
    for task, skill in H.TASK_SKILLS.items():
        block = _compact_selected_skill(task)
        kept = [n for n in H.skill_names() if f"SKILL {n}" in block]
        assert kept == [skill[0]], f"{task}: compact harness must serve one task skill"
