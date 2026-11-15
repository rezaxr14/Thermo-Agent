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

def test_harness_size_is_bounded():
    # AGENTS.md plus one task skill must leave room for the plant text inside max_length 4096.
    assert len(H.augment(MSGS, "agents+skills", "design")[0]["content"]) < 9000
