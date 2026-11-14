import random, pytest
from thermo_agent.agent.dsl import plant_to_text, text_to_plant, parse_edits, edits_to_patch, patch_to_edits_text, EditError
from thermo_agent.synth.repair_dataset import sample_plant
from thermo_agent.synth.faults import INJECTORS, apply_patch, _dump
from thermo_agent.core.schema import Plant

@pytest.mark.parametrize("topo", ["cgam_orc", "dc_absorption", "dc_electric"])
def test_plant_text_roundtrip_and_size(topo):
    p, _, _ = sample_plant(random.Random(1), topology=topo)
    t = plant_to_text(p); q = text_to_plant(t)
    assert _dump(q) == _dump(p)
    assert len(t) < 4000                       # compact enough for a 4B model prompt

def test_edits_roundtrip_matches_inverse_patch():
    rng = random.Random(2)
    for kind in INJECTORS:
        orig, _, _ = sample_plant(rng); broken, f = INJECTORS[kind](orig, rng)
        txt = patch_to_edits_text(broken, f.inverse_patch)
        patch = edits_to_patch(broken, parse_edits(txt))
        assert patch == f.inverse_patch
        assert _dump(Plant.model_validate(apply_patch(_dump(broken), patch))) == _dump(orig)

@pytest.mark.parametrize("bad", ["no block", "<edits>\nSET AC.eta_s 0.8\n</edits>", "<edits>\nDROP AC\n</edits>",
                                 "<edits>\n" + "\n".join(f"SET AC.eta_s = 0.{i}" for i in range(1, 9)) + "\n</edits>"])
def test_malformed_edits_rejected(bad):
    with pytest.raises(EditError): parse_edits(bad)

def test_edits_cannot_add_or_touch_nonexistent_fields():
    p, _, _ = sample_plant(random.Random(3), topology="dc_absorption")
    for txt in ["<edits>\nSET AC.nonexistent = 1.0\n</edits>", "<edits>\nSET ZZ.eta_s = 0.9\n</edits>",
                "<edits>\nSET @air.fixed.x = 0.5\n</edits>", "<edits>\nSET @nostream.m_dot = 1.0\n</edits>"]:
        with pytest.raises(EditError): edits_to_patch(p, parse_edits(txt))
