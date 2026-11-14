"""The Methods draft quotes constants; keep them equal to the code."""
import re, pathlib
from thermo_agent.train import config as C, rewards as RR
from thermo_agent.design import rewards as DR
from thermo_agent.audit import rewards as AR
from thermo_agent.ledger import rewards as LR
from thermo_agent.core import components as K
from thermo_agent.synth.topologies import ORC_T_MAX

T = (pathlib.Path(__file__).parent.parent / "paper" / "methods_draft.md").read_text()

def test_reward_weights_in_text_match_code():
    for w in (RR.REWARD_WEIGHTS, DR.REWARD_WEIGHTS, AR.REWARD_WEIGHTS, LR.REWARD_WEIGHTS):
        assert "[" + ", ".join(f"{x:g}" for x in w) + "]" in T

def test_training_settings_in_text_match_code():
    s, g = C.SftSettings(), C.GrpoSettings()
    assert f"r = {s.lora.r}, alpha = {s.lora.alpha}" in T and "learning rate 2e-4" in T and f"{s.epochs} epochs" in T
    assert f"r = {g.lora.r}, alpha = {g.lora.alpha}" in T and "learning rate 5e-5" in T and f"group size {g.num_generations}" in T and f"{g.max_completion_length} completion tokens" in T

def test_component_constants_in_text_match_code():
    assert K.PARAMS["absorption_chiller"]["eps"][1] == 0.27 and K.PARAMS["absorption_chiller"]["COP_max"][1] == 0.8 and "eps = 0.27, COP_max = 0.8" in T
    assert K.PARAMS["electric_chiller"]["eps"][1] == 0.5 and K.PARAMS["electric_chiller"]["approach_K"][1] == 5.0
    assert K.PARAMS["heat_rejector"]["fan_frac"][1] == 0.015 and "1.5 %" in T
    assert K.PARAMS["orc_evaporator"]["T_stack_min"][1] == 333.15 and "333 K" in T
    assert "420, 480, 560 K" in T and [ORC_T_MAX[f] for f in ("R245fa", "Cyclopentane", "Toluene")] == [420.0, 480.0, 560.0]
