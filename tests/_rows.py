"""Fixtures for tests that need one real repair row. Building the plant is the slow part, so it is cached."""
from __future__ import annotations

import functools
import random

from thermo_agent.agent.dsl import patch_to_edits_text
from thermo_agent.synth.faults import INJECTORS, NotApplicable
from thermo_agent.synth.topologies import dc_chp, size_generation


@functools.lru_cache(maxsize=16)
def make_row(fault_name: str = "wrong_fluid", seed: int = 0) -> dict:
    """A plant, a real injected fault, and the ground-truth expert answer as a dataset row."""
    plant = size_generation(dc_chp, P_it=20e6, cooling="absorption", prime_mover="turbine", rejection="dry")
    last = None
    for s in range(seed, seed + 12):
        try:
            broken, fault = INJECTORS[fault_name](plant, random.Random(s))
            break
        except NotApplicable as exc:
            last = exc
    else:
        raise RuntimeError(f"injector {fault_name} never applied: {last}")
    text = "<diagnosis>placeholder</diagnosis>\n" + patch_to_edits_text(broken, fault.inverse_patch)
    return {"id": f"row-{fault_name}-{seed}", "split": "train", "topology": plant.name,
            "fault_type": fault.type, "prompt": [{"role": "system", "content": "You are a plant engineer."},
                                                 {"role": "user", "content": "Repair the plant."}],
            "completion": [{"role": "assistant", "content": text}],
            "plant_broken": broken.model_dump_json(by_alias=True),
            "plant_original": plant.model_dump_json(by_alias=True), "site": "",
            "deducible": bool(fault.deducible), "min_edits": len(fault.inverse_patch), "detected": True}
