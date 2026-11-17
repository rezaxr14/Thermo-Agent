"""The corpus must train on ORC, co-generation and Rankine, and hold out diagram *combinations* instead of families.

Holding out every engine and the steam plant (the old rule) meant the model never trained on ORC, co-gen or
Rankine at all. These tests pin the replacement: `cgam_steam_orc` and `engine-*-ext*` are the held-out
combinations, every other diagram trains, and audit and ledger can both be built over the architecture catalog.
"""
import json
import random

import pytest

from thermo_agent.audit import data as audit_data
from thermo_agent.ledger import data as ledger_data
from thermo_agent.synth import architectures as A
from thermo_agent.synth.splits import is_ood_topology, topology_label
from thermo_agent.synth.topologies import dc_chp, size_generation


def test_held_out_combinations_are_the_steam_plant_and_engine_with_ideal_rejection():
    assert is_ood_topology("cgam_steam_orc")
    assert is_ood_topology("engine-absorption-ext")
    assert is_ood_topology("engine-electric-ext-orc")
    assert is_ood_topology("engine-absorption-ext-orc-dh_heat_pump")


def test_families_the_model_must_learn_are_not_held_out():
    for name in ("cgam_orc", "dc_engine_electric", "dc_engine_absorption", "dc_absorption", "dc_electric", "dc_chp",
                 "turbine-absorption-dry-orc", "engine-absorption-wet", "turbine-electric-wet-orc-dh_exhaust",
                 "engine-absorption-wet-orc-dh_heat_pump"):
        assert not is_ood_topology(name), f"{name} must be training data"


def _mix_plant(prime_mover, rejection):
    """The five-template generator names engine plants without encoding their rejection boundary."""
    return size_generation(dc_chp, P_it=20e6, cooling="absorption", prime_mover=prime_mover,
                           rejection=rejection, name=f"dc_{prime_mover}_absorption_sampled")


@pytest.mark.parametrize("rejection", ["ext", "dry", "wet"])
def test_mix_engine_rejection_is_in_the_topology_label(rejection):
    """Otherwise ideal-rejection engines leak into train through `data/processed`."""
    plant = _mix_plant("engine", rejection)
    assert topology_label(plant) == ("dc_engine_absorption-ext" if rejection == "ext" else "dc_engine_absorption")
    assert is_ood_topology(topology_label(plant)) == (rejection == "ext")
    assert (audit_data.split_of(plant, "aud-test") == "test_ood_topology") == (rejection == "ext")


def test_mix_turbine_with_ideal_rejection_is_not_held_out():
    assert not is_ood_topology(topology_label(_mix_plant("turbine", "ext")))


def test_ledger_rows_from_mix_engine_ext_are_held_out():
    from thermo_agent.synth.splits import topology_label as label
    plant = _mix_plant("engine", "ext")
    row = ledger_data.make_row("led-mix-ext", plant, random.Random("led-mix-ext"))
    assert row is not None
    assert row["topology"] == label(plant) == "dc_engine_absorption-ext"
    assert row["split"] == "test_ood_topology"


def test_the_catalog_keeps_orc_engines_and_steam_in_training():
    """At least one ORC, one engine and one turbine diagram must survive into training."""
    training = [a["name"] for a in A.ARCHITECTURES if not is_ood_topology(a["name"])]
    assert any("orc" in n for n in training), "ORC must be in the training set"
    assert any(n.startswith("engine-") for n in training), "co-generation must be in the training set"
    assert any(n.startswith("turbine-") for n in training)
    assert len(training) >= 50


def _sample(name):
    rng = random.Random(f"test-{name}")
    return A.sample_plant(A.by_name(name), rng)[0]


def test_audit_can_build_rows_on_catalog_diagrams():
    plant = _sample("engine-absorption-dry-orc")
    row = audit_data.make_row("aud-arch-test", plant)
    assert row is not None, "an ORC co-generation plant must yield an audit row"
    assert row["topology"] == "engine-absorption-dry-orc"
    assert row["split"] in ("train", "val", "test_id"), "an engine with dry rejection trains"
    assert row["best_gain_w"] > 0 and "<analysis>" in row["completion"][0]["content"]


def test_ledger_can_build_rows_on_catalog_diagrams():
    plant = _sample("engine-absorption-dry-orc")
    row = ledger_data.make_row("led-arch-test", plant, random.Random("led"))
    assert row is not None, "an ORC co-generation plant must yield a ledger row"
    assert row["topology"] == "engine-absorption-dry-orc"
    assert row["split"] in ("train", "val", "test_id")
    assert row["fault_types"] and "<fix>" in row["completion"][0]["content"]


@pytest.mark.parametrize("task,module", [("audit", audit_data), ("ledger", ledger_data)])
def test_architectures_flag_writes_ids_and_meta(tmp_path, task, module):
    name = "turbine-absorption-dry"
    counts = module.build(tmp_path, 2, seed=0, workers=1, arch=True, names=[name])
    assert sum(counts.values()) >= 1
    files = list(tmp_path.glob(f"{task}_*.jsonl"))
    assert files, "the builder must write its split files"
    rows = [json.loads(l) for f in files for l in f.open(encoding="utf-8")]
    assert all(r["id"].startswith(f"{task[:3]}-arch-") for r in rows)
    assert all(r["topology"] == name for r in rows)
    meta = tmp_path / f"meta_{task}_arch.json"
    assert json.loads(meta.read_text(encoding="utf-8"))["params"]["architectures"] == 1
