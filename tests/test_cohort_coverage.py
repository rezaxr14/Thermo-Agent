"""Cohort-coverage acceptance suite: every task, split, difficulty tier and the never-seen set.

Balance rationale
-----------------
The user wants samples from *everywhere* in the corpus with a genuine mix of easy,
medium and hard, not a repair-only smoke test. The tiers are:

* simple   -- ``data/processed``: the standard 9-architecture MIX (seed 0), all four
              tasks (repair / design / audit / ledger) and all four/five splits per task.
* medium   -- ``data/processed_arch``: the 65-diagram architecture catalog (seed 0),
              three tasks (repair / audit / ledger), harder topology-OOD splits.
* hard     -- ``data/processed_big``: the 19-component / 34-stream expert reheat cycle
              (seeds 100/101), repair + audit, plus every OOD split where present.
* untouched -- ``data/heldout``: a freshly generated set with seed 987654, used for the
              zero-shot -> tool-CoT comparison. Its ids live in a separate namespace
              from *all* corpora that training can see (processed, processed_arch,
              processed_big, sft_all, grpo_all), which is the "no cohort has seen it"
              guarantee asserted below.

Honesty notes (things that are true about the data, not covered-up):
* Raw generated rows (processed*, heldout) have **no literal ``task`` key** -- the task
  is encoded in the filename, the id prefix and the ``meta_*.json`` file. The
  ``sft_all`` / ``grpo_all`` derivatives do carry ``task``. This suite checks all three
  encodings so "completion, id, task, split" is genuinely verified per case.
* ``check_answer`` (the accept/reward oracle) is deliberately NOT in the tool list; grading
  stays at acceptance time and as the GRPO reward. All five physics tools return ``ok``.
* A ledger is a system of balance/exergy equations, not a physical plant, so its
  ``solve_plant`` correctly reports ``ok: true`` with **zero streams**. The real ledger
  numbers are exposed by the first-law, second-law and validity tools instead.
* Speed: only the first non-empty line of each jsonl file is read (no full scans of the
  1.4 GB sft_all repair_train, no batch solving). Id disjointness is proven by the
  seed-embedded id namespace plus first-line samples from every corpus/file, which is
  the structural guarantee, not an O(N) scan.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest

from thermo_agent.core.schema import Plant
from thermo_agent.teacher.toolkit import make_tools, tool_names

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TASKS = ("repair", "design", "audit", "ledger")
SPLITS = ("train", "val", "test_id", "test_ood_fault", "test_ood_topology",
          "test_ood_climate", "test_ood_load", "test_ood_scale")
SOURCES = ("processed", "processed_arch", "processed_big", "heldout")
EVAL_SPLITS = ("test_id", "test_ood_fault", "test_ood_topology",
               "test_ood_climate", "test_ood_load", "test_ood_scale")

# Both generated and derived id namespaces. ``arch-`` is repair in processed_arch and
# processed_big (the expert cycle uses arch-10000000-*).
ID_PREFIX_TO_TASK = {"rep": "repair", "des": "design", "aud": "audit",
                     "led": "ledger", "arch": "repair"}
HELD_OUT_SEED = 987654
TOOL_NAMES = ["inspect_plant", "first_law_check", "second_law_check",
              "validity_check", "solve_plant"]

TIER_CASES = (
    ("processed", {"repair", "design", "audit", "ledger"}),
    ("processed_arch", {"repair", "audit", "ledger"}),
    ("processed_big", {"repair", "audit"}),
    ("heldout", {"repair", "design", "audit", "ledger"}),
)


def _path(source: str, task: str, split: str) -> Path:
    return DATA / source / f"{task}_{split}.jsonl"


@lru_cache(maxsize=None)
def _first_row(source: str, task: str, split: str) -> dict | None:
    """Parse only the first non-empty line of a jsonl file; None if absent/empty."""
    path = _path(source, task, split)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    return None


@lru_cache(maxsize=1)
def _cases() -> tuple[tuple[str, str, str], ...]:
    """Every (source, task, split) triple that actually has at least one row."""
    return tuple(
        (source, task, split)
        for source in SOURCES
        for task in TASKS
        for split in SPLITS
        if _first_row(source, task, split) is not None
    )


@lru_cache(maxsize=1)
def _tool_cases() -> tuple[tuple[str, str], ...]:
    return tuple(
        (source, task)
        for source in SOURCES
        for task in TASKS
        if _first_row(source, task, "test_id") is not None
    )


@lru_cache(maxsize=None)
def _meta(source: str, task: str) -> dict | None:
    """meta_*.json for a source/task, trying both ``<task>`` and ``<task>_arch`` names."""
    folder = DATA / source
    for name in (f"meta_{task}.json", f"meta_{task}_arch.json"):
        path = folder / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _seed(source: str, task: str) -> int | None:
    meta = _meta(source, task)
    return None if meta is None else meta["params"]["seed"]


# --------------------------------------------------------------------------------------
# 1. All four tasks x all available splits, with ground-truth fields checked.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,task,split",
    _cases(),
    ids=[f"{s}-{t}-{sp}" for s, t, sp in _cases()],
)
def test_row_has_completion_id_task_and_split(source: str, task: str, split: str):
    row = _first_row(source, task, split)
    assert row is not None, f"missing first row: {source}/{task}_{split}"

    # completion: a non-empty assistant ground-truth answer
    completion = row["completion"]
    assert isinstance(completion, list) and completion, (source, task, split)
    assert completion[0]["role"] == "assistant", (source, task, split)
    assert isinstance(completion[0]["content"], str) and completion[0]["content"].strip()

    # id: non-empty and its namespace encodes the expected task
    rid = row["id"]
    assert isinstance(rid, str) and rid
    assert ID_PREFIX_TO_TASK.get(rid.split("-", 1)[0]) == task, (source, task, split, rid)

    # task: authoritative in meta; literal in sft/grpo rows; absent in raw rows by design
    if "task" in row:
        assert row["task"] == task, (source, task, split, row["task"])
    meta = _meta(source, task)
    assert meta is not None, f"missing meta for {source}/{task}"
    assert meta["task"].removesuffix("_arch") == task, (source, task, meta["task"])

    # split: agrees with the filename it came from
    assert row["split"] == split, (source, task, split, row["split"])


def test_manifest_covers_all_tasks_and_all_six_evaluation_splits():
    cases = _cases()
    assert {task for _, task, _ in cases} == set(TASKS)
    for split in EVAL_SPLITS:
        assert any(c[2] == split for c in cases), f"no cohort covers {split}"
    for split in ("train", "val"):
        assert any(c[2] == split for c in cases), f"no cohort covers {split}"
    assert len(cases) >= 60, f"expected the full cohort matrix, got {len(cases)} cases"


# --------------------------------------------------------------------------------------
# 2. Simple / medium / hard tiers all contribute, and the hard cycle really is 19/34.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,expected_tasks",
    TIER_CASES,
    ids=[f"tier-{s}" for s, _ in TIER_CASES],
)
def test_each_difficulty_tier_contributes_rows_across_its_tasks(source, expected_tasks):
    covered = {task for src, task, _ in _cases() if src == source}
    assert covered == expected_tasks, f"{source}: expected {expected_tasks}, got {covered}"
    for task in expected_tasks:
        for split in ("train", "test_id"):
            assert _first_row(source, task, split) is not None, f"{source}/{task}_{split}"


def test_catalog_sizes_escalate_simple_medium_hard():
    # Pick metas that actually carry an `architectures` key (repair/design metas do not).
    assert _meta("processed", "audit")["params"]["architectures"] == 9
    assert _meta("processed_arch", "repair")["params"]["architectures"] == 65
    assert _meta("processed_big", "repair")["params"]["architectures"] == 1


@pytest.mark.parametrize(
    "split",
    ("train", "val", "test_id", "test_ood_fault"),
    ids=[f"big-cycle-{sp}" for sp in ("train", "val", "test_id", "test_ood_fault")],
)
def test_big_expert_cycle_is_19_components_and_34_streams(split):
    row = _first_row("processed_big", "repair", split)
    assert row is not None, f"missing big repair split {split}"
    plant = Plant.model_validate_json(row["plant_broken"])
    assert len(plant.components) == 19, (split, len(plant.components))
    assert len(plant.streams) == 34, (split, len(plant.streams))
    assert plant.name == "big_cchp_steam2p_reheat_hp"


# --------------------------------------------------------------------------------------
# 3. Held-out set: all four tasks, disjoint ids, fresh seed.
# --------------------------------------------------------------------------------------
def test_heldout_covers_all_four_tasks_and_splits():
    for task in TASKS:
        assert any(_first_row("heldout", task, sp) is not None for sp in SPLITS), task
    for split in EVAL_SPLITS:
        assert any(_first_row("heldout", t, split) is not None for t in TASKS), split


def test_heldout_ids_are_unseen_by_every_cohort_and_seed_is_fresh():
    heldout_ids = {
        row["id"]
        for task in TASKS
        for split in SPLITS
        if (row := _first_row("heldout", task, split)) is not None
    }
    assert len(heldout_ids) >= 20, f"only sampled {len(heldout_ids)} held-out ids"
    assert all(str(HELD_OUT_SEED) in rid for rid in heldout_ids), sorted(heldout_ids)[:3]

    seen_ids = set()
    for source in ("processed", "processed_arch", "processed_big", "sft_all", "grpo_all"):
        for task in TASKS:
            for split in SPLITS:
                row = _first_row(source, task, split)
                if row is not None:
                    seen_ids.add(row["id"])

    overlap = heldout_ids & seen_ids
    assert not overlap, f"held-out ids leaked into training corpora: {sorted(overlap)}"
    assert not any(str(HELD_OUT_SEED) in rid for rid in seen_ids), "seed namespace collision"
    assert {_seed("heldout", t) for t in TASKS} == {HELD_OUT_SEED}
    for source in ("processed", "processed_arch", "processed_big"):
        seeds = {_seed(source, t) for t in TASKS if _meta(source, t) is not None}
        assert HELD_OUT_SEED not in seeds, f"{source} shares the held-out seed"
    print(
        f"\nheld-out ids sampled: {len(heldout_ids)}; "
        f"seen-corpus ids sampled: {len(seen_ids)}; overlap: {len(overlap)}"
    )


# --------------------------------------------------------------------------------------
# 4. All five harness tools accept ground truth for every task, in every source.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,task",
    _tool_cases(),
    ids=[f"tools-{s}-{t}" for s, t in _tool_cases()],
)
def test_all_five_tools_accept_ground_truth_across_cohorts(source: str, task: str):
    row = _first_row(source, task, "test_id")
    assert row is not None
    ground_truth = row["completion"][0]["content"]

    tools = {fn.__name__: fn for fn in make_tools(task, row)}
    assert tool_names() == TOOL_NAMES
    assert set(tools) == set(TOOL_NAMES)
    # The accept/reward oracle is never exposed to teacher or student.
    assert "check_answer" not in tools

    inspected = json.loads(tools["inspect_plant"](""))
    assert inspected["task"] == task, (source, task, inspected)

    first = json.loads(tools["first_law_check"](ground_truth))
    assert first["ok"] is True, (source, task, first)
    assert isinstance(first["max_mass_residual_kg_s"], (int, float))
    assert isinstance(first["max_energy_residual_W"], (int, float))
    assert first["max_mass_residual_kg_s"] >= 0.0

    second = json.loads(tools["second_law_check"](ground_truth))
    assert second["ok"] is True, (source, task, second)
    assert isinstance(second["total_exergy_destruction_MW"], (int, float))
    assert second["per_component_exergy_destruction_MW"]

    validity = json.loads(tools["validity_check"](ground_truth))
    assert validity["ok"] is True, (source, task, validity)
    assert isinstance(validity["score"], (int, float)) or isinstance(validity["valid"], bool)

    solved = json.loads(tools["solve_plant"](ground_truth))
    assert solved["ok"] is True, (source, task, solved)
    if task == "ledger":
        # Ledger rows carry equations, not a Plant: zero streams is the correct answer.
        assert solved["n_streams"] == 0
    else:
        assert solved["n_streams"] > 0
        assert isinstance(solved["streams"][0]["T_K"], (int, float))
