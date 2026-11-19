"""Training-path audit 2026-11-19 item 1: every loader reads task AND tier.

The GRPO sampler used to bucket every real row into ``<missing>`` because the
corpora have no literal tier column; SFT used to skip ``frontier_*`` shards
altogether. These tests pin both fixes with content-less-free, CPU-only rows.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import pytest

from thermo_agent.synth import tiers as tier_mod
from thermo_agent.train import data, grpo, sft


def repair_row(rid: str, n_components: int, **extra) -> dict:
    """A minimal repair row whose difficulty is measurable by synth.tiers."""
    row = {
        "id": rid,
        "task": "repair",
        "plant_broken": json.dumps({"components": [{"id": f"C{i}"} for i in range(n_components)]}),
        "prompt": [{"role": "user", "content": f"row {rid}"}],
        "completion": [{"role": "assistant", "content": "<diagnosis/><edits/>"}],
    }
    row.update(extra)
    return row


def write_shard(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_attach_task_tier_derives_real_equal_thirds_without_a_tier_column():
    rows = [repair_row(f"p{i}", 4 + i) for i in range(9)]
    out, info = data.attach_task_tier(rows, "repair")
    assert all("task" in r and "tier" in r for r in out)
    assert info["unresolved"] == []
    counts = Counter(r["tier"] for r in out)
    assert counts == {"easy": 3, "medium": 3, "hard": 3}
    # Rows keep their input order; the tiers themselves are the three size-sorted thirds.
    assert [r["tier"] for r in out] == [tier_mod.TIERS[i // 3] for i in range(9)]


def test_attach_task_tier_keeps_the_two_variants_of_one_plant_together():
    rows = [repair_row("plant7", 5)] + [repair_row("plant7#v1", 5), repair_row("plant7#v2", 5)]
    out, info = data.attach_task_tier(rows, "repair")
    assert len({r["tier"] for r in out}) == 1
    assert info["unresolved"] == []
    assert info["derived"] == {"plant7": out[0]["tier"]}


def test_explicit_tier_wins_and_a_typo_is_a_hard_error():
    rows = [repair_row("a", 4, tier="Medium"), repair_row("b", 4, difficulty="hard")]
    out, info = data.attach_task_tier(rows, "repair")
    assert [r["tier"] for r in out] == ["medium", "hard"]
    assert info["derived"] == {}
    with pytest.raises(ValueError, match="invalid tier"):
        data.attach_task_tier([repair_row("c", 4, tier="mediumd")], "repair")


def test_unscorable_row_is_counted_and_joint_rows_refuses_it(tmp_path):
    rows = [repair_row("p0", 5), {"id": "nop", "task": "repair", "prompt": [], "completion": []}]
    _, info = data.attach_task_tier(rows, "repair")
    assert info["unresolved"] == ["nop"]
    with pytest.raises(ValueError, match="no measurable artefact"):
        data.assert_tiers_resolved("repair", info)
    write_shard(tmp_path / "repair_train.jsonl", rows)
    with pytest.raises(ValueError, match="no measurable artefact"):
        grpo.joint_rows(tmp_path)


def test_joint_rows_samples_uniformly_over_task_x_tier_from_real_style_rows(tmp_path):
    """Nine repair rows without a tier column must become 9 cells, not one."""
    for task in grpo.JOINT_TASKS:
        # Repair has no tier column (derive it); the other tasks carry an explicit
        # column, exercising both accepted forms of the decided tier field.
        rows = []
        for i in range(9):
            extra = {} if task == "repair" else {"tier": tier_mod.TIERS[i % 3]}
            rows.append(repair_row(f"{task}-p{i}", 4 + i, task=task, **extra))
        write_shard(tmp_path / f"{task}_train.jsonl", rows)
    loaded = grpo.joint_rows(tmp_path)
    assert len(loaded) == 4 * 9
    cells = {(r["task"], r["tier"]) for r in loaded}
    assert cells == {(t, tier) for t in grpo.JOINT_TASKS for tier in tier_mod.TIERS}
    sampled = grpo.sample_joint(loaded, 48, seed=0)
    counts = Counter((r["task"], r["tier"]) for r in sampled)
    assert len(counts) == 12
    assert set(counts.values()) == {4}  # 48 / 12 cells: task x tier really is uniform
    assert [r["id"] for r in sampled] == [r["id"] for r in grpo.sample_joint(loaded, 48, seed=0)]


def test_frontier_shards_are_read_by_both_loaders(tmp_path):
    deterministic = repair_row("det", 6)
    frontier = repair_row("front", 7)
    write_shard(tmp_path / "repair_train.jsonl", [deterministic])
    write_shard(tmp_path / "frontier_repair_train.jsonl", [frontier])
    paths = data.shard_paths(tmp_path, "repair", "train")
    assert [p.name for p in paths] == ["frontier_repair_train.jsonl", "repair_train.jsonl"]
    sft_rows = sft.read_rows(tmp_path, ["repair"], "train")
    assert {r["id"] for r in sft_rows} == {"det", "front"}
    assert all(r["task"] == "repair" and r["tier"] in tier_mod.TIERS for r in sft_rows)
    joint = grpo.joint_rows(tmp_path)  # other tasks are absent; the repair shards are still loaded
    assert {r["id"] for r in joint if r["task"] == "repair"} == {"det", "front"}


def test_sft_read_rows_attaches_task_and_tier_on_plain_corpora(tmp_path):
    write_shard(tmp_path / "audit_train.jsonl", [
        {"id": f"a{i}", "task": "audit", "plant": json.dumps({"components": [{"id": "X"}] * (3 + i)}),
         "prompt": [{"role": "user", "content": "x"}], "completion": [{"role": "assistant", "content": "y"}]}
        for i in range(3)
    ])
    rows = sft.read_rows(tmp_path, ["audit"], "train")
    assert {r["tier"] for r in rows} <= set(tier_mod.TIERS)
    assert {r["task"] for r in rows} == {"audit"}
