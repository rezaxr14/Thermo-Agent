"""C4a / C9: derived grad_accum, joint-task sampling and dispatch, stage names, rollout logging.

CPU only: no GPU, no TRL trainer, no reward implementation is called on a real plant.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from thermo_agent.train import grpo, rollout_log  # noqa: E402
from thermo_agent.train.config import GrpoSettings  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. C4a: derived grad_accum
# --------------------------------------------------------------------------- #
def test_derived_grad_accum_is_k_for_the_decided_k_values():
    for k in (4, 8, 16):
        accum = grpo.derive_grad_accum(k)
        assert accum == k
        # 4 prompts * k completions = per_device * accum * world
        sequences = GrpoSettings().per_device_batch * accum * 2
        assert sequences == 4 * k
        assert sequences % k == 0
        assert sequences // k == 4                       # exactly 4 prompts per optimizer step
    # The formula is derived, not a constant: other valid topologies give other accum values.
    assert grpo.derive_grad_accum(8, per_device_batch=4, world_size=2) == 4


def test_non_decided_k_is_rejected():
    for k in (1, 6, 3, 32):
        with pytest.raises(ValueError, match="not a decided GRPO group size"):
            grpo.derive_grad_accum(k)
    with pytest.raises(ValueError, match="k=1"):
        grpo.validate_k(1)
    with pytest.raises(ValueError, match="k=6"):
        grpo.validate_k(6)


def test_inexact_derivation_raises_with_a_clear_message():
    # 4*4/(3*2) is not an integer: refuse instead of silently changing prompts per step.
    with pytest.raises(ValueError, match="C4a: cannot derive grad_accum") as exc:
        grpo.derive_grad_accum(4, per_device_batch=3, world_size=2)
    assert "not an integer" in str(exc.value)
    assert "exactly 4 prompts" in str(exc.value)


def test_cli_override_still_wins(capsys):
    src = (ROOT / "thermo_agent" / "train" / "grpo.py").read_text(encoding="utf-8")
    assert "g.grad_accum = a.grad_accum if a.grad_accum is not None else _derived_accum" in src
    assert "[C4a]" in src
    # The derivation is printed for every run.
    assert "prompts_per_step*k/(per_device_batch*world)" in src


# --------------------------------------------------------------------------- #
# 2. C9: joint corpus and uniform (task x tier) sampler
# --------------------------------------------------------------------------- #
TIERS = ("easy", "hard")


def _write_joint_corpus(tmp_path: pathlib.Path, per_cell: int = 3) -> pathlib.Path:
    for task in grpo.JOINT_TASKS:
        rows = []
        for tier in TIERS:
            for i in range(per_cell):
                rows.append({"id": f"{task}-{tier}-{i}", "task": task, "tier": tier,
                             "prompt": [{"role": "user", "content": f"{task} {tier} {i}"}]})
        (tmp_path / f"{task}_train.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return tmp_path


def test_joint_rows_carry_task_and_tier(tmp_path):
    rows = grpo.joint_rows(_write_joint_corpus(tmp_path))
    assert len(rows) == 4 * len(TIERS) * 3
    assert {r["task"] for r in rows} == set(grpo.JOINT_TASKS)
    assert {(r["task"], r["tier"]) for r in rows} == {(t, tier) for t in grpo.JOINT_TASKS for tier in TIERS}
    # Every reward column used by any task exists (with neutral defaults) in every joint row,
    # EXCEPT the scoring-only ref_* answers: those are joined at scoring time, never stored
    # in a training row. id/record_id are kept as join keys.
    for row in rows:
        assert not any(str(k).startswith("ref_") for k in row)
        assert "id" in row and "record_id" in row
        for _, _, cols in grpo.TASKS.values():
            assert {c for c in cols if not c.startswith("ref_")} <= set(row)
    assert grpo.row_tier({"tier": "hard"}) == "hard"
    assert grpo.row_tier({}) == grpo.UNKNOWN_TIER


def test_joint_sampler_is_roughly_uniform_over_task_x_tier(tmp_path):
    rows = grpo.joint_rows(_write_joint_corpus(tmp_path, per_cell=3))
    sample = grpo.sample_joint(rows, 80, seed=0)
    assert len(sample) == 80
    counts = Counter((r["task"], r["tier"]) for r in sample)
    assert len(counts) == 4 * len(TIERS)
    assert set(counts.values()) == {10}                       # 80 / 8 cells
    per_task = Counter(r["task"] for r in sample)
    assert set(per_task.values()) == {20}
    # Same seed -> same prompt order (C1: identical prompt order for k=4/8/16).
    assert [r["id"] for r in sample] == [r["id"] for r in grpo.sample_joint(rows, 80, seed=0)]


def test_joint_sampler_recycles_small_cells_and_handles_missing_tier(tmp_path):
    rows = grpo.joint_rows(_write_joint_corpus(tmp_path, per_cell=1))
    sample = grpo.sample_joint(rows, 9, seed=1)
    assert len(sample) == 9
    assert {r["task"] for r in sample} == set(grpo.JOINT_TASKS)
    plain = [{"task": "ledger", "prompt": "x"}]
    assert grpo.sample_joint(plain, 3)[0] is plain[0]


# --------------------------------------------------------------------------- #
# 3. C9: per-row reward dispatch
# --------------------------------------------------------------------------- #
def test_reward_funcs_for_task_and_row_and_unknown_task():
    for task, prefix in (("repair", "r_"), ("design", "d_"),
                         ("audit", "a_"), ("ledger", "l_")):
        funcs, weights = grpo.reward_funcs_for_task(task)
        assert funcs and weights and len(funcs) == len(weights)
        assert all(f.__name__.startswith(prefix) for f in funcs)
        row_funcs, row_weights = grpo.reward_funcs_for_row({"task": task})
        assert [f.__name__ for f in row_funcs] == [f.__name__ for f in funcs]
        assert row_weights == weights
    with pytest.raises(ValueError, match="unknown GRPO"):
        grpo.reward_funcs_for_task("bogus")
    with pytest.raises(ValueError, match="unknown GRPO"):
        grpo.reward_funcs_for_row({"task": "bogus"})
    with pytest.raises(ValueError, match="unknown GRPO"):
        grpo.reward_funcs_for_row({})


def test_joint_rewards_are_the_union_with_aligned_weights():
    funcs, weights, owner = grpo.joint_rewards()
    assert len(funcs) == len(weights) == 22
    assert len({f.__name__ for f in funcs}) == len(funcs)     # TRL maps weights by name
    assert set(owner.values()) == set(grpo.JOINT_TASKS)
    for fn, weight in zip(funcs, weights):
        task = owner[fn.__name__]
        t_funcs, t_weights = grpo.reward_funcs_for_task(task)
        expected = dict(zip([f.__name__ for f in t_funcs], t_weights))
        assert weight == expected[fn.__name__]


def test_two_rows_in_one_batch_get_their_own_tasks_rewards():
    def r_probe(prompts=None, completions=None, **kw):
        return [float(v) for v in kw["mark"]]

    def l_probe(prompts=None, completions=None, **kw):
        return [10.0 * float(v) for v in kw["mark"]]

    funcs = grpo.dispatch_rewards([r_probe, l_probe],
                                  owner={"r_probe": "repair", "l_probe": "ledger"})
    assert [f.__name__ for f in funcs] == ["r_probe", "l_probe"]
    batch = dict(prompts=["p0", "p1"], completions=["c0", "c1"],
                 task=["repair", "ledger"], mark=[1, 2])
    # Repair rows are scored only by the repair probe; ledger rows only by the ledger probe.
    assert funcs[0](**batch) == [1.0, 0.0]
    assert funcs[1](**batch) == [0.0, 20.0]

    bad = dict(batch, task=["repair", "bogus"])
    with pytest.raises(ValueError, match="unknown GRPO row task"):
        funcs[0](**bad)
    with pytest.raises(ValueError, match="dataset column"):
        funcs[0](prompts=["p"], completions=["c"])


def test_dispatcher_preserves_names_for_real_joint_rewards():
    funcs = grpo.dispatch_rewards()
    registry = grpo.joint_rewards()[2]
    assert [f.__name__ for f in funcs] == [f.__name__ for f in grpo.joint_rewards()[0]]
    # A batch of only repair rows must get zeros from every non-repair function.
    batch = dict(prompts=["p"], completions=["c"], task=["repair"])
    non_repair = [f for f in funcs if registry[f.__name__] != "repair"]
    assert all(f(**batch) == [0.0] for f in non_repair)


# --------------------------------------------------------------------------- #
# 4. Stage names: write joint, read both
# --------------------------------------------------------------------------- #
def test_stage_names_write_joint_and_read_legacy(tmp_path):
    assert grpo.stage_name("audit", 8) == "grpo_k8"
    assert grpo.stage_names_for_reading("audit", 8) == ("grpo_k8", "grpo_audit_k8")
    (tmp_path / "grpo_audit_k8").mkdir()
    assert grpo.find_stage_dir(tmp_path, "audit", 8).name == "grpo_audit_k8"
    (tmp_path / "grpo_k8").mkdir()
    assert grpo.find_stage_dir(tmp_path, "audit", 8).name == "grpo_k8"


def test_eval_runner_reads_both_stage_names():
    src = (ROOT / "kaggle" / "eval_runner.py").read_text(encoding="utf-8")
    assert "stage_names_for_reading(TASK, _k)" in src
    assert 'f"grpo_{TASK}_k{_k}"' not in src
    train = (ROOT / "kaggle" / "train_runner.py").read_text(encoding="utf-8")
    assert 'GRPO_STAGE = f"grpo_k{GRPO_K}"' in train


# --------------------------------------------------------------------------- #
# 5. Per-batch rollout logging (additive, no reward change)
# --------------------------------------------------------------------------- #
def test_batch_logger_writes_one_line_per_batch_with_all_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout_log.ddp, "local_rank_from_env", lambda env=None: 0)
    path = tmp_path / "runs" / "grpo_k4" / "rollouts.jsonl"

    def a(prompts=None, completions=None, **kw):
        return [1.0, 0.0]

    def b(prompts=None, completions=None, **kw):
        return [0.5, 0.25]

    funcs = rollout_log.RolloutBatchLogger([1.0, 1.0], 2, path, stage="grpo_k4").wrap([a, b])
    assert [f.__name__ for f in funcs] == ["a", "b"]
    tool_turn = {"role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "inspect_plant"}}]}
    tool_result = {"role": "tool", "name": "inspect_plant", "content": "{}"}
    final = {"role": "assistant", "content": "<edits>SET X = 1</edits>"}
    for _ in range(2):
        completions = [[tool_turn, tool_result, final], [tool_result]]  # second one is truncated
        assert funcs[0](prompts=["p", "q"], completions=completions) == [1.0, 0.0]
        assert funcs[1](prompts=["p", "q"], completions=completions) == [0.5, 0.25]

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    first = lines[0]
    for key in ("stage", "batch", "reward", "reward_values", "zero_variance_share",
                "groups", "truncation_rate", "turns", "tool_calls", "n"):
        assert key in first
    assert first["stage"] == "grpo_k4" and first["batch"] == 1
    assert first["reward_values"] == [1.5, 0.25] and first["reward"] == 0.875
    assert first["zero_variance_share"] == 0.0 and first["groups"] == 1
    assert first["truncation_rate"] == 0.5
    assert first["turns"] == 1.0 and first["tool_calls"] == 1.5  # (2+0)/2 assistant turns, (2+1)/2 calls
    assert lines[1]["batch"] == 2


def test_batch_logger_zero_variance_and_rank_guard(tmp_path, monkeypatch):
    path = tmp_path / "rollouts.jsonl"
    logger = rollout_log.RolloutBatchLogger([1.0], 2, path)

    def same(prompts=None, completions=None, **kw):
        return [1.0, 1.0]

    wrapped = logger.wrap([same])[0]
    wrapped(prompts=["p", "q"], completions=["a", "b"])
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["zero_variance_share"] == 1.0
    assert rec["turns"] == 1.0 and rec["tool_calls"] == 0.0

    monkeypatch.setattr(rollout_log.ddp, "local_rank_from_env", lambda env=None: 1)
    wrapped(prompts=["p", "q"], completions=["a", "b"])
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
