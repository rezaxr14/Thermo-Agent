import json, random
import pytest
from thermo_agent.synth.build_repair_data import build as build_repair
from thermo_agent.ledger.data import build as build_ledger
from thermo_agent.train import reward_diagnostics as RD
from thermo_agent.train.perturb import simulate_policy, mutate

def _rows(d, task): return [json.loads(l) for l in open(d / f"{task}_train.jsonl")]

def test_simulated_policy_interpolates_between_expert_and_damage():
    rng = random.Random(0); e = "<fix>\nFIX T0 = 298.15\n</fix>"
    assert all(simulate_policy(e, rng, 0.0) == e for _ in range(20))
    assert sum(simulate_policy(e, rng, 1.0) != e for _ in range(50)) >= 45

@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("rd"); build_repair(d, 90, seed=3); build_ledger(d, 60, seed=3); return d

def test_perfect_policy_has_no_learning_signal_and_worse_policies_have_spread(data):
    for task in ("repair", "ledger"):
        full = RD.diagnose(_rows(data, task), task, [0.0, 0.3, 0.6, 1.0], groups=6, G=6); r = full["by_policy_error_rate"]; assert full["fresh_evaluation_ms"] > 0
        assert r["0.0"]["zero_variance_groups"] == 1.0 and r["0.0"]["mean_within_group_std"] == 0.0
        assert r["0.3"]["mean_within_group_std"] > 0 and r["0.0"]["mean_reward"] > r["0.6"]["mean_reward"] > r["1.0"]["mean_reward"] - 1e-9
        assert r["0.3"]["ms_per_completion"] > 0 and set(r["0.3"]["channel_means"]) and r["0.3"]["zero_variance_expected_if_all_correct_or_all_wrong_alike"] == pytest.approx(0.7 ** 6, abs=1e-3)

def test_advice_flags_groups_without_spread():
    flat = {"fresh_evaluation_ms": 900.0, "by_policy_error_rate": {"0.3": {"zero_variance_groups": 0.8, "mean_within_group_std": 0.01, "ms_per_completion": 900}}}
    n = RD.advise("x", flat); assert len(n) == 3
    ok = {"fresh_evaluation_ms": 20.0, "by_policy_error_rate": {"0.3": {"zero_variance_groups": 0.1, "mean_within_group_std": 0.4, "ms_per_completion": 10}}}; assert RD.advise("x", ok) == []

def test_none_rewards_are_skipped_like_trl_does():
    assert RD.total_reward([[1.0], [None], [0.5]], [2.0, 5.0, 4.0], 0) == pytest.approx(4.0)
