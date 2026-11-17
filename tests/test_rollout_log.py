"""GRPO rollouts must be saved with their channel scores and retain the names reward_weights uses."""
import json

from thermo_agent.train.rollout_log import wrap_rewards


def test_wrapped_rewards_keep_names_and_log_one_line_per_call(tmp_path, monkeypatch):
    from thermo_agent.train import ddp
    monkeypatch.setattr(ddp, "local_rank_from_env", lambda env=None: 0)

    def r_physics(completions=None, **kwargs):
        return [1.0 for _ in completions]

    def r_format(completions=None, **kwargs):
        return [0.5 for _ in completions]

    funcs = wrap_rewards([r_physics, r_format], tmp_path / "rollouts.jsonl", stage="grpo_ledger")
    assert [f.__name__ for f in funcs] == ["r_physics", "r_format"]
    assert funcs[0](["answer"], prompts=["problem"]) == [1.0]
    assert funcs[1](["answer"], prompts=["problem"]) == [0.5]
    lines = [json.loads(l) for l in (tmp_path / "rollouts.jsonl").open(encoding="utf-8")]
    assert len(lines) == 2 and lines[0]["stage"] == "grpo_ledger"
    assert lines[0]["reward_fn"] == "r_physics" and lines[0]["prompt"] == "problem"
    assert lines[0]["completions"] == ["answer"] and lines[0]["rewards"] == [1.0]


def test_no_path_means_no_io():
    def f(completions=None, **kwargs):
        return [0.0]
    assert wrap_rewards([f], None)[0](["x"]) == [0.0]
