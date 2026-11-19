"""Fails if a decided value (docs/DECISIONS_2026-11-19.md) drifts in code. Change a value only by a new DECISIONS entry."""
from thermo_agent.train.config import SftSettings, GrpoSettings, LoraSettings


def test_sft_values():
    s = SftSettings()
    assert (s.lr, s.epochs, s.max_length, s.per_device_batch, s.grad_accum, s.max_grad_norm) == (1e-4, 3, 8192, 2, 8, 1.0)
    assert (s.save_steps, s.eval_steps) == (50, 50)
    assert (s.lr_scheduler, s.warmup_ratio, s.weight_decay, s.early_stopping_patience) == ("cosine", 0.05, 0.0, 3)
    assert (s.lora.r, s.lora.alpha, s.lora.dropout, s.lora.target_modules) == (32, 64, 0.05, "all-linear")


def test_grpo_values():
    g = GrpoSettings()
    assert (g.lr, g.beta, g.loss_type, g.scale_rewards, g.mask_truncated) == (2e-5, 0.0, "dapo", "group", True)
    assert (g.max_completion_length, g.per_device_batch, g.prompts_per_step) == (4096, 2, 4)
    assert (g.temperature, g.top_p, g.top_k, g.min_p, g.repetition_penalty) == (1.0, 1.0, 0, 0.0, 1.0)
    assert (g.lora.r, g.lora.alpha) == (8, 16)


def test_grpo_prompts_per_step_constant_across_k():
    for k in (4, 8, 16):
        for world in (2,):
            accum = max(1, (GrpoSettings().prompts_per_step * k) // (GrpoSettings().per_device_batch * world))
            assert accum == k                                   # 4 prompts per step for every k
            assert GrpoSettings().per_device_batch * accum * world == 4 * k


def test_eval_and_agent_limits():
    from thermo_agent.eval import model_eval as me
    from thermo_agent.train import grpo
    from thermo_agent.teacher import agent_graph as ag
    assert (me._DEFAULT_TOOL_CALLS, me._TOOL_MAX_NEW_TOKENS, me._TEXT_MAX_NEW_TOKENS) == (8, 4096, 4096)
    assert grpo.DEFAULT_MAX_TOOL_CALLING_ITERATIONS == 6 and grpo.DEFAULT_MAX_COMPLETION_LENGTH == 4096
    assert ag.DEFAULT_MAX_ROUNDS == 6
