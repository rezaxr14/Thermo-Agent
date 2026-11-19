"""DDP and dtype guards used by the dual-T4 training path. Pure Python, no torch needed."""
from thermo_agent.train import ddp
from thermo_agent.train.config import GrpoSettings, SftSettings


def test_t4_gets_float16_and_ampere_gets_bfloat16():
    assert ddp.dtype_for_capability(7, 5) == "float16"
    assert ddp.dtype_for_capability(8, 0) == "bfloat16"
    assert ddp.dtype_for_capability(8, 9) == "bfloat16"
    assert ddp.dtype_for_capability(9, 0) == "bfloat16"


def test_bfloat16_is_refused_on_turing():
    ddp.check_dtype_supported("float16", (7, 5))
    ddp.check_dtype_supported("bfloat16", (8, 0))
    try:
        ddp.check_dtype_supported("bfloat16", (7, 5))
    except ValueError as exc:
        assert "float16" in str(exc)
    else:
        raise AssertionError("bf16 on T4 should be refused")


def test_world_size_and_local_rank_from_env():
    assert ddp.world_size_from_env({}) == 1
    assert ddp.world_size_from_env({"WORLD_SIZE": "2"}) == 2
    assert ddp.local_rank_from_env({"LOCAL_RANK": "1"}) == 1
    assert ddp.is_distributed({"WORLD_SIZE": "2"})
    assert not ddp.is_distributed({})
    for bad in ("two", "0", "-1"):
        try:
            ddp.world_size_from_env({"WORLD_SIZE": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"WORLD_SIZE={bad} should be refused")


def test_effective_batch_multiplies_ranks():
    assert ddp.effective_batch(2, 8, 1) == 16
    assert ddp.effective_batch(2, 8, 2) == 32
    for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        try:
            ddp.effective_batch(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should be refused")


def test_grpo_groups_divide_the_generation_batch():
    # TRL 1.18.1: generation_batch_size = per_device x world x accum must be divisible by num_generations.
    assert ddp.grpo_batches_divisible(8, 8, 2, 8)
    assert ddp.grpo_batches_divisible(2, 8, 2, 8)         # 2x8x2 = 32
    assert ddp.grpo_batches_divisible(2, 16, 2, 16)       # 2x16x2 = 64
    assert not ddp.grpo_batches_divisible(1, 4, 1, 8)     # 4 sequences cannot hold a group of 8
    assert not ddp.grpo_batches_divisible(1, 1, 2, 8)
    assert not ddp.grpo_batches_divisible(2, 8, 2, 3)
    try:
        ddp.assert_grpo_batches(1, 1, 2, 8)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("a 2-prompt batch cannot hold groups of 8")


def test_dual_t4_defaults_are_consistent():
    """Two ranks, the shipped batch sizes: SFT 2x8x2; GRPO keeps 4 prompts per step, so accum = k and 2 x k x 2 = 4k."""
    s, g = SftSettings(), GrpoSettings()
    assert ddp.effective_batch(s.per_device_batch, s.grad_accum, 2) == 32
    # TRL 1.18.1 only requires generation_batch_size (= per_device x world x accum) % num_generations == 0
    for k in (4, 8, 16):
        accum = max(1, (g.prompts_per_step * k) // (g.per_device_batch * 2))
        assert ddp.effective_batch(g.per_device_batch, accum, 2) == g.prompts_per_step * k
        assert ddp.grpo_batches_divisible(g.per_device_batch, accum, 2, k)
    assert s.max_grad_norm == 1.0 and g.max_grad_norm == 1.0
