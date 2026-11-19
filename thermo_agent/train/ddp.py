"""Data-parallel helpers for the dual-T4 target.

Why DDP and not FSDP/ZeRO: the two T4s sit on the same host and share PCIe, with no NVLink. DDP replicates the
model, so the only tensors that cross PCIe are the LoRA gradients — tens of MB in fp16 — and Accelerate's
accumulation context (used by the HF Trainer) skips the all-reduce on every micro-step except the last. Sharding
the 4B base weights with FSDP/ZeRO-3 would stream roughly 8 GB per optimizer step over that same link, which is
what we are trying to avoid. So: replicate, shrink what crosses, accumulate locally.

The helpers here are pure Python except `describe_runtime`, which imports torch lazily, so they can be tested on
a CPU-only machine.
"""
from __future__ import annotations

import json
import os

SM_AMPERE = 8

DTYPES = ("bfloat16", "float16")


def dtype_for_capability(major: int, minor: int = 0) -> str:
    """Turing (T4, 7.5) has no bf16 kernels; Ampere and newer prefer bf16."""
    return "bfloat16" if major >= SM_AMPERE else "float16"


def check_dtype_supported(dtype: str, capability: tuple[int, int]) -> None:
    """Refuse bf16 on pre-Ampere hardware; warn when fp16 is used where bf16 exists."""
    major, minor = capability
    if dtype == "bfloat16" and major < SM_AMPERE:
        raise ValueError(
            f"bfloat16 is not supported on compute capability {major}.{minor} (T4 is 7.5); "
            "pass --dtype float16"
        )


def world_size_from_env(env: dict[str, str] | None = None) -> int:
    """Number of data-parallel processes. torchrun sets WORLD_SIZE; a bare run is one process."""
    env = os.environ if env is None else env
    raw = env.get("WORLD_SIZE", "1")
    try:
        world = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE={raw!r} is not an integer") from exc
    if world < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world}")
    return world


def local_rank_from_env(env: dict[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    return int(env.get("LOCAL_RANK", "0"))


def is_distributed(env: dict[str, str] | None = None) -> bool:
    return world_size_from_env(env) > 1


def effective_batch(per_device: int, grad_accum: int, world_size: int = 1) -> int:
    """Examples per optimizer step, summed over ranks."""
    if min(per_device, grad_accum, world_size) < 1:
        raise ValueError("per_device, grad_accum and world_size must be >= 1")
    return per_device * grad_accum * world_size


def grpo_batches_divisible(per_device: int, grad_accum: int, world_size: int, num_generations: int) -> bool:
    """TRL shards prompts over ranks, then draws `num_generations` completions per prompt.

    The global prompt batch has to be split into whole groups of `num_generations`, so the effective batch
    must be a multiple of the group size.
    """
    if num_generations < 1 or per_device < 1 or grad_accum < 1 or world_size < 1:
        return False
    # TRL 1.18.1 (GRPOConfig.__post_init__): generation_batch_size = per_device x world x accum must be divisible by
    # num_generations. The per-device batch itself need not be a multiple of k (that was the pre-0.18 rule).
    return effective_batch(per_device, grad_accum, world_size) % num_generations == 0


def grpo_batch_error(per_device: int, grad_accum: int, world_size: int, num_generations: int) -> str:
    """Exact, actionable message for a batch that cannot hold whole GRPO groups."""
    eff = effective_batch(per_device, grad_accum, world_size)
    return (
        f"GRPO batch is not group-aligned: per_device_train_batch_size={per_device} x grad_accum={grad_accum} x "
        f"world={world_size} = {eff} must be divisible by num_generations (k={num_generations}). For k={num_generations} "
        f"use --per-device-batch 2 --grad-accum {num_generations} on two ranks (4 prompts per optimizer step)."
    )


def assert_grpo_batches(per_device: int, grad_accum: int, world_size: int, num_generations: int) -> None:
    """Refuse a batch that would split a group of `num_generations` completions across steps/ranks."""
    if not grpo_batches_divisible(per_device, grad_accum, world_size, num_generations):
        raise ValueError(grpo_batch_error(per_device, grad_accum, world_size, num_generations))


def assert_fp16_lora_ddp(
    dtype: str,
    init_kwargs: dict | None,
    lora_config: object | None,
    *,
    deepspeed: object = None,
    fsdp: object = None,
) -> None:
    """Guard the documented recipe: full fp16 LoRA, DDP replication, no QLoRA/FSDP/ZeRO.

    Kept as pure Python so the CPU tests can assert the recipe without importing torch or building a model.
    """
    if dtype != "float16":
        raise ValueError("Only fp16 LoRA DDP is supported on the dual-T4 target; pass --dtype float16")
    if "quantization_config" in (init_kwargs or {}):
        raise ValueError("QLoRA (a 4-bit quantization_config) is not part of the fp16 LoRA DDP recipe; drop --qlora")
    if deepspeed not in (None, "", {}):
        raise ValueError("DeepSpeed/ZeRO is not used: the recipe replicates the model with DDP")
    if fsdp not in (None, "", {}):
        raise ValueError("FSDP is not used: the recipe replicates the model with DDP")
    if lora_config is None:
        raise ValueError("The recipe needs a LoRA config (full fine-tuning is not used)")


def barrier() -> None:
    """Wait for all ranks. No-op for a single process; used before a shared checkpoint is read."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def model_init_kwargs(dtype: str) -> dict:
    """TRL passes this to `from_pretrained`; transformers renamed `torch_dtype` to `dtype` in v5.

    `attn_implementation="sdpa"` is deliberate: Qwen3.5's hybrid layers are eager by default on
    this config, and one full-attention layer at ~5.7k tokens materialises a ~1 GiB score matrix
    (16 heads x seq^2 x fp16), which is exactly the allocation that OOM'd smoke v26 on a T4.
    SDPA's memory-efficient kernel is available on Turing for fp16 and does not change weights.
    """
    import inspect
    import os
    import torch
    from transformers import AutoModelForCausalLM

    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    kwargs = {"dtype": getattr(torch, dtype)} if "dtype" in params else {"torch_dtype": getattr(torch, dtype)}
    kwargs["attn_implementation"] = os.environ.get("THERMO_ATTN", "sdpa")
    return kwargs


def describe_runtime() -> dict:
    """GPU names, capabilities and the chosen dtype. Imports torch lazily so CPU tests stay light."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "world_size": world_size_from_env(), "dtype": None, "gpus": []}
    gpus = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        gpus.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": f"{major}.{minor}",
                "dtype": dtype_for_capability(major, minor),
                "total_mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
            }
        )
    return {
        "cuda": True,
        "world_size": world_size_from_env(),
        "local_rank": local_rank_from_env(),
        "torch": torch.__version__,
        "gpus": gpus,
    }


def topology_line() -> str:
    """One-line JSON record for the training log, safe to call before any model is loaded."""
    try:
        return json.dumps(describe_runtime(), sort_keys=True)
    except Exception as exc:  # torch missing is a real possibility on a CPU box
        return json.dumps({"error": str(exc)})
