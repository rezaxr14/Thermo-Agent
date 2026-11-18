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
    # TRL checks the per-device prompt batch as well as the effective batch; both must hold whole groups.
    if per_device % num_generations:
        return False
    return effective_batch(per_device, grad_accum, world_size) % num_generations == 0


def assert_grpo_batches(per_device: int, grad_accum: int, world_size: int, num_generations: int) -> None:
    eff = effective_batch(per_device, grad_accum, world_size)
    if not grpo_batches_divisible(per_device, grad_accum, world_size, num_generations):
        raise ValueError(
            f"effective batch {per_device}x{grad_accum}x{world_size}={eff} is not divisible by "
            f"num_generations={num_generations} (per-device batch {per_device} must also be a multiple); the "
            "dual-T4 setting is --per-device-batch 4 --grad-accum 2"
        )


def barrier() -> None:
    """Wait for all ranks. No-op for a single process; used before a shared checkpoint is read."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def model_init_kwargs(dtype: str) -> dict:
    """TRL passes this to `from_pretrained`; transformers renamed `torch_dtype` to `dtype` in v5."""
    import inspect
    import torch
    from transformers import AutoModelForCausalLM

    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    return {"dtype": getattr(torch, dtype)} if "dtype" in params else {"torch_dtype": getattr(torch, dtype)}


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
