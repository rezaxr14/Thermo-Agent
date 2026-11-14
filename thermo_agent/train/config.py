"""Training settings. Values come from: Schulman et al. 2025 (LoRA on all linear layers, LR ~10x full fine-tuning,
low rank is enough for RL), Luo et al. 2026 (SFT then GRPO with a group of 8), and TRL's GRPOConfig defaults.

Dual-T4 target (see docs/DUAL_T4.md): full fp16 LoRA, two DDP ranks, no 4-bit quantisation. The T4 is Turing,
so fp16 is the only half-precision format it has fast kernels for; bf16 must not be selected there.
"""
from dataclasses import dataclass, field

BASE_MODEL = "Qwen/Qwen3.5-4B"


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"


@dataclass
class SftSettings:
    lr: float = 2e-4
    epochs: int = 2
    max_length: int = 3072             # prompt (<= ~6.5k chars) + completion
    per_device_batch: int = 2
    grad_accum: int = 8
    max_grad_norm: float = 1.0         # fp16 needs clipping before the optimizer step
    lora: LoraSettings = field(default_factory=LoraSettings)


@dataclass
class GrpoSettings:
    lr: float = 5e-5                   # LoRA RL learning rate; TRL's LoRA example uses 5e-5
    num_generations: int = 8           # group size G
    max_completion_length: int = 320
    temperature: float = 1.0
    per_device_batch: int = 8          # prompts per device; 8 divides G=8, and 8x2x2 = 32 divides by G too
    grad_accum: int = 2
    max_steps: int = 300
    loss_type: str = "dapo"            # TRL default; 'grpo' reproduces the original length normalisation
    beta: float = 0.0                  # no KL term -> no reference model in memory
    mask_truncated: bool = True
    max_grad_norm: float = 1.0
    lora: LoraSettings = field(default_factory=lambda: LoraSettings(r=8, alpha=16))
