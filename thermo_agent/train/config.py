"""Training settings. Values come from: Schulman et al. 2025 (LoRA on all linear layers, LR ~10x full fine-tuning,
low rank is enough for RL), Luo et al. 2026 (SFT then GRPO with a group of 4), and TRL's GRPOConfig defaults.

Dual-T4 target (see docs/DUAL_T4.md): full fp16 LoRA, two DDP ranks, no 4-bit quantisation. The T4 is Turing,
so fp16 is the only half-precision format it has fast kernels for; bf16 must not be selected there.
"""
from dataclasses import dataclass, field, fields

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
    epochs: int = 1                    # one pass over the 20-variant corpus is ~30 h on two T4s; more would not fit quota
    max_length: int = 8192             # prompt + completion, both gated at 8000 tokens by synth.combine
    per_device_batch: int = 2
    grad_accum: int = 8
    max_grad_norm: float = 1.0         # fp16 needs clipping before the optimizer step
    # An epoch of the 20-variant corpus is ~30 h on two T4s, far longer than one 12 h Kaggle session. Save and
    # validate every few hundred steps so a dropped session resumes from its last checkpoint instead of restarting.
    save_steps: int = 200
    eval_steps: int = 200
    lora: LoraSettings = field(default_factory=LoraSettings)


@dataclass
class GrpoSettings:
    lr: float = 5e-5                   # LoRA RL learning rate; TRL's LoRA example uses 5e-5
    num_generations: int = 4           # group size G; 8 x 320-token rollouts on a T4 exceed 15 GB
    max_completion_length: int = 320
    temperature: float = 1.0
    per_device_batch: int = 4          # TRL requires the per-device batch to be a multiple of G; 4 x G=4 = 16 sequences
    grad_accum: int = 2                # 4 x 2 x 2 ranks = 16 prompts per optimizer step
    max_steps: int = 300
    save_steps: int = 50
    scale_rewards: str = "group"           # normalise advantages inside each group, not across prompts
    loss_type: str = "dapo"            # TRL default; 'grpo' reproduces the original length normalisation
    beta: float = 0.0                  # no KL term -> no reference model in memory
    mask_truncated: bool = True
    max_grad_norm: float = 1.0
    lora: LoraSettings = field(default_factory=lambda: LoraSettings(r=8, alpha=16))


def supported(cls, **kwargs) -> dict:
    """Keep only the arguments this installed Trainer/GRPO config actually accepts.

    Kaggle's TRL version lags the authoring environment; passing an unknown keyword kills a multi-hour run in the
    first second. Dropping an unsupported optimisation flag is safe, dropping a training setting is not, so the
    dropped keys are printed for the run log.
    """
    allowed = {f.name for f in fields(cls)} if hasattr(cls, "__dataclass_fields__") else set()
    kept = {k: v for k, v in kwargs.items() if not allowed or k in allowed}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[train-config] {cls.__name__} ignores unsupported keys: {dropped}", flush=True)
    return kept
