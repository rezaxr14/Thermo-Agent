"""Training settings. Values come from: Schulman et al. 2025 (LoRA on all linear layers, LR ~10x full fine-tuning,
low rank is enough for RL), Luo et al. 2026 (SFT then GRPO with a group of 4), and TRL's GRPOConfig defaults.

Dual-T4 target (see docs/DUAL_T4.md): full fp16 LoRA, two DDP ranks, no 4-bit quantisation. The T4 is Turing,
so fp16 is the only half-precision format it has fast kernels for; bf16 must not be selected there.
"""
from dataclasses import dataclass, field, fields

BASE_MODEL = "Qwen/Qwen3.5-4B"


@dataclass
class LoraSettings:
    r: int = 32                        # B2 (SFT). GRPO overrides to r=8/alpha=16 (B7).
    alpha: int = 64                    # B3 = 2 x r
    dropout: float = 0.05
    target_modules: str = "all-linear"


@dataclass
class SftSettings:
    lr: float = 1e-4                   # A3
    epochs: int = 3                    # A1: upper bound; early stopping (A10) decides the real length
    lr_scheduler: str = "cosine"       # A10
    warmup_ratio: float = 0.05         # A10
    weight_decay: float = 0.0          # A10
    early_stopping_patience: int = 3   # A10: evaluations without a better val loss (3 x 50 steps)
    max_length: int = 8192             # prompt + completion, both gated at 8000 tokens by synth.combine
    per_device_batch: int = 2
    grad_accum: int = 8
    max_grad_norm: float = 1.0         # fp16 needs clipping before the optimizer step
    # An epoch of the 20-variant corpus is ~30 h on two T4s, far longer than one 12 h Kaggle session. Save and
    # validate every few hundred steps so a dropped session resumes from its last checkpoint instead of restarting.
    save_steps: int = 50               # A7 (must equal eval_steps for load_best_model_at_end)
    eval_steps: int = 50
    lora: LoraSettings = field(default_factory=LoraSettings)


@dataclass
class GrpoSettings:
    lr: float = 2e-5                   # C2
    num_generations: int = 4           # C4: G = k, overridden per run with --num-generations (k = 4, 8, 16)
    max_completion_length: int = 4096  # C5: bounds the whole multi-turn completion (all turns + tool results)
    temperature: float = 1.0           # C6: exact on-policy sampling, no truncation of the distribution
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    per_device_batch: int = 2          # C4
    grad_accum: int = 4                # derived per run: prompts_per_step * k / (per_device_batch * world) (C4a)
    prompts_per_step: int = 4          # C4a: constant across k so the sweep isolates group size
    max_steps: int = 100               # C3: replaced by S, the largest value that lets k=16 finish in one session
    lr_scheduler: str = "constant_with_warmup"
    warmup_steps: int = 10
    save_steps: int = 25
    scale_rewards: str = "group"           # normalise advantages inside each group, not across prompts
    loss_type: str = "dapo"            # TRL default; 'grpo' reproduces the original length normalisation
    beta: float = 0.0                  # no KL term -> no reference model in memory
    mask_truncated: bool = True
    max_grad_norm: float = 1.0
    lora: LoraSettings = field(default_factory=lambda: LoraSettings(r=8, alpha=16))


def activation_checkpointing_kwargs() -> dict:
    """Memory settings that let an 8,192-token row backpropagate on a 14.56 GiB T4.

    Smoke v28 (2026-11-19) died in ``accelerator.backward`` with 13.55 GiB in use
    and a 1.10 GiB allocation failing at 1.01 GiB free. The saved activations for
    ``layers x sequence x hidden`` dominate at that length; ``offload=True`` (new
    name in transformers 5.23, reached through the deprecated alias we already
    pass) keeps the tensors for recomputation in pinned host memory. The old
    ``use_reentrant=False`` choice is unchanged. No weights, labels or gate change.
    """
    return {"use_reentrant": False, "offload": True}


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
