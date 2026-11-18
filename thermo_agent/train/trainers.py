"""Trainer subclasses that keep the LM head off the activation budget.

The stock TRL trainer asks the model for the loss, which builds the whole
``(batch x sequence, vocabulary)`` logits tensor. On a T4 with Qwen3.5-4B (248,320-token
vocabulary) that tensor alone is 4 GiB in fp16 and 8 GiB once the fp32 cross-entropy
upcast is counted, on top of 8 GiB of weights. `ChunkedSFTTrainer` instead asks the
patched forward for the same loss computed on `token_chunk x vocab_chunk` tiles.

Only the loss computation changes: weights, labels, masking, optimizer, scheduler,
checkpointing and DDP wrapping are the ones TRL already builds.
"""
from __future__ import annotations

import torch
from peft import PeftModel

from .chunked_loss import add_chunked_lm_head

# Arguments a causal-LM backbone accepts; anything else the collator adds stays out.
_BACKBONE_KEYS = ("input_ids", "attention_mask", "position_ids", "token_type_ids", "inputs_embeds")


def _base_model(model):
    """The plain causal LM, whether or not PEFT or DDP wraps it."""
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    elif hasattr(model, "module"):  # torch.nn.parallel.DistributedDataParallel
        model = _base_model(model.module)
    return model


class ChunkedSFTTrainerMixin:
    loss_token_chunk = 512
    loss_vocab_chunk = 8192

    def _install_chunked_loss(self) -> None:
        base = _base_model(self.model)
        add_chunked_lm_head(base, self.loss_token_chunk, self.loss_vocab_chunk)
        print(
            f"[loss] chunked cross-entropy: {self.loss_token_chunk} tokens x "
            f"{self.loss_vocab_chunk} vocabulary entries per tile (no full logits)",
            flush=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None:  # nothing to score, e.g. a generation-only batch
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        kwargs = {k: inputs[k] for k in _BACKBONE_KEYS if k in inputs}
        outputs = model(**kwargs, labels=labels, chunked_loss=True, num_items_in_batch=num_items_in_batch)
        loss = outputs.loss
        # `num_items_in_batch` counts tokens on every rank, while DDP averages this rank's
        # loss by the world size, so scale up to recover the global token mean.
        if (
            num_items_in_batch is not None
            and getattr(self.args, "average_tokens_across_devices", False)
            and self.accelerator.num_processes > 1
        ):
            loss = loss * self.accelerator.num_processes
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluation only needs the loss; skip the full-logits pass TRL would add."""
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager(), torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach() if isinstance(loss, torch.Tensor) else loss), None, None


def build_sft_trainer(*args, loss_token_chunk=512, loss_vocab_chunk=8192, **kwargs):
    """Return an SFTTrainer whose loss is computed in bounded tiles."""
    from trl import SFTTrainer

    class ChunkedSFTTrainer(ChunkedSFTTrainerMixin, SFTTrainer):
        def __init__(self, *a, **kw):
            self.loss_token_chunk = loss_token_chunk
            self.loss_vocab_chunk = loss_vocab_chunk
            super().__init__(*a, **kw)
            self._install_chunked_loss()

    return ChunkedSFTTrainer(*args, **kwargs)
