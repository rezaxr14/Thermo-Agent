"""Memory-bounded cross-entropy for a 4B model on a 16 GB T4.

The LM head projects every scored token onto the full vocabulary. Qwen3.5-4B has 248,320
tokens, so a batch of 2 x 4096 tokens produces 4 GiB of fp16 logits, and ``CrossEntropyLoss``
upcasts another 4 GiB to fp32 for the log-softmax. That does not fit next to the 8 GiB of
fp16 weights: the first dual-T4 attempt died with 13.61 GiB allocated and 126 MiB short.

The same negative log-likelihood is computed here tile by tile: a tile of tokens is
projected through a tile of the vocabulary, folded into an online log-sum-exp in fp32, and
the tile logits are dropped. The backward pass recomputes each tile and turns it into the
logit gradient in place, so the peak is one ``token_chunk x vocab_chunk`` tile instead of
the whole ``(batch x seq, vocab)`` tensor. Defaults (512 tokens x 8192 vocabulary entries)
hold 16 MiB of fp32 logits per tile on the T4.

No architecture change is involved: the same weights, the same labels, the same objective.
"""
from __future__ import annotations

import functools
import types
from dataclasses import dataclass

import torch


@dataclass
class ChunkedLossOutput:
    """What the patched forward returns instead of full logits."""

    loss: torch.Tensor
    nll: torch.Tensor
    count: int


def _tile_logits(hidden: torch.Tensor, weight: torch.Tensor, bias, start: int, end: int) -> torch.Tensor:
    """``hidden @ weight[start:end].T (+ bias)`` in fp32, with the matmul in the compute dtype."""
    tile = torch.mm(hidden, weight[start:end].t()).float()
    if bias is not None:
        tile = tile + bias[start:end].float()
    return tile


class _ChunkedNLL(torch.autograd.Function):
    """Per-token NLL of ``hidden @ weight.T`` without materializing the logits."""

    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, token_chunk, vocab_chunk):
        vocab = weight.shape[0]
        nll = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
        log_norm = torch.empty_like(nll)
        for start in range(0, hidden.shape[0], token_chunk):
            stop = min(start + token_chunk, hidden.shape[0])
            h = hidden[start:stop]
            t = targets[start:stop]
            running_max = torch.full((h.shape[0],), float("-inf"), device=h.device, dtype=torch.float32)
            running_sum = torch.zeros_like(running_max)
            target_logit = torch.zeros_like(running_max)
            for v0 in range(0, vocab, vocab_chunk):
                v1 = min(v0 + vocab_chunk, vocab)
                tile = _tile_logits(h, weight, bias, v0, v1)
                in_tile = (t >= v0) & (t < v1)
                if in_tile.any():
                    index = (t - v0).clamp(0, v1 - v0 - 1)
                    gathered = tile.gather(1, index[:, None]).squeeze(1)
                    target_logit = torch.where(in_tile, gathered, target_logit)
                tile_max = tile.max(dim=1).values
                new_max = torch.maximum(running_max, tile_max)
                # online log-sum-exp: rescale the running sum to the new maximum
                running_sum = running_sum * torch.exp(running_max - new_max)
                running_sum = running_sum + torch.exp(tile - new_max[:, None]).sum(dim=1)
                running_max = new_max
            log_norm[start:stop] = running_max + running_sum.log()
            nll[start:stop] = log_norm[start:stop] - target_logit
        ctx.save_for_backward(hidden, weight, bias, targets, log_norm)
        ctx.token_chunk = token_chunk
        ctx.vocab_chunk = vocab_chunk
        return nll

    @staticmethod
    def backward(ctx, grad_nll):
        hidden, weight, bias, targets, log_norm = ctx.saved_tensors
        vocab = weight.shape[0]
        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None and bias.requires_grad else None
        for start in range(0, hidden.shape[0], ctx.token_chunk):
            stop = min(start + ctx.token_chunk, hidden.shape[0])
            h = hidden[start:stop]
            t = targets[start:stop]
            g = grad_nll[start:stop]
            d_hidden = torch.zeros((stop - start, hidden.shape[1]), device=hidden.device, dtype=torch.float32)
            norm = log_norm[start:stop]
            for v0 in range(0, vocab, ctx.vocab_chunk):
                v1 = min(v0 + ctx.vocab_chunk, vocab)
                tile = _tile_logits(h, weight, bias, v0, v1)
                # Normalize over the full vocabulary: `norm` was folded over every vocab tile.
                probability = torch.exp(tile - norm[:, None])
                in_tile = (t >= v0) & (t < v1)
                if in_tile.any():
                    index = (t[in_tile] - v0).clamp(0, v1 - v0 - 1)
                    probability[in_tile.nonzero(as_tuple=True)[0], index] -= 1
                probability.mul_(g[:, None])
                d_hidden += torch.mm(probability.to(hidden.dtype), weight[v0:v1]).float()
                if grad_weight is not None:
                    grad_weight[v0:v1] += torch.mm(probability.t().to(hidden.dtype), h).float()
                if grad_bias is not None:
                    grad_bias[v0:v1] += probability.sum(dim=0)
            grad_hidden[start:stop] = d_hidden
        return (
            grad_hidden.to(hidden.dtype),
            None if grad_weight is None else grad_weight.to(weight.dtype),
            None if grad_bias is None else grad_bias.to(bias.dtype),
            None,
            None,
            None,
        )


def chunked_token_nll(hidden, weight, bias, targets, token_chunk=512, vocab_chunk=8192) -> torch.Tensor:
    """Negative log-likelihood of every row of ``hidden`` under ``softmax(hidden @ weight.T)``.

    ``hidden`` is ``(N, H)``, ``weight`` is ``(V, H)``, ``targets`` is ``(N,)`` and every row is
    scored, so callers drop ignored labels first. The result is fp32 and differentiable in
    ``hidden`` (and in ``weight``/``bias`` when those require gradients).
    """
    if hidden.numel() == 0:
        return torch.zeros(0, device=hidden.device, dtype=torch.float32)
    if token_chunk < 1 or vocab_chunk < 1:
        raise ValueError("token_chunk and vocab_chunk must be >= 1")
    return _ChunkedNLL.apply(hidden, weight, bias, targets, int(token_chunk), int(vocab_chunk))


def add_chunked_lm_head(model, token_chunk: int = 512, vocab_chunk: int = 8192):
    """Patch a causal LM so ``model(..., chunked_loss=True)`` returns a bounded-memory loss.

    The patch keeps the original forward untouched for every other call (generation, eval
    logits, TRL's own bookkeeping), so this cannot change what the model computes outside
    training. Adding it on the unwrapped base model means PEFT adapters are still applied:
    the patched forward calls the decoder, whose linear layers the adapters wrap.
    """
    if getattr(model, "_thermo_chunked_head", False):
        return model
    original_forward = model.forward

    @functools.wraps(original_forward)
    def forward(
        self,
        *args,
        chunked_loss: bool = False,
        labels: torch.Tensor | None = None,
        num_items_in_batch=None,
        **kwargs,
    ):
        if not chunked_loss:
            if labels is not None:
                kwargs["labels"] = labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return original_forward(*args, **kwargs)

        kwargs.pop("labels", None)
        kwargs.pop("num_items_in_batch", None)
        kwargs["use_cache"] = False
        decoder = self.get_decoder()
        if decoder is self:
            raise RuntimeError(
                f"{type(self).__name__}.get_decoder() returned the causal LM itself; the chunked loss "
                "needs the transformer backbone so the LM head is not applied twice"
            )
        outputs = decoder(*args, **kwargs)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if labels is None:
            raise ValueError("chunked_loss=True needs labels")
        # Next-token shift, as in the model's own causal-LM loss.
        hidden = hidden[:, :-1]
        labels = labels[:, 1:]
        mask = labels.reshape(-1) != -100
        valid_hidden = hidden.reshape(-1, hidden.shape[-1])[mask]
        valid_labels = labels.reshape(-1)[mask]
        head = self.get_output_embeddings()
        nll = chunked_token_nll(
            valid_hidden, head.weight, getattr(head, "bias", None), valid_labels, token_chunk, vocab_chunk
        )
        count = int(valid_labels.numel())
        denominator = num_items_in_batch if num_items_in_batch is not None else max(count, 1)
        if isinstance(denominator, torch.Tensor):
            denominator = denominator.clamp(min=1)
        else:
            denominator = max(int(denominator), 1)
        loss = nll.sum() / denominator
        if count == 0:
            # Keep the graph alive so an all-masked micro-batch contributes a differentiable zero.
            loss = loss + valid_hidden.sum() * 0.0
        return ChunkedLossOutput(loss=loss, nll=nll, count=count)

    model.forward = types.MethodType(forward, model)
    model._thermo_chunked_head = True
    return model


def has_chunked_head(model) -> bool:
    return bool(getattr(model, "_thermo_chunked_head", False))
