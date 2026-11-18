"""The chunked cross-entropy must equal the full-logits loss it replaces."""
import math

import pytest
import torch
from torch.nn import functional as F

from thermo_agent.train.chunked_loss import add_chunked_lm_head, chunked_token_nll


def test_matches_full_cross_entropy():
    torch.manual_seed(0)
    hidden = torch.randn(37, 64)
    weight = torch.randn(211, 64)
    bias = torch.randn(211)
    targets = torch.randint(0, 211, (37,))
    reference = F.cross_entropy(hidden @ weight.T + bias, targets, reduction="none")
    for token_chunk, vocab_chunk in ((8, 16), (37, 211), (5, 64), (1, 7)):
        got = chunked_token_nll(hidden, weight, bias, targets, token_chunk, vocab_chunk)
        torch.testing.assert_close(got, reference, rtol=1e-4, atol=1e-5)


def test_gradients_match_full_cross_entropy():
    torch.manual_seed(1)
    hidden = torch.randn(11, 32, requires_grad=True)
    weight = torch.randn(53, 32, requires_grad=True)
    targets = torch.randint(0, 53, (11,))
    F.cross_entropy(hidden @ weight.T, targets, reduction="sum").backward()
    expected_hidden, expected_weight = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    chunked_token_nll(hidden, weight, None, targets, token_chunk=3, vocab_chunk=8).sum().backward()
    torch.testing.assert_close(hidden.grad, expected_hidden, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(weight.grad, expected_weight, rtol=1e-4, atol=1e-5)


def test_empty_input_is_a_zero_not_an_error():
    nll = chunked_token_nll(torch.zeros(0, 8), torch.zeros(16, 8), None, torch.zeros(0, dtype=torch.long))
    assert nll.shape == (0,)


def _tiny_model(vocab=97, hidden=32, layers=2):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"][:layers],
        full_attention_interval=2,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
    )
    return Qwen3_5ForCausalLM(config)


def test_patched_forward_matches_the_model_loss():
    torch.manual_seed(2)
    model = _tiny_model().eval()
    input_ids = torch.randint(3, 97, (2, 19))
    labels = input_ids.clone()
    labels[:, :7] = -100  # prompt tokens are not scored, as with completion_only_loss
    with torch.no_grad():
        reference = model(input_ids=input_ids, labels=labels).loss
        add_chunked_lm_head(model, token_chunk=4, vocab_chunk=16)
        output = model(input_ids=input_ids, labels=labels, chunked_loss=True)
    torch.testing.assert_close(output.loss, reference, rtol=1e-5, atol=1e-5)


def test_patched_forward_leaves_normal_calls_untouched():
    torch.manual_seed(3)
    model = _tiny_model().eval()
    input_ids = torch.randint(3, 97, (2, 11))
    with torch.no_grad():
        before = model(input_ids=input_ids).logits
        add_chunked_lm_head(model)
        after = model(input_ids=input_ids)
    assert before.shape == after.logits.shape
    torch.testing.assert_close(before, after.logits)


def test_gradients_flow_through_the_patched_forward():
    torch.manual_seed(4)
    model = _tiny_model()
    add_chunked_lm_head(model, token_chunk=5, vocab_chunk=8)
    input_ids = torch.randint(3, 97, (2, 13))
    model(input_ids=input_ids, labels=input_ids, chunked_loss=True).loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # Qwen3.5 ties the LM head to the embedding, so the embedding must receive gradient.
    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_input_embeddings().weight.grad.abs().sum() > 0


def _tiny_tokenizer(vocab_size=97):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(vocab_size)}
    vocab["[UNK]"] = vocab_size
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="t0", eos_token="t1")


def test_trainer_step_with_chunked_loss_is_finite():
    from datasets import Dataset
    from trl import SFTConfig

    from thermo_agent.train.trainers import build_sft_trainer

    torch.manual_seed(5)
    model = _tiny_model()
    dataset = Dataset.from_list(
        [{"input_ids": list(range(3, 27)), "labels": list(range(3, 27))} for _ in range(2)]
    )
    args = SFTConfig(
        output_dir="logs/_chunked_trainer_test",
        max_steps=1,
        max_length=24,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        completion_only_loss=False,
    )
    trainer = build_sft_trainer(model=model, args=args, train_dataset=dataset, processing_class=_tiny_tokenizer(),
                                loss_token_chunk=4, loss_vocab_chunk=16)
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    assert getattr(trainer.model, "_thermo_chunked_head", False) or getattr(
        trainer.model.get_base_model(), "_thermo_chunked_head", False
    )
