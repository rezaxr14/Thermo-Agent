"""Gate 4: tool results come from the environment, never from the SFT target.

A trajectory is stored as one assistant message containing ``<tool_call>`` and
``<tool_result>`` blocks. ``completion_only_loss=True`` only masks the prompt, so without this
mask the student would learn to predict tool output. These tests use a char-per-token fake
tokenizer: no `transformers` download, no GPU, pure CPU.
"""
import torch
import torch.nn.functional as F

from thermo_agent.train.render import generation_prompt
from thermo_agent.train.sft import _tool_result_spans, mask_tool_result_spans, preprocess_row


_MASTER = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " <>/=[]{}\"'_|,.:+!?-\n\t"
)


class CharTokenizer:
    """Fake fast tokenizer: one token per character, offsets ``(i, i + 1)``.

    Because each token covers exactly one character, the number of masked tokens equals the
    number of characters inside the ``<tool_result>`` spans (tags included) — the property the
    tests below assert explicitly.
    """

    def __init__(self):
        self.alphabet = "".join(sorted(set(_MASTER)))
        self.vocab = {c: i for i, c in enumerate(self.alphabet)}
        self.eos_token = "<|im_end|>"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        ids = [self.vocab[c] for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text


def labels_for(tok, text, prompt_chars=0):
    """Offsets + labels with the first `prompt_chars` tokens masked (completion_only_loss)."""
    enc = tok(text, return_offsets_mapping=True)
    labels = [-100] * prompt_chars + enc["input_ids"][prompt_chars:]
    return enc["offset_mapping"], labels


def test_tool_result_span_masked_but_tool_call_and_answer_trainable():
    tok = CharTokenizer()
    prompt = "<|user|>solve<|assistant|>"
    completion = (
        "<think>reason</think>\n"
        "<tool_call name=\"first_law_check\">{\"x\":1}</tool_call>\n"
        "<tool_result>Solver OK 42</tool_result>\n"
        "Final answer: 42"
    )
    text = prompt + completion
    offsets, labels = labels_for(tok, text, len(prompt))
    out = mask_tool_result_spans(text, offsets, labels)

    spans = _tool_result_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    # Proof: char-per-token => masked completion tokens == characters inside the span.
    masked_in_completion = sum(1 for i in range(len(prompt), len(text)) if out[i] == -100)
    assert masked_in_completion == end - start == len("<tool_result>Solver OK 42</tool_result>") == 39

    for i, (char_start, char_end) in enumerate(offsets):
        if start <= char_start and char_end <= end:
            assert out[i] == -100, (i, text[char_start:char_end])
        else:
            assert out[i] == labels[i], (i, text[char_start:char_end])
    assert out[text.index("<tool_call")] != -100
    assert out[text.index("Final answer")] != -100


def test_native_tool_response_masked_like_tool_result():
    """Qwen's native chat template wraps tool output in <tool_response>; it is environment output too."""
    tok = CharTokenizer()
    prompt = "<|user|>solve<|assistant|>"
    completion = (
        "<tool_call><function=validity_check><parameter=x>1</parameter></function></tool_call>\n"
        "<tool_response>valid=false failed_checks=[solver:E_PROP_RANGE]</tool_response>\n"
        "Final answer: fixed"
    )
    text = prompt + completion
    offsets, labels = labels_for(tok, text, len(prompt))
    out = mask_tool_result_spans(text, offsets, labels)

    spans = _tool_result_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end].startswith("<tool_response>")
    assert text[start:end].endswith("</tool_response>")
    masked_in_completion = sum(1 for i in range(len(prompt), len(text)) if out[i] == -100)
    assert masked_in_completion == end - start
    assert out[text.index("<tool_call")] != -100
    assert out[text.index("Final answer")] != -100


def test_native_and_xml_results_in_one_trace_both_masked():
    tok = CharTokenizer()
    text = (
        "ask<tool_result>xml</tool_result>between"
        "<tool_response>native</tool_response>done"
    )
    offsets, labels = labels_for(tok, text)
    out = mask_tool_result_spans(text, offsets, labels)

    spans = _tool_result_spans(text)
    assert len(spans) == 2
    assert all(out[i] == -100 for i in range(spans[0][0], spans[0][1]))
    assert all(out[i] == -100 for i in range(spans[1][0], spans[1][1]))
    assert all(out[i] != -100 for i in range(spans[0][1], spans[1][0]))
    assert all(out[i] != -100 for i in range(spans[1][1], len(text)))


def test_two_spans_at_start_and_end_no_off_by_one():
    tok = CharTokenizer()
    text = "<tool_result>A</tool_result>middle<tool_result>B</tool_result>"
    offsets, labels = labels_for(tok, text)
    out = mask_tool_result_spans(text, offsets, labels)

    spans = _tool_result_spans(text)
    assert spans == [(0, 28), (34, 62)]
    masked = [i for i, label in enumerate(out) if label == -100]
    assert masked == list(range(0, 28)) + list(range(34, 62))  # exactly, no off-by-one
    assert all(out[i] != -100 for i in range(28, 34))  # 'middle' stays trainable
    assert len(masked) == (spans[0][1] - spans[0][0]) + (spans[1][1] - spans[1][0]) == 56


def test_unclosed_tool_result_masks_to_end_of_text():
    tok = CharTokenizer()
    text = "prefix<tool_result>unclosed tail"
    offsets, labels = labels_for(tok, text)
    out = mask_tool_result_spans(text, offsets, labels)

    cut = text.index("<tool_result>")
    assert _tool_result_spans(text) == [(cut, len(text))]
    assert all(out[i] == -100 for i in range(cut, len(text)))
    assert all(out[i] != -100 for i in range(cut))


def test_plain_row_keeps_every_non_prompt_label():
    tok = CharTokenizer()
    prompt = "<|user|>q<|assistant|>"
    text = prompt + "plain answer, no tool markup here"
    offsets, labels = labels_for(tok, text, len(prompt))
    out = mask_tool_result_spans(text, offsets, labels)

    assert out == labels
    assert all(out[i] == -100 for i in range(len(prompt)))
    assert all(out[i] != -100 for i in range(len(prompt), len(text)))


def test_loss_and_gradient_match_reference_that_drops_masked_positions():
    torch.manual_seed(0)
    tok = CharTokenizer()
    text = "A<tool_result>42</tool_result>B"
    offsets, labels = labels_for(tok, text)
    out = mask_tool_result_spans(text, offsets, labels)
    keep = torch.tensor([label != -100 for label in out])
    target = torch.tensor(out)

    logits = torch.randn(len(text), len(tok.alphabet), requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(logits, target, ignore_index=-100)
    loss_ref = F.cross_entropy(reference[keep], target[keep])

    assert abs(loss.item() - loss_ref.item()) < 1e-4
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(logits.grad, reference.grad, atol=1e-4)


def test_preprocess_row_emits_labels_with_prompt_and_tool_result_masked():
    """The wiring: preprocess_row produces trainer-ready input_ids + masked labels."""
    tok = CharTokenizer()
    row = {"prompt": [{"role": "user", "content": "q"}],
           "completion": [{"content": "<tool_result>x</tool_result>answer"}]}
    out = preprocess_row(tok, row)

    text = "".join(tok.alphabet[i] for i in out["input_ids"])
    labels = out["labels"]
    assert len(labels) == len(out["input_ids"])
    prompt_len = len(generation_prompt(tok, row["prompt"]))  # char-per-token
    span_start = text.index("<tool_result>")
    span_end = text.index("</tool_result>") + len("</tool_result>")
    assert all(label == -100 for label in labels[:prompt_len])  # completion_only_loss
    assert all(label == -100 for label in labels[span_start:span_end])  # environment output
    assert labels[span_end] != -100  # the assistant's own answer is still trained
