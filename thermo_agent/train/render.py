"""Render training pairs exactly as GRPO and inference will see the prompt.

Measured with the cached Qwen3.5-4B tokenizer: with ``enable_thinking=True``
``apply_chat_template(sys+user, add_generation_prompt=True)`` is a LITERAL PREFIX of
``apply_chat_template(sys+user+assistant_containing_<think>)``. With ``False`` the template
injects an empty closed ``<think>\n\n</think>\n\n`` into the generation prompt, which is NOT
a prefix. Our stored trajectories keep ``<think>...</think>`` inside the assistant content,
so ``completion_only_loss`` masking aligns only when reasoning is enabled. TRAIN_ON_REASONING
is exported so SFT and GRPO share one value.
"""
from __future__ import annotations

TRAIN_ON_REASONING = True
CHAT_KW = {"enable_thinking": TRAIN_ON_REASONING}


def generation_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **CHAT_KW)


def render_pair(tok, row: dict) -> dict:
    return {"prompt": generation_prompt(tok, row["prompt"]), "completion": row["completion"][0]["content"]}


def prefix_report(tok, row: dict) -> dict:
    """Does the whole-conversation render start with the generation prompt? False means conversational SFT would mismatch GRPO."""
    full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False, **CHAT_KW)
    gp = generation_prompt(tok, row["prompt"])
    return {"prefix_matches": full.startswith(gp), "generation_prompt_tail": gp[-80:], "full_render_at_that_point": full[len(gp) - 80: len(gp) + 40] if len(full) >= len(gp) else full[-120:]}
