from thermo_agent.train.render import render_pair, prefix_report, generation_prompt, CHAT_KW

class FakeTok:
    """Mimics Qwen3 reasoning mode: the <think> opener stays at the end of the generation prompt."""
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, enable_thinking=True, **kw):
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt: s += "<|assistant|>" + ("" if enable_thinking else "<think></think>")
        return s

ROW = {"prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], "completion": [{"role": "assistant", "content": "ANSWER"}]}

def test_text_format_matches_the_generation_prompt_exactly():
    assert CHAT_KW["enable_thinking"] is True
    p = render_pair(FakeTok(), ROW); assert p["prompt"] == generation_prompt(FakeTok(), ROW["prompt"]) and p["prompt"].endswith("<|assistant|>")
    assert "<think></think>" not in p["prompt"]  # reasoning enabled: no empty closed think block
    assert p["completion"] == "ANSWER<|end|>"

def test_prefix_report_confirms_alignment_with_reasoning_enabled():
    assert prefix_report(FakeTok(), ROW)["prefix_matches"] is True
