from thermo_agent.train.render import render_pair, prefix_report, generation_prompt

class FakeTok:
    """Mimics a template that adds an empty thinking block to the generation prompt but not to history (as Qwen3 templates do)."""
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, enable_thinking=True, **kw):
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt: s += "<|assistant|>" + ("" if enable_thinking else "<think></think>")
        return s

ROW = {"prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], "completion": [{"role": "assistant", "content": "ANSWER"}]}

def test_text_format_matches_the_generation_prompt_exactly():
    p = render_pair(FakeTok(), ROW); assert p["prompt"] == generation_prompt(FakeTok(), ROW["prompt"]) and p["prompt"].endswith("<think></think>")
    assert p["completion"] == "ANSWER"

def test_prefix_report_flags_the_mismatch_that_conversational_sft_would_cause():
    assert prefix_report(FakeTok(), ROW)["prefix_matches"] is False
