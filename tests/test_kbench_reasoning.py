"""gpt-6 chat-completions rejects tools + reasoning_effort; the adapter must send "none"."""
from thermo_agent.teacher.kbench_adapter import reasoning_for_model


def test_openai_gpt6_needs_reasoning_none():
    # Exact 400 from the hosted run: "Function tools with reasoning_effort are not supported
    # for gpt-6-luna ... set reasoning_effort to 'none'".
    assert reasoning_for_model("openai/gpt-6-luna") == "none"
    assert reasoning_for_model("openai/gpt-6.2-sol") == "none"
    assert reasoning_for_model("gpt-6-luna") == "none"


def test_other_models_keep_backend_default():
    assert reasoning_for_model("anthropic/claude-haiku-5-5") is None
    assert reasoning_for_model("google/gemini-3.8-flash") is None
    assert reasoning_for_model("") is None
    assert reasoning_for_model("unknown") is None


def test_make_respond_only_forwards_reasoning_when_set(monkeypatch):
    """reasoning=None must not add the kwarg (models that support tools+reasoning keep traces)."""
    from thermo_agent.teacher import kbench_adapter

    captured = {}

    def fake_native(llm, tools, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(kbench_adapter, "native_tool_agent", fake_native)

    class _Chat:
        messages = []
        usage = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Chats:
        def new(self, *a, **k):
            return _Chat()

    class _Kbench:
        chats = _Chats()

    class _Actors:
        user = type("U", (), {"send": staticmethod(lambda *a, **k: None)})()

        @staticmethod
        def Actor(*a, **k):
            return type("A", (), {"send": staticmethod(lambda *a, **k: None)})()

    for level, expected in ((None, False), ("none", True)):
        captured.clear()
        respond = kbench_adapter.make_respond(
            object(), _Kbench(), _Actors(), system_prompt="sys", reasoning=level)
        try:
            respond([{"role": "system", "content": "sys"},
                     {"role": "user", "content": "task"}], [])
        except RuntimeError:
            pass
        assert ("reasoning" in captured) is expected
        if expected:
            assert captured["reasoning"] == "none"
