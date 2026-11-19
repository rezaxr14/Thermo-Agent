"""Gate 7: the H arm's tool-using multi-turn generator (no network/GPU; fake ``generate``).

Everything here drives :class:`thermo_agent.eval.model_eval.ToolLoop` with a scripted
callable, so the physics tools and the real verifier run on tiny committed fixture rows.
"""
from __future__ import annotations

import json

import pytest

from thermo_agent.eval import model_eval as ME
from thermo_agent.teacher import toolkit
from thermo_agent.teacher.validate import validate


CALL_INSPECT = '<tool_call name="inspect_plant">{"candidate": ""}</tool_call>'
CALL_VALIDITY = '<tool_call name="validity_check">{"candidate": "not an answer"}</tool_call>'


class ScriptedGen:
    """Fake ``generate(messages) -> str`` with optional token/truncation bookkeeping."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_truncated = False
        self.truncate_on: set[int] = set()
        self.current_id = None

    def set_current_id(self, rid):
        self.current_id = rid

    def __call__(self, messages):
        self.prompts.append(messages)
        i = len(self.prompts) - 1
        self.last_truncated = i in self.truncate_on
        self.last_prompt_tokens = 10
        self.last_completion_tokens = 5
        return self.replies[min(i, len(self.replies) - 1)]


@pytest.fixture
def repair_row(corpus_fixture_rows):
    return corpus_fixture_rows["repair"][0]


def _truth(row):
    return row["completion"][0]["content"]


def _data_dir(tmp_path, row, task, split="test_id"):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    (d / f"{task}_{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# 1. one tool call, then the final XML answer
# ---------------------------------------------------------------------------
def test_scripted_single_tool_call_then_final(repair_row):
    truth = _truth(repair_row)
    gen = ScriptedGen([CALL_INSPECT, truth])
    loop = ME.ToolLoop(gen, repair_row, "repair", max_turns=4, max_tool_calls=6)
    final = loop.run()
    rec = loop.last_record

    assert len(gen.prompts) == 2, "loop stops at the first final answer"
    assert rec["turns"] == 2
    assert rec["tool_calls"] == ["inspect_plant"]
    assert rec["tool_results_chars"] == [len(rec["turn_log"][0]["tool_results"][0])]
    assert rec["final_text"] == truth
    assert rec["stop_reason"] == "final" and rec["limit_reached"] is False
    assert rec["truncated"] is False

    # the verbatim result is in the NEXT prompt, wrapped in <tool_result>
    result = rec["turn_log"][0]["tool_results"][0]
    next_prompt = gen.prompts[1][-1]["content"]
    assert "<tool_result>" in next_prompt and result in next_prompt

    # token counts accumulate across turns (fake reports 10 / 5 per call)
    assert rec["prompt_tokens"] == 20 and rec["completion_tokens"] == 10

    # no ground truth is revealed to the model
    assert truth not in gen.prompts[0][0]["content"]
    assert "check_answer" not in gen.prompts[0][0]["content"]

    # the final answer is graded by the real verifier
    assert validate("repair", final, repair_row)["accepted"] is True


# ---------------------------------------------------------------------------
# 2. a model that keeps calling tools hits a limit (no infinite loop)
# ---------------------------------------------------------------------------
def test_tool_call_cap_stops_without_infinite_loop(repair_row):
    gen = ScriptedGen([CALL_VALIDITY])
    loop = ME.ToolLoop(gen, repair_row, "repair", max_turns=4, max_tool_calls=2)
    loop.run()
    rec = loop.last_record
    assert rec["stop_reason"] == "max_tool_calls"
    assert rec["limit_reached"] is True
    assert rec["tool_calls"] == ["validity_check", "validity_check"]
    assert len(rec["tool_results_chars"]) == 2
    assert len(gen.prompts) <= 4


def test_turn_cap_stops_tool_calling_episode(repair_row):
    gen = ScriptedGen([CALL_INSPECT])
    loop = ME.ToolLoop(gen, repair_row, "repair", max_turns=3, max_tool_calls=6)
    loop.run()
    rec = loop.last_record
    assert rec["stop_reason"] == "max_turns"
    assert rec["limit_reached"] is True
    assert rec["turns"] == 3
    assert rec["tool_calls"] == ["inspect_plant"] * 3
    assert len(gen.prompts) == 3


def test_token_budget_hit_mid_turn_sets_truncated(repair_row):
    gen = ScriptedGen([CALL_INSPECT, _truth(repair_row)])
    gen.truncate_on = {0}
    loop = ME.ToolLoop(gen, repair_row, "repair", max_turns=4, max_tool_calls=6)
    loop.run()
    assert loop.last_record["truncated"] is True


# ---------------------------------------------------------------------------
# 3. prose with no XML and no tool call: stops at the turn budget, rejected
# ---------------------------------------------------------------------------
def test_prose_without_xml_stops_at_budget_and_is_rejected(repair_row, tmp_path):
    gen = ScriptedGen(["I am not sure."])
    loop = ME.ToolLoop(gen, repair_row, "repair", max_turns=3, max_tool_calls=6)
    final = loop.run()
    rec = loop.last_record
    assert len(gen.prompts) == 3
    assert rec["turns"] == 3
    assert rec["tool_calls"] == []
    assert rec["stop_reason"] == "max_turns" and rec["limit_reached"] is True
    assert validate("repair", final, repair_row)["accepted"] is False

    data = _data_dir(tmp_path, repair_row, "repair")
    gen2 = ScriptedGen(["I am not sure."])
    rep, _ = ME.run(gen2, data, ["repair"], splits=["test_id"], limit=1,
                    tool_access="tools", max_turns=3)
    assert rep["repair"]["test_id"]["success"] == 0.0


# ---------------------------------------------------------------------------
# 4. --tool-access none: one generation call, no tool text anywhere
# ---------------------------------------------------------------------------
def test_no_tool_access_is_single_turn_and_shows_no_tool_text(repair_row, tmp_path):
    truth = _truth(repair_row)
    gen = ScriptedGen([truth])
    data = _data_dir(tmp_path, repair_row, "repair")
    rep, _ = ME.run(gen, data, ["repair"], splits=["test_id"], limit=1, tool_access="none")

    assert len(gen.prompts) == 1, "prompt-text ablation keeps the single repair_loop call"
    blob = json.dumps(gen.prompts)
    for token in ("<tool_call", "<tool_result", "TOOL ENVIRONMENT", "MULTI-TURN TOOL",
                  "inspect_plant", "validity_check"):
        assert token not in blob
    assert rep["repair"]["test_id"]["success"] == 1.0
    assert rep["harness_config"]["tool_access"] == "none"
    assert rep["harness_config"]["max_new_tokens"] == 400
    assert rep["harness_config"]["max_turns"] == 3


def test_harness_prompt_text_mode_unchanged(repair_row, tmp_path):
    """The ablation still runs the task feedback loop, not the tool loop."""
    gen = ScriptedGen(["garbage"])
    data = _data_dir(tmp_path, repair_row, "repair")
    ME.run(gen, data, ["repair"], splits=["test_id"], limit=1,
           tool_access="none", max_turns=2)
    assert len(gen.prompts) == 2, "repair_loop drives the turns when tools are off"


# ---------------------------------------------------------------------------
# prompt construction + record attachment
# ---------------------------------------------------------------------------
def test_tool_contract_lists_all_five_tools_without_oracle(repair_row):
    gen = ScriptedGen([_truth(repair_row)])
    ME.ToolLoop(gen, repair_row, "repair").run()
    system = gen.prompts[0][0]["content"]
    for name in toolkit.tool_names():
        assert name in system
    assert "<tool_call" in system and "<tool_result>" in system
    assert "answer-oracle" in system and "check_answer" not in system


def test_episode_record_attached_to_log_and_examples(repair_row, tmp_path):
    gen = ScriptedGen([CALL_INSPECT, _truth(repair_row)])
    data = _data_dir(tmp_path, repair_row, "repair")
    rep, log = ME.run(gen, data, ["repair"], splits=["test_id"], limit=1,
                      tool_access="tools", max_turns=4, max_tool_calls=6)
    metrics = rep["repair"]["test_id"]
    assert metrics["success"] == 1.0
    ex = metrics["examples"][0]
    assert ex["tool_episode"]["turns"] == 2
    assert ex["tool_episode"]["tool_calls"] == ["inspect_plant"]
    assert metrics["tool_summary"]["mean_turns"] == 2
    assert metrics["tool_summary"]["mean_tool_calls"] == 1
    assert any(e.get("task") == "repair" and "tool_episode" in e for e in log)


# ---------------------------------------------------------------------------
# 5. harness_config in the report equals the CLI values / defaults
# ---------------------------------------------------------------------------
def _run_cli(monkeypatch, tmp_path, row, extra_args):
    data = _data_dir(tmp_path, row, "repair")
    truth = _truth(row)
    made = []

    class FakeHF:
        def __init__(self, model, adapter, dtype, load_4bit, max_new_tokens, enable_thinking=True):
            self.max_new_tokens = max_new_tokens
            self.enable_thinking = enable_thinking
            made.append(self)

        def __call__(self, messages):
            return truth

    monkeypatch.setattr(ME, "HFGenerator", FakeHF)
    out = tmp_path / f"report{len(extra_args)}.json"
    argv = ["model_eval", str(data), "--tasks", "repair", "--limit", "1", "--out", str(out)] + extra_args
    monkeypatch.setattr("sys.argv", argv)
    ME.main()
    return json.loads(out.read_text(encoding="utf-8")), made


def test_report_harness_config_matches_cli_values(repair_row, tmp_path, monkeypatch, capsys):
    rep, made = _run_cli(monkeypatch, tmp_path, repair_row, [
        "--harness", "agents+skills", "--tool-access", "tools",
        "--max-turns", "5", "--max-tool-calls", "8", "--max-new-tokens", "1024",
    ])
    capsys.readouterr()
    expected = {"mode": "agents+skills", "tool_access": "tools", "max_turns": 5,
                "max_tool_calls": 8, "max_new_tokens": 1024}
    assert rep["harness_config"] == expected
    assert rep["meta"]["harness_config"] == expected
    assert made[0].max_new_tokens == 1024


def test_report_harness_config_tools_defaults(repair_row, tmp_path, monkeypatch, capsys):
    rep, made = _run_cli(monkeypatch, tmp_path, repair_row, ["--tool-access", "tools"])
    capsys.readouterr()
    assert rep["harness_config"] == {"mode": "none", "tool_access": "tools", "max_turns": 4,
                                     "max_tool_calls": 6, "max_new_tokens": 2048}
    assert made[0].max_new_tokens == 2048


def test_report_harness_config_prompt_text_defaults(repair_row, tmp_path, monkeypatch, capsys):
    rep, made = _run_cli(monkeypatch, tmp_path, repair_row, [])
    capsys.readouterr()
    assert rep["harness_config"] == {"mode": "none", "tool_access": "none", "max_turns": 3,
                                     "max_tool_calls": 6, "max_new_tokens": 400}
    assert made[0].max_new_tokens == 400
