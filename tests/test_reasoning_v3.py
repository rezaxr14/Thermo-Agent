"""2026-11-20: chain of thought is mandatory. Deterministic traces reason from real tool numbers, the AI-trace
ingest rejects tool traces without reasoning, and the tools/skills are on by default. Runs on tracked heldout_v2 rows."""
import json
import pathlib
import re

import pytest

from thermo_agent.synth import tool_traces as TT
from thermo_agent.teacher import toolkit, transfer, validate
from thermo_agent.agent import tool_prompt as TP

H2 = pathlib.Path(__file__).resolve().parent.parent / "data" / "heldout_v2"
TASKS = ("repair", "audit", "ledger", "design")


def _row(task):
    return json.loads((H2 / f"{task}_test_id.jsonl").read_text(encoding="utf-8").splitlines()[0])


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    monkeypatch.setattr(TT, "_ntok", lambda text: len(text.split()))


def _anchors(text):
    """Numbers and identifiers (E_FLUID_MISMATCH, chw_ret, E[s8], CC) that a reasoning sentence can cite."""
    return set(re.findall(r"-?\d+\.?\d*", text)) | set(re.findall(r"[A-Za-z]+_[A-Za-z_0-9]+|[A-Z]{2,}[A-Za-z0-9_]*|\w+\[\w+\]", text))


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("variant", ["direct", "recover"])
def test_deterministic_reasoning_is_grounded_in_tool_numbers(task, variant):
    tr = TT.build_trace(_row(task), task, variant=variant, strict=False)
    assert tr["accepted"] and tr["tool_calls"] >= 2
    prev_result = ""
    gt = TT.ground_truth(_row(task))
    for turn in tr["turns"]:
        assert len(turn["reasoning"]) >= validate.MIN_COT_CHARS                     # every turn reasons
        assert "Previous tool result" not in turn["reasoning"]                      # the v2 boilerplate is gone
        if prev_result:                                                             # later turns cite real numbers
            allowed = _anchors(prev_result) | _anchors(gt)
            assert _anchors(turn["reasoning"]) & allowed, turn["reasoning"]
        if turn["tool_results"]:
            prev_result = turn["tool_results"][0]
    assert validate.reasoning_report({"turns": [t for t in tr["turns"] if t["tool_calls"]]})["ok"]


def test_reasoning_is_not_boilerplate_across_rows():
    """v2: 12.7% of think blocks were distinct and 8 sentences made up 45%. v3: only the one-sentence opener of
    repair/audit/ledger traces may repeat; every other block is bound to that row's tool numbers."""
    allb, bound = [], []
    for task in TASKS:
        rows = [json.loads(l) for l in (H2 / f"{task}_test_id.jsonl").read_text(encoding="utf-8").splitlines()[:12]]
        for row in rows:
            tr = TT.build_trace(row, task, variant="direct", strict=False)
            texts = [t["reasoning"] for t in tr["turns"]]
            allb += texts
            bound += texts[1:] if task != "design" else texts
    assert len(set(allb)) / len(allb) >= 0.8
    assert len(set(bound)) / len(bound) >= 0.95


def test_reasoning_gate_rejects_silent_tool_calls_and_accepts_visible_prose():
    silent = {"turns": [{"assistant": "", "reasoning": "", "tool_calls": [{"name": "inspect_plant"}]} for _ in range(3)]}
    assert not validate.reasoning_report(silent)["ok"]                     # the 237 archived traces looked like this
    prose = {"turns": [{"assistant": f"I check step {i}: the residual 0.{i}5 MW looks too large, so I inspect it.",
                        "reasoning": "", "tool_calls": [{"name": "inspect_plant"}]} for i in range(3)]}
    assert validate.reasoning_report(prose)["ok"]                          # luna-style: visible prose IS the reasoning
    same = {"turns": [{"assistant": "Let me check the plant now please.", "reasoning": "",
                       "tool_calls": [{"name": "inspect_plant"}]} for _ in range(4)]}
    assert not validate.reasoning_report(same)["ok"]                       # one repeated sentence is not a chain of thought
    first_silent = {"turns": [{"assistant": "", "reasoning": "", "tool_calls": [{"name": "inspect_plant"}]}]
                    + prose["turns"]}
    assert not validate.reasoning_report(first_silent)["ok"]


def _ingest(tmp_path, trajectory):
    row = _row("repair")
    gt = TT.ground_truth(row)
    comp = tmp_path / "completions.jsonl"
    rec = {"id": row["id"], "text": gt, "trajectory": trajectory, "sft_completion": "x\n" + gt}
    comp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return transfer.ingest(str(H2), str(comp), str(tmp_path / "out"), None, ["repair"], "test_id",
                           model="unit", run_id="r1")


def test_ingest_rejects_tool_trace_without_reasoning(tmp_path):
    silent = {"turns": [{"assistant": "", "reasoning": "", "tool_calls": [{"name": "inspect_plant"}], "tool_results": ["{}"]}],
              "final": "x"}
    stats = _ingest(tmp_path, silent)
    assert stats["accepted"] == 0 and stats["stats"].get("reason:no_reasoning") == 1


def test_ingest_accepts_tool_trace_with_reasoning(tmp_path):
    good = {"turns": [{"assistant": "The solver status shows a fluid mismatch on chw_ret, so I will inspect it first.",
                       "reasoning": "", "tool_calls": [{"name": "inspect_plant"}], "tool_results": ["{}"]}], "final": "x"}
    stats = _ingest(tmp_path, good)
    assert stats["accepted"] == 1 and stats["stats"].get("accepted_tool_using") == 1


def test_tools_and_on_demand_skills_are_on_by_default():
    names = toolkit.tool_names()
    assert len(names) == 16 and {"list_skills", "load_skill"} <= set(names)
    tools = {f.__name__: f for f in toolkit.make_tools("repair", _row("repair"))}
    assert set(tools) == set(names)
    cat = json.loads(tools["list_skills"](""))                              # blank == empty filter object
    assert cat["ok"] and cat["n"] > 0 and all("text" not in m for m in cat["skills"])
    msgs = TP.build_compact_harness_messages([{"role": "user", "content": "q"}], "repair", None)
    system = msgs[0]["content"]
    assert TP.RATIONALE_LINE in system and "load_skill" in system and "list_skills" in system


def test_ledger_tool_states_recomputed_values_without_naming_the_faulty_line():
    row = _row("ledger")
    out = json.loads({f.__name__: f for f in toolkit.make_tools("ledger", row)}["inspect_plant"](""))
    blob = json.dumps(out)
    table = out["line_table"]
    rows = [dict(zip(table["columns"], row)) for row in table["rows"]]
    assert "truth_MW" not in blob and "failed_lines" not in out
    assert '"ok": false' not in json.dumps(rows).lower()
    assert any(x.get("recomputed_MW") is not None for x in rows)
