"""P0-1: a harness record's tool trace must survive ingest as the SFT completion.

The drivers journal `trajectory` (with `final`) plus a separate `sft_completion`.
`transfer.ingest` merges those two before calling `teacher_row`; this test round-trips
that synthetic record through `teacher_row` and checks the tool-using completion is kept
and labelled correctly.
"""
from thermo_agent.teacher import validate

FINAL = ("<diagnosis>Final repair after tool feedback.</diagnosis>\n"
         "<edits>\nSET @HEATX.eta = 0.85\n</edits>")
FIRST_GUESS = "<diagnosis>First guess.</diagnosis>\n<edits>\nSET @HEATX.eta = 8.5\n</edits>"
CALL = '<tool_call name="first_law_check">{"candidate": "..."}</tool_call>'
RESULT = "<tool_result>energy residual 1.2e7 W (too large)</tool_result>"


def verdict():
    return {"accepted": True, "reasons": ["restored a valid plant"], "reward": 1.0,
            "failed": [], "exact": True}


def row():
    return {"id": "row-1", "split": "train", "topology": "dc_chp",
            "prompt": [{"role": "user", "content": "Repair the plant."}]}


def harness_record(sft_completion: str) -> dict:
    """Shape written by scripts/kaggle_harness_cli.py / kaggle/teacher_harness.py."""
    return {
        "id": "row-1", "task": "repair", "model": "gemini-3.5-flash-lite", "harness": "tools",
        "completion": FINAL,
        "trajectory": {
            "turns": [{"assistant": FIRST_GUESS, "tool_calls": [{"name": "first_law_check", "arguments": {}}],
                       "tool_results": [RESULT]}],
            "notes_used": [], "tools_available": ["first_law_check"], "final": FINAL,
        },
        "sft_completion": sft_completion,
    }


def ingested_row(record: dict) -> dict:
    """What transfer.ingest passes to validate.teacher_row for a journal record."""
    trajectory = dict(record["trajectory"])
    trajectory["sft_completion"] = record["sft_completion"]
    return validate.teacher_row("repair", record["completion"], row(), verdict(),
                                tag="gemini-3-5-flash-lite", trajectory=trajectory)


def test_tool_using_trajectory_becomes_the_tagged_sft_completion():
    sft = ("<think>check the candidate</think>\n" + FIRST_GUESS + "\n" + CALL + "\n" + RESULT +
           "\n<think>scale is wrong, fix it</think>\n" + FINAL)
    out = ingested_row(harness_record(sft))

    completion = out["completion"][0]["content"]
    assert "<tool_call" in completion and "<tool_result" in completion, "tool turns must be kept"
    assert completion == sft, "the full trace, not the plain answer, is the SFT completion"
    assert completion.endswith(FINAL), "the accepted final answer ends the completion"
    assert out["source"] == "frontier_tools"
    assert out["trajectory"]["final"] == FINAL, "final must be persisted for provenance"


def test_record_without_tool_turns_is_not_tagged_frontier_tools():
    # Trajectory + final are present, but the stored completion has no <tool_call>/<tool_result>.
    sft = "<think>no tools needed</think>\n" + FINAL
    out = ingested_row(harness_record(sft))

    completion = out["completion"][0]["content"]
    assert "<tool_call" not in completion and "<tool_result" not in completion
    assert out["source"] != "frontier_tools"
    assert out["source"] == "frontier"
