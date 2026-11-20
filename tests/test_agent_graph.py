"""The CoT data-gen harness is a LangGraph state machine using REAL numeric physics tools."""
import json

from thermo_agent.teacher import agent_graph
from thermo_agent.teacher import compact_memory
from thermo_agent.teacher import memory as mem
from thermo_agent.teacher.agent_graph import DEFAULT_MAX_ROUNDS, build_context, finalize, run
from thermo_agent.teacher.toolkit import SKILL_TOOL_NAMES, make_tools, tool_names
from thermo_agent.teacher.thermo_tools import DYNAMIC_TOOL_NAMES


def _row():
    return json.loads(open("data/processed/repair_train.jsonl", encoding="utf-8").readline())


def _note(note_id, record_id):
    """A rejected failure note whose text overlaps the query below (so retrieval returns it)."""
    return mem.Note(id=note_id, task="repair", fault_type="unit_slip",
                    architecture="dc_chp", symptoms="unbalanced plant",
                    lesson="wrong power of ten", fix="SET eta = 0.85",
                    source="test", accepted=False, record_id=record_id)


def test_tools_are_real_solver_results():
    assert tool_names() == ["inspect_plant", "first_law_check", "second_law_check", "validity_check", "solve_plant"]
    assert tool_names(include_dynamic=True) == tool_names() + list(DYNAMIC_TOOL_NAMES) + list(SKILL_TOOL_NAMES)
    assert len(tool_names(include_dynamic=True)) == 16
    row = _row()
    gt = row["completion"][0]["content"]
    t = {x.__name__: x for x in make_tools("repair", row)}
    first = json.loads(t["first_law_check"](gt))
    assert first["ok"] is True
    assert first["max_mass_residual_kg_s"] == 0.0
    assert first["max_energy_residual_W"] == 0.0
    second = json.loads(t["second_law_check"](gt))
    assert second["ok"] is True and "total_exergy_destruction_MW" in second
    solve = json.loads(t["solve_plant"](gt))
    assert solve["ok"] and solve["n_streams"] > 0 and "T_K" in solve["streams"][0]


def test_graph_self_corrects_and_records_trajectory():
    row = _row()
    gt = row["completion"][0]["content"]

    def respond(messages, tools):
        if any("Tool results:" in m["content"] for m in messages):
            return gt, [], "balances close"
        return gt, [{"name": "first_law_check", "arguments": {"candidate": gt}}], "checking"

    state = run(row, "repair", respond, max_rounds=DEFAULT_MAX_ROUNDS)
    assert state["accepted"] is True
    assert state["verdict"]["accepted"] is True
    sft = state["sft_completion"]
    assert "<tool_call" in sft and "<tool_result" in sft
    assert "mass_residual_kg_s" in sft  # a real numeric tool result is in the SFT text
    assert state["final"] == gt
    assert sft.rstrip().endswith(gt)  # the accepted answer is always the last block


def test_finalize_guarantees_nonempty_sft_ending_in_final_answer():
    row = _row()
    gt = row["completion"][0]["content"]

    # No recorded turns: fall back to a single assistant turn holding the final answer.
    out = finalize({"task": "repair", "row": row, "last_text": gt,
                    "last_reasoning": "", "turns": []})
    assert out["turns"] == [{"assistant": gt, "reasoning": "",
                             "tool_calls": [], "tool_results": []}]
    assert out["sft_completion"].strip() == gt
    assert out["final"] == gt

    # Recorded turns with an empty answer must not yield empty SFT text.
    out2 = finalize({"task": "repair", "row": row, "last_text": gt,
                     "last_reasoning": "",
                     "turns": [{"assistant": "",
                                "reasoning": "",
                                "tool_calls": [{"name": "first_law_check", "arguments": {}}],
                                "tool_results": []}]})
    assert out2["sft_completion"].strip() == gt

    # Already ending with the final answer: do not duplicate it.
    turns = [{"assistant": "earlier candidate", "reasoning": "",
              "tool_calls": [], "tool_results": []},
             {"assistant": gt, "reasoning": "", "tool_calls": [], "tool_results": []}]
    out3 = finalize({"task": "repair", "row": row, "last_text": gt,
                     "last_reasoning": "", "turns": turns})
    assert out3["sft_completion"].rstrip().endswith(gt)
    assert out3["sft_completion"].count(gt) == 1


# --------------------------------------------------------------------------- #
# P0-9: one rounds constant, one memory retrieval, no double tool execution
# --------------------------------------------------------------------------- #
def test_system_prompt_states_the_single_rounds_constant():
    # DECISIONS D4 fixes 6 rounds; this test guards that the prompt states the same number the
    # graph enforces, not a second hard-coded value (it asserted 4 while the code said 6).
    assert DEFAULT_MAX_ROUNDS == 6
    row = _row()
    ctx = build_context({"row": row, "task": "repair", "notes": [], "mode": "agents+skills"})
    assert "up to %d rounds" % DEFAULT_MAX_ROUNDS in ctx["system_prompt"]

    # A driver override is forwarded into build_context, so the prompt states the number
    # actually passed to the adapter instead of a hard-coded constant.
    ctx2 = build_context({"row": row, "task": "repair", "notes": [],
                          "mode": "agents+skills", "max_rounds": 7})
    assert "up to 7 rounds" in ctx2["system_prompt"]
    assert "up to %d rounds" % DEFAULT_MAX_ROUNDS not in ctx2["system_prompt"]


def test_build_context_retrieves_memory_once_and_notes_used_match_served(monkeypatch):
    row = {"id": "row-current",
           "prompt": [
               {"role": "system", "content": "You are a plant engineer."},
               {"role": "user", "content": "unit_slip dc_chp unbalanced plant wrong power of ten"},
           ]}
    notes = [
        _note("note-current", "row-current"),  # mined from this very row -> must be excluded
        _note("note-a", "rep-a"),
        _note("note-b", "rep-b"),
    ]

    calls = []
    real_retrieve = mem.retrieve

    def spy_retrieve(query, notes_, k=3, task=None, exclude_record="", min_score=0.02):
        calls.append({"k": k, "task": task, "exclude_record": exclude_record})
        return real_retrieve(query, notes_, k=k, task=task,
                             exclude_record=exclude_record, min_score=min_score)

    served = []
    real_compact = compact_memory.compact_notes

    def spy_compact(notes_, k=5):
        served.append(list(notes_))  # exactly the notes rendered into the system prompt
        return real_compact(notes_, k=k)

    monkeypatch.setattr(mem, "retrieve", spy_retrieve)
    monkeypatch.setattr(compact_memory, "compact_notes", spy_compact)

    ctx = build_context({"row": row, "task": "repair", "notes": notes,
                         "k_memory": 3, "mode": "agents+skills"})

    assert len(calls) == 1, "memory must be retrieved exactly once per episode"
    assert calls[0]["exclude_record"] == row["id"]
    assert calls[0]["task"] == "repair"
    assert len(served) == 1
    assert ctx["notes_used"], "the fixture must actually retrieve at least one note"
    assert ctx["notes_used"] == [n.id for n in served[0]], (
        "notes_used must be exactly the notes rendered into the served context")
    assert "note-current" not in ctx["notes_used"], "the current record's note was served"


def test_four_tuple_finalizes_without_reexecuting_tools(monkeypatch):
    """A full 4-tuple trajectory jumps to finalize; the graph never runs the tools again."""
    row = _row()
    counter = {"n": 0}

    def spy_tool(candidate):
        counter["n"] += 1
        return "SPY RESULT"

    spy_tool.__name__ = "spy_tool"
    monkeypatch.setattr(agent_graph, "make_tools", lambda task, row_, **kw: [spy_tool])

    answer = row["completion"][0]["content"]
    turns = [{"assistant": answer, "reasoning": "done", "tool_calls": [], "tool_results": []}]

    def respond(messages, tools):
        # The adapter already ran the tool in its native loop; the 4-tuple still carries the
        # call. The graph must discard those calls and finalize with the recorded turns.
        return answer, [{"name": "spy_tool", "arguments": {"candidate": "REEXEC_MARKER"}}], "done", turns

    state = run(row, "repair", respond)

    assert counter["n"] == 0, "LangGraph re-executed a tool from the 4-tuple trajectory"
    assert state["done"] is True
    assert state["rounds"] == 0
    assert state["last_calls"] == []
    assert state["turns"] == turns


def test_spy_would_catch_a_second_execution_on_the_three_tuple_path(monkeypatch):
    """Sensitivity control: the spy above really does observe the graph's own tool execution."""
    row = _row()
    counter = {"n": 0}

    def spy_tool(candidate):
        counter["n"] += 1
        return "SPY RESULT"

    spy_tool.__name__ = "spy_tool"
    monkeypatch.setattr(agent_graph, "make_tools", lambda task, row_, **kw: [spy_tool])

    answer = row["completion"][0]["content"]

    def respond(messages, tools):
        if any("Tool results:" in m["content"] for m in messages):
            return answer, [], "done"
        return answer, [{"name": "spy_tool", "arguments": {"candidate": "x"}}], "checking"

    state = run(row, "repair", respond)
    assert counter["n"] == 1
    assert state["rounds"] == 1

def test_build_context_records_served_skill_and_prompt_hash():
    """The trace must be able to prove which skill was served and the exact prompt bytes."""
    row = {"id": "row-design",
           "prompt": [{"role": "system", "content": "You design plants."},
                      {"role": "user", "content": "SITE it_load_w=1"}]}
    ctx = build_context({"row": row, "task": "design", "notes": [],
                         "mode": "agents+skills"})
    assert ctx["served_skill"] == "design-catalog"
    assert len(ctx["system_prompt_sha256"]) == 64
    assert ctx["memory_block"] == ""
    # Repair now has its own diagnosis skill; the served id tracks TASK_SKILLS.
    repair = build_context({"row": row, "task": "repair", "notes": [],
                            "mode": "agents+skills"})
    assert repair["served_skill"] == "repair-diagnosis"


def test_build_context_dynamic_opt_in_controls_tools_and_prompt():
    """The hosted flag decides both the executable tool list and the served contract text."""
    row = _row()
    base = {"row": row, "task": "repair", "notes": [], "mode": "agents+skills"}
    default = build_context(base)
    dynamic = build_context({**base, "include_dynamic": True})

    assert [t.__name__ for t in default["tools"]] == tool_names()
    assert [t.__name__ for t in dynamic["tools"]] == tool_names(include_dynamic=True)
    assert len(dynamic["tools"]) == 16
    assert "check_answer" not in dynamic["tool_map"]
    assert default["include_dynamic"] is False and dynamic["include_dynamic"] is True

    # The compact contract names the dynamic tools only when opted in.
    assert all(n not in default["system_prompt"] for n in DYNAMIC_TOOL_NAMES)
    assert all(n in dynamic["system_prompt"] for n in DYNAMIC_TOOL_NAMES)
    assert "JSON object string" in dynamic["system_prompt"]
    assert "JSON object string" not in default["system_prompt"]
    # execute a dynamic tool through the same fn(candidate) parser path the graph uses.
    fn = dynamic["tool_map"]["fluid_state"]
    import json as _json
    out = _json.loads(fn(_json.dumps({"fluid": "Water", "T_K": 300.0, "P_kPa": 101.325})))
    assert out["ok"] is True and out["fluid"] == "Water"


def test_build_context_records_the_exact_memory_block(monkeypatch):
    row = {"id": "row-current",
           "prompt": [{"role": "system", "content": "You are a plant engineer."},
                      {"role": "user", "content": "unit_slip dc_chp unbalanced plant wrong power of ten"}]}
    notes = [_note("note-a", "rep-a"), _note("note-b", "rep-b")]
    ctx = build_context({"row": row, "task": "repair", "notes": notes,
                         "k_memory": 3, "mode": "agents+skills"})
    assert ctx["notes_used"], "fixture must retrieve at least one note"
    for line in ctx["memory_block"].splitlines()[1:]:
        assert line.startswith("- "), "served block must be the compact render"
    assert ctx["memory_block"] in ctx["system_prompt"]
