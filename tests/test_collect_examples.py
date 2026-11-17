"""Paper examples must be deterministic excerpts of real stage outputs, with a schema beside them."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_examples", ROOT / "paper/tools/collect_examples.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_examples_come_from_each_stage_with_a_schema(tmp_path):
    dg = tmp_path / "processed"; plant = json.dumps({"name": "p", "components": [{"type": "gas_engine"}], "streams": []})
    _write(dg / "repair_test_id.jsonl", [
        {"id": "r1", "fault_type": "wrong_fluid", "topology": "dc_engine_absorption", "plant_broken": plant,
         "completion": "<diagnosis>wrong fluid at CC</diagnosis>", "prompt": [{"role": "user", "content": "fix it"}]},
        {"id": "r2", "fault_type": "unit_slip", "topology": "dc_chp", "plant_broken": plant,
         "completion": "<diagnosis>x</diagnosis>", "prompt": [{"role": "user", "content": "fix it"}]}])
    sft = tmp_path / "sft"
    _write(sft / "repair_train.jsonl", [
        {"id": "s1", "harness": "agents+skills", "memory_notes": ["note"], "completion": [{"content": "answer"}]}])
    harness = tmp_path / "harness.jsonl"
    _write(harness, [{"id": "h1", "task": "repair", "tool_calls": [{"name": "check_answer"}] * 3,
                      "completion": "final", "reasoning": "thought", "usage": {"total_cost_nanodollars": 10}}])
    roll = tmp_path / "rollouts.jsonl"
    _write(roll, [{"group": 1, "stage": "grpo_ledger", "reward_fn": "l_exact", "prompt": "p",
                   "completions": ["a", "b"], "rewards": [0.1, 0.9]}])
    args = argparse = type("A", (), {"datagen": [dg], "sft": [sft], "frontier": [], "harness": [harness],
                                     "rollouts": [roll], "evals": []})()
    out = mod.collect(args)
    sources = {e["source"] for e in out["examples"]}
    assert {"datagen", "sft", "harness", "grpo"} <= sources
    assert out["schema"]["examples"] and out["n"] == len(out["examples"])
    dat = next(e for e in out["examples"] if e["source"] == "datagen")
    assert dat["graph"]["components"] == ["gas_engine"]
    harness_ex = next(e for e in out["examples"] if e["source"] == "harness")
    assert len(harness_ex["fields"]["tool_calls"]) == 3
