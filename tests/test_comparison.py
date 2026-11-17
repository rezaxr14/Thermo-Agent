"""The paper's model comparison must be built from report JSON, not hand-written cells."""
import json

from thermo_agent.eval import comparison


def _report(path, model, family, harness, tools, zero_shot, success):
    path.write_text(json.dumps({
        "meta": {"model": model, "display_name": model, "model_family": family, "harness": harness,
                 "tool_access": tools, "zero_shot": zero_shot},
        "repair": {"test_id": {"n": 20, "success": success, "success_ci95": [0.4, 0.8],
                               "first_turn_success": success, "exact_on_deducible": 0.2, "mean_solves": 4.0}},
    }), encoding="utf-8")


def test_manifest_axes_flow_into_comparison_rows(tmp_path):
    r1, r2 = tmp_path / "ours.json", tmp_path / "frontier.json"
    _report(r1, "qwen4b", "ours", "agents+skills", "none", False, 0.6)
    _report(r2, "frontier-x", "frontier", "none", "none", True, 0.3)
    manifest = {
        "our4b": {"report": str(r1)},
        "frontier_zero": {"report": str(r2), "family": "frontier", "harness": "none",
                          "tool_access": "none", "zero_shot": True},
    }
    out = comparison.build(manifest)
    assert out["arms"] == 2 and len(out["rows"]) == 2
    ours, zero = out["rows"]
    assert (ours["model_family"], ours["harness"], ours["zero_shot"]) == ("ours", "agents+skills", False)
    assert (zero["model_family"], zero["tool_access"], zero["zero_shot"]) == ("frontier", "none", True)
    assert ours["metrics"]["success"] == 0.6 and zero["metrics"]["success"] == 0.3
    assert "success" in ours["ci95"]


def test_missing_reports_are_recorded_not_fatal(tmp_path):
    out = comparison.build({"gone": {"report": str(tmp_path / "missing.json")}})
    assert out["rows"] == [] and "missing report" in out["provenance"]["gone"]


def test_manifest_loader_accepts_bare_and_wrapped_maps(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"arms": {"a": {"report": "a.json"}}}', encoding="utf-8")
    loaded = comparison.load_manifest(p, tmp_path)
    assert loaded["a"]["report"] == str(tmp_path / "a.json")
