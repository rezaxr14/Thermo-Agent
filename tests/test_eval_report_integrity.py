"""CPU-only integrity checks for the frozen eval protocol (docs/EVAL_PROTOCOL.md).

Four defects are guarded here: silent truncation of held-out cells, missing provenance on
reports, a fake Z-vs-H contrast between ours_sft/ours_harness, and Holm correction applied
across unrelated comparisons instead of within named families. Everything is stubbed: no
GPU, no model run, no network -- eval_runner is imported by path with Kaggle helpers stubbed
and main() is driven with a fake ``run`` that uses the real row loader.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import types
from unittest import mock

import pytest

from thermo_agent.eval import compare as CMP
from thermo_agent.eval import model_eval as ME

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_RUNNER = ROOT / "kaggle" / "eval_runner.py"
ARMS = ("ours_base", "ours_sft", "ours_harness", "ours_sft_grpo_k4", "ours_sft_grpo_k8",
        "ours_sft_grpo_k16", "frontier_zero", "frontier_tools", "frontier_harness", "other_model")


# ---------------------------------------------------------------------------
# Fake held-out data + a fake scoring pass that still uses the real row loader.
# ---------------------------------------------------------------------------

def _heldout(tmp_path, n=70, task="repair", split="test_ood_fault"):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    ids = [f"{task}-{split}-{i:03d}" for i in range(n)]
    rows = "\n".join(json.dumps({"id": rid, "split": split}) for rid in ids) + "\n"
    (d / f"{task}_{split}.jsonl").write_text(rows, encoding="utf-8")
    return d, ids


def _fake_run():
    """Mirror run()'s cell shape but replace scoring with the exact ids load_rows would score."""
    real_load = ME.load_rows

    def run(generate, data_dirs, tasks, splits=None, limit=40, mode="none", max_turns=None,
            polish_evals=0, refine_turns=0, tool_access="none", max_tool_calls=None, max_new_tokens=None):
        report = {}
        for task in tasks:
            report[task] = {}
            for sp in (splits or ME.DEFAULT_SPLITS[task]):
                rows = real_load(data_dirs, task, sp, limit)
                report[task][sp] = {"n": len(rows) if rows else 0,
                                    "examples": [{"id": r["id"]} for r in rows or []]}
        report["harness_config"] = ME.harness_config(mode, tool_access, max_turns,
                                                     max_tool_calls, max_new_tokens)
        return report, []

    return run


def _call_main(monkeypatch, data, out, limit, *, tasks=("repair",), splits=("test_ood_fault",)):
    """Drive model_eval.main() end-to-end with a replay backend and stubbed scoring."""
    class DummyGen:
        def __init__(self, *a, **k):
            pass

        def __call__(self, messages):
            return ""

    replay = out.with_suffix(".replay.jsonl")
    replay.write_text("", encoding="utf-8")
    monkeypatch.setattr(ME, "run", _fake_run())
    monkeypatch.setattr(ME, "HFGenerator", DummyGen)
    monkeypatch.setattr(sys, "argv", ["model_eval", str(data), "--backend", "replay",
                                      "--replay", str(replay), "--tasks", *tasks,
                                      "--splits", *splits, "--limit", str(limit),
                                      "--out", str(out)])
    ME.main()
    return json.loads(out.read_text(encoding="utf-8"))


def test_limit_zero_scores_every_row_and_a_forced_30_raises_loudly(tmp_path, monkeypatch):
    data, ids = _heldout(tmp_path, 70)  # repair/test_ood_fault has 70 rows in the frozen set
    good = _call_main(monkeypatch, data, tmp_path / "full.json", 0)
    assert good["repair"]["test_ood_fault"]["n"] == 70
    assert good["meta"]["split_ids"]["repair"]["test_ood_fault"] == ids

    bad = tmp_path / "truncated.json"
    with pytest.raises(ValueError, match=r"repair/test_ood_fault") as exc:
        _call_main(monkeypatch, data, bad, 30)
    message = str(exc.value)
    assert "30" in message and "70" in message, message
    assert not bad.exists(), "a truncated run must not leave a report behind"


PROVENANCE = ("eval_set_version", "eval_set_hash", "split_ids", "harness_config_hash",
              "git_commit", "data_dirs")


def test_report_carries_every_provenance_field(tmp_path, monkeypatch):
    data, ids = _heldout(tmp_path, 70)
    rep = _call_main(monkeypatch, data, tmp_path / "prov.json", 0)
    meta = rep["meta"]
    assert set(PROVENANCE) <= set(meta)
    assert "arm_config" in rep and "arm_config" in meta
    expected_version = ME.eval_set_hash([data], ["repair"], ["test_ood_fault"])
    assert meta["eval_set_version"] == meta["eval_set_hash"] == expected_version
    assert len(meta["eval_set_version"]) == 64
    assert meta["split_ids"]["repair"]["test_ood_fault"] == ids
    assert meta["data_dirs"] == [str(data)]
    assert meta["git_commit"] is None or isinstance(meta["git_commit"], str)
    assert isinstance(meta["harness_config_hash"], str) and len(meta["harness_config_hash"]) == 64


def test_harness_config_hash_changes_with_limit_and_is_stable_otherwise(tmp_path, monkeypatch):
    data, _ = _heldout(tmp_path / "hash", 70)
    first = _call_main(monkeypatch, data, tmp_path / "a.json", 0)
    again = _call_main(monkeypatch, data, tmp_path / "b.json", 0)
    capped = _call_main(monkeypatch, data, tmp_path / "c.json", 70)
    h = first["meta"]["harness_config_hash"]
    assert h == again["meta"]["harness_config_hash"]
    assert h != capped["meta"]["harness_config_hash"]
    assert h == ME.harness_config_hash(first["meta"]["harness_config"], 0)


# ---------------------------------------------------------------------------
# eval_runner arm stubs (same approach as tests/test_eval_runner_arms.py).
# ---------------------------------------------------------------------------

def _fake_kaggle_notebook(root, replay_source):
    mod = types.ModuleType("thermo_agent.kaggle_notebook")
    mod.setup = lambda *a, **k: root
    mod.find_input = lambda name, *a, **k: replay_source
    mod.run = lambda *a, **k: None
    mod.token = lambda *a, **k: ""
    mod.assert_pinned_environment = lambda *a, **k: None
    return mod


def _load_runner(tmp_path, monkeypatch, name="integrity_runner"):
    root = tmp_path / "worker" / "thermo-agent"
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("paper/assets", "reports", "runs", "data", "replay_inputs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # eval_runner refuses to score before checking the frozen held-out corpus at import time.
    heldout = root / "data" / "heldout"
    heldout.mkdir(parents=True, exist_ok=True)
    (heldout / "repair_test_id.jsonl").write_text('{"id": "held-1"}\n', encoding="utf-8")
    for key in ("HF_TOKEN", "ADAPTER", "GRPO_ADAPTER", "FORCE", "SPLITS", "LIMIT", "DATA_DIRS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TASK", "ledger")
    monkeypatch.setenv("ARMS", ",".join(ARMS))
    for k in (4, 8, 16):
        ck = root / "runs" / f"grpo_ledger_k{k}" / "checkpoint-1"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "adapter_config.json").write_text("{}", encoding="utf-8")
    merged = root / "runs" / "sft_merged"
    merged.mkdir(exist_ok=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "thermo_agent.kaggle_notebook",
                        _fake_kaggle_notebook(root, root / "replay_inputs"))
    cmds = []

    def fake_run(argv, *a, **k):
        cmds.append([str(x) for x in argv])
        return subprocess.CompletedProcess(argv, 0)

    with mock.patch("os.walk", side_effect=lambda *a, **k: iter([("/kaggle/input", [], ["kaggle_bundle.zip"])])), \
         mock.patch("zipfile.ZipFile"), mock.patch("shutil.rmtree"), \
         mock.patch("os.chdir"), mock.patch("subprocess.run", side_effect=fake_run):
        spec = importlib.util.spec_from_file_location(name, EVAL_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, cmds


def _capture(module, arm):
    cmds = []

    def fake_run(argv, *a, **k):
        cmds.append([str(x) for x in argv])
        return subprocess.CompletedProcess(argv, 0)

    with mock.patch("subprocess.run", side_effect=fake_run):
        module.run_arm(arm)
    return cmds[-1]


def test_ours_sft_is_zeroshot_and_ours_harness_is_a_real_h_arm(tmp_path, monkeypatch):
    module, _ = _load_runner(tmp_path, monkeypatch)
    assert module.LIMIT == 0, "LIMIT must default to 'all rows' for the frozen held-out set"
    assert module.SPECS["ours_sft"]["tool_access"] == "none"
    assert module.SPECS["ours_harness"]["tool_access"] == "tools"

    sft, harness = _capture(module, "ours_sft"), _capture(module, "ours_harness")
    assert sft[sft.index("--tool-access") + 1] == "none"
    assert harness[harness.index("--tool-access") + 1] == "tools"
    assert harness[harness.index("--harness") + 1] == "agents+skills"
    assert sft[sft.index("--limit") + 1] == harness[harness.index("--limit") + 1] == "0"

    # Once output/display labels are ignored, --tool-access is the ONLY difference.
    def normalise(cmd):
        out, skip = [], False
        for token in cmd:
            if skip:
                skip = False
                continue
            if token in ("--out", "--display-name"):
                skip = True
                continue
            out.append(token)
        return out

    a, b = normalise(sft), normalise(harness)
    assert len(a) == len(b)
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diffs) == 1, diffs
    i, x, y = diffs[0]
    assert a[i - 1] == b[i - 1] == "--tool-access" and (x, y) == ("none", "tools")

    # Both configs are recorded in arm_config, so a report cannot be read as the wrong arm.
    assert module.arm_config("ours_sft")["tool_access"] == "none"
    assert module.arm_config("ours_harness")["tool_access"] == "tools"


# ---------------------------------------------------------------------------
# Holm families.
# ---------------------------------------------------------------------------

def _discordant_report(splits):
    """A and B differ on A-only ids in `splits = {split: {metric: n_hits}}`."""
    a, b = {"repair": {}}, {"repair": {}}
    for split, metrics in splits.items():
        ids = [f"e{i}" for i in range(10)]
        ea, eb = [{"id": rid} for rid in ids], [{"id": rid} for rid in ids]
        for metric, hits in metrics.items():
            hit_ids = {f"e{i}" for i in range(hits)}
            for ex in ea:
                ex[metric] = 1 if ex["id"] in hit_ids else 0
            for ex in eb:
                ex[metric] = 0
        a["repair"][split] = {"examples": ea}
        b["repair"][split] = {"examples": eb}
    return a, b


def test_holm_corrects_within_named_families_only():
    assert len(CMP.HEADLINE_FAMILIES) == 4
    a, b = _discordant_report({
        "test_id": {"success": 3, "first_turn_success": 1},  # raw p = 0.25 and 1.0
        "test_ood_fault": {"success": 3},                    # raw p = 0.25, alone in family B
    })
    rows = CMP.compare_families({
        "A": [(a, b, "repair", "test_id", "success"),
              (a, b, "repair", "test_id", "first_turn_success")],
        "B": [(a, b, "repair", "test_ood_fault", "success")],
    })
    assert all("family" in r and "p_holm" in r for r in rows)
    fa = {r["metric"]: r for r in rows if r["family"] == "A"}
    fb = [r for r in rows if r["family"] == "B"]
    assert set(fa) == {"success", "first_turn_success"} and len(fb) == 1
    assert fa["success"]["p"] == pytest.approx(0.25)
    assert fa["success"]["p_holm"] == pytest.approx(0.5)   # corrected against m=2 only
    assert fa["first_turn_success"]["p"] == pytest.approx(1.0)
    assert fa["first_turn_success"]["p_holm"] == pytest.approx(1.0)
    assert fb[0]["p"] == pytest.approx(0.25)
    assert fb[0]["p_holm"] == pytest.approx(0.25)          # not 0.75: no cross-family correction


def test_legacy_compare_still_corrects_all_rows_together():
    a, b = _discordant_report({"test_id": {"success": 3, "first_turn_success": 1}})
    rows = CMP.compare(a, b, ["repair"])
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["success"]["p_holm"] == pytest.approx(0.5)
    assert all("family" not in r for r in rows)
