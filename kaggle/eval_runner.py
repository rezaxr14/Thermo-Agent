"""Kaggle notebook source: run every comparison arm through the same evaluator.

Build with `python kaggle/make_notebook.py`. Each arm writes a `model_eval` report JSON; the final cells build the
comparison manifest, measure tool use and collect paper examples. Running one arm per cell keeps a session bounded
and resumable: reports are never overwritten unless `FORCE=1`.
"""

# %% [markdown]
# # thermo-agent: evaluation of all comparison arms
#
# Attach `thermo-agent-code` (`scripts/make_kaggle_bundle.ps1`), `thermo-agent-corpus` (`data/sft` + `data/grpo`) and
# `thermo-teacher-prompts`. Add the `HF_TOKEN` secret to pull our trained adapter. For the trained-model arms choose
# **GPU T4** (one is enough for greedy generation); the replay and endpoint arms need no GPU.
#
# Arms: DECISIONS §2F F4 -- base 4B / SFT / GRPO k4,k8,k16 each in Z (plain prompt) and H (compact
# harness + tools), frontier Z/H replayed from the frozen Kaggle answer files, and other-model
# placeholders. Legacy ids stay runnable so existing reports remain readable.

# %%
# Bootstrap with the standard library only: unpack the code bundle before importing thermo_agent.
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "thermo-agent"
bundles = []
for _root, _dirs, _files in os.walk("/kaggle/input", followlinks=True):
    if "kaggle_bundle.zip" in _files:
        bundles.append(pathlib.Path(_root) / "kaggle_bundle.zip")
if not bundles:
    raise SystemExit("attach the Kaggle dataset thermo-agent-code")
shutil.rmtree(ROOT, ignore_errors=True)
with zipfile.ZipFile(bundles[0]) as zf:
    zf.extractall(WORK)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# %%
from thermo_agent.kaggle_notebook import assert_pinned_environment, find_input, run, setup, token

# Stage the frozen held-out split and the tool corpora (plus the standard train corpora that setup()
# already staged) so every arm is scored on the same never-seen rows and the tool arms have data.
ROOT = setup(corpora=("sft", "grpo", "sft_all", "grpo_all", "heldout", "sft_tools_v3_h", "grpo_tools_v3_h"))
# Pin check in the notebook itself; no source builds in any Kaggle worker.
assert_pinned_environment(ROOT / "kaggle" / "environment.lock.json")

# %%
def require_test_corpus(name):
    """Fail before scoring if the frozen held-out corpus has no *_test_*.jsonl splits."""
    d = ROOT / "data" / name
    tests = sorted(str(p) for p in d.glob("*_test_*.jsonl")) if d.is_dir() else []
    if not tests:
        raise SystemExit(
            f"[corpus] no *_test_*.jsonl in {d} (missing path: {d / '*_test_*.jsonl'}); "
            f"attach a corpus dataset with a {name} test split before scoring"
        )
    print(f"[corpus] {name}: {len(tests)} test split(s)", flush=True)
    return d


# %%
# DATA_DIRS defaults to heldout; fail loudly here rather than emitting empty reports downstream.
require_test_corpus("heldout")

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
# LIMIT=0 means "no cap": score every row in every held-out split. The frozen eval set has cells
# up to 70 rows (repair/test_ood_fault), so any positive default would silently truncate them.
# model_eval treats --limit 0 as no cap and refuses to emit a report whose scored ids are fewer
# than the split's manifest ids, so an accidental override fails loudly instead of biasing a number.
LIMIT = int(os.environ.get("LIMIT", "0"))
FORCE = os.environ.get("FORCE", "0") == "1"
SPLITS = os.environ.get("SPLITS", "").split(",") if os.environ.get("SPLITS") else None
# DATA_DIRS: which ground truth to score on. Default to the frozen held-out set (data/heldout,
# fresh seed 987654): every arm is scored on the same never-seen rows, and it is excluded from
# every training corpus and teacher shard. Same splits for every arm (SPLITS, if set, applies to all).
DEFAULT_DATA_DIRS = "heldout"
DATA_DIRS = [ROOT / "data" / d for d in os.environ.get(
    "DATA_DIRS", DEFAULT_DATA_DIRS).split(",")]
# F4 (DECISIONS §2F): every local family is scored in BOTH modes. The legacy ids are kept so old
# reports stay readable; the *_z / *_h ids are the explicit Z (plain prompt, no harness, no tools)
# and H (compact harness, tools, memory) arms of the F4 table.
F4_LOCAL_FAMILIES = ("ours_base", "ours_sft", "ours_sft_grpo_k4",
                     "ours_sft_grpo_k8", "ours_sft_grpo_k16")
DEFAULT_ARMS_LIST = (
    # legacy ids that existing reports/manifests already use
    "ours_base,ours_sft,ours_harness,ours_sft_grpo_k4,ours_sft_grpo_k8,"
    "ours_sft_grpo_k16,frontier_zero,frontier_tools,frontier_harness,other_model,"
    # explicit F4 Z / H table for the local families
    "ours_base_z,ours_base_h,ours_sft_z,ours_sft_h,"
    "ours_sft_grpo_k4_z,ours_sft_grpo_k4_h,ours_sft_grpo_k8_z,ours_sft_grpo_k8_h,"
    "ours_sft_grpo_k16_z,ours_sft_grpo_k16_h,"
    # clear frontier Z/H names (same frozen replay files as the legacy frontier ids)
    "frontier_z,frontier_h")
ARMS = os.environ.get("ARMS", DEFAULT_ARMS_LIST).split(",")
find_input("thermo-teacher-prompts")     # fail early if the prompt dataset is missing

# %%
# Stage the saved teacher answers used by the replay arms. They come from a small private
# dataset; without it the frontier arms have nothing to score in the worker.
REPLAY_FILES = ("eval_all.jsonl", "harness_eval_completions2.jsonl",
                "harness_completions.jsonl", "eval_gptnano_replay.jsonl")
try:
    replay_source = find_input("thermo-agent-eval-replays")
except SystemExit:
    replay_source = None
    print("[replay] no replay dataset attached; frontier arms are unavailable", flush=True)
if replay_source is not None:
    for name in REPLAY_FILES:
        target = WORK / name
        if target.exists():
            continue
        found = next((p for p in replay_source.rglob(name) if p.is_file()), None)
        if found:
            shutil.copyfile(found, target)
            print("[replay] staged", name, flush=True)

# %%
from thermo_agent.train import grpo as grpo_train, hf_sync
from thermo_agent.eval import comparison, tool_usage

token()
# Pull the newest SFT adapter and merged model from the same private repo the trainer used.
REPO = hf_sync.PROJECT_REPO_ID
sft_root = ROOT / "runs"
if os.environ.get("HF_TOKEN"):
    pulled = hf_sync.pull_checkpoint(sft_root, REPO, "sft", os.environ["HF_TOKEN"])
    print("SFT adapter:", pulled)
    merged = hf_sync.pull_merged(sft_root / "sft_merged", REPO, os.environ["HF_TOKEN"], prefix="merged/sft")
    print("merged model:", merged)
else:
    pulled, merged = None, None
ADAPTER = pulled or os.environ.get("ADAPTER", "")
# With no Kaggle Secret, the trained arms take the adapter from a local checkpoint in the
# attached dataset/code tree via ADAPTER (and GRPO_ADAPTER for the RL stage). No token is stored.
GRPO_ADAPTER = os.environ.get("GRPO_ADAPTER", merged or ADAPTER)
# GRPO LoRAs are trained on top of the merged SFT weights, so score them on that same base -- never
# on the raw Qwen weights (model_eval's default). TASK is the single task each checkpoint was trained on.
SFT_MERGED = str(merged or (ROOT / "runs" / "sft_merged"))
TASK = os.environ.get("TASK", "ledger")


def _has_adapter_weights(path):
    """True only for a real adapter/checkpoint dir with something loadable, not an empty placeholder."""
    p = pathlib.Path(path) if path else None
    if not p or not p.is_dir():
        return False
    if any((p / f).exists() for f in ("adapter_config.json", "adapter_model.safetensors", "config.json")):
        return True
    return any(c.is_dir() and c.name.startswith("checkpoint-") for c in p.iterdir())


def require_adapter(path, stage):
    """A missing adapter is a hard error: never score the base model under a GRPO label."""
    if not _has_adapter_weights(path):
        raise SystemExit(
            f"GRPO adapter for stage {stage} is missing or empty at {path}; "
            "refusing to score the base model under a GRPO label"
        )
    return path


# Pull each distinct k checkpoint into its own dir so k=4/8/16 never share weights. C9 writes the
# joint stage grpo_k{k}; we still read the legacy grpo_{task}_k{k} dir for older checkpoints.
GRPO_K_ADAPTERS = {}
for _k in (4, 8, 16):
    _aliases = grpo_train.stage_names_for_reading(TASK, _k)  # (joint, legacy)
    _dir = None
    for _stage in _aliases:
        _candidate = ROOT / "runs" / _stage
        if os.environ.get("HF_TOKEN"):
            _pulled = hf_sync.pull_checkpoint(ROOT / "runs", REPO, _stage, os.environ["HF_TOKEN"])
            if _pulled:
                _candidate = pathlib.Path(_pulled)
        if _has_adapter_weights(_candidate):
            _dir = _candidate
            break
    if _dir is None:
        _dir = ROOT / "runs" / _aliases[0]   # where a new joint run writes its checkpoints
    GRPO_K_ADAPTERS[_k] = require_adapter(_dir, _aliases[0])

# %%
# One spec per arm. `replay` scores saved Kaggle answers with the same verifiers and prompt rows; the frontier
# export already carries its harness text, so zero-shot here means "no extra inference text beyond the prompt".
#
# F4 modes (DECISIONS §2F): `mode` is the arm's evaluation mode, not a generation flag.
#   Z: plain prompt -- harness "none", tool_access "none" (no inference text, no tools).
#   H: compact harness -- harness "agents+skills", tool_access "tools" (tool loop + memory).
# Legacy ids (ours_sft, ours_harness, ours_grpo, ours_sft_grpo_k*, frontier_*, other_model) are kept
# exactly as reports already use them; the *_z / *_h ids are the explicit F4 counterparts. Every spec
# carries zero_shot, harness and tool_access. GRPO arms keep tools in H only: their Z arm never sets
# tool_access=tools, and unlike the legacy *_k* ids the table arms do NOT carry train_task (the
# static guard that identifies the four legacy GRPO checkpoints stays valid).
SPECS = {
    # --- ours base 4B -------------------------------------------------------------------
    "ours_base": {"backend": "hf", "mode": "Z", "model": "Qwen/Qwen3.5-4B", "family": "ours", "harness": "none",
                  "tool_access": "none", "zero_shot": True},
    "ours_base_z": {"backend": "hf", "mode": "Z", "model": "Qwen/Qwen3.5-4B", "family": "ours", "harness": "none",
                    "tool_access": "none", "zero_shot": True},
    "ours_base_h": {"backend": "hf", "mode": "H", "model": "Qwen/Qwen3.5-4B", "family": "ours",
                    "harness": "agents+skills", "tool_access": "tools", "zero_shot": True},
    # --- ours SFT (legacy "SFT none" + explicit Z/H counterparts) ------------------------
    "ours_sft": {"backend": "hf", "mode": "legacy", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER,
                 "family": "ours", "harness": "agents+skills", "tool_access": "none", "zero_shot": False},
    "ours_sft_z": {"backend": "hf", "mode": "Z", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER,
                   "family": "ours", "harness": "none", "tool_access": "none", "zero_shot": False},
    "ours_sft_h": {"backend": "hf", "mode": "H", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER,
                   "family": "ours", "harness": "agents+skills", "tool_access": "tools", "zero_shot": False},
    # The real Z-vs-H contrast: *_z is prompt-only (no tools), *_h is the tool-using
    # H mode (--tool-access tools --harness agents+skills). Identical weights, different inference.
    "ours_harness": {"backend": "hf", "mode": "legacy", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER,
                     "family": "ours", "harness": "agents+skills", "tool_access": "tools", "zero_shot": False},
    # --- ours GRPO (legacy H arms + explicit Z/H counterparts) ---------------------------
    # GRPO is trained inside the multi-turn tool environment, so the legacy k arms are H (tools).
    "ours_grpo": {"backend": "hf", "mode": "legacy", "model": SFT_MERGED, "adapter": GRPO_ADAPTER,
                  "family": "ours", "harness": "agents+skills", "tool_access": "tools", "zero_shot": False,
                  "train_task": TASK},
    # SFT + GRPO at group sizes k=4,8,16. Each pulls its OWN distinct checkpoint dir (written by
    # the train notebook as runs/grpo_k<k>) so the three scores never use the same weights.
    # The base is the merged SFT policy (SFT_MERGED); train_task records which task each legacy
    # checkpoint was trained on, and is written into every report and the manifest.
    "ours_sft_grpo_k4": {"backend": "hf", "mode": "legacy", "model": SFT_MERGED, "family": "ours",
                          "harness": "agents+skills", "tool_access": "tools", "zero_shot": False,
                          "grpo_k": 4, "train_task": TASK},
    "ours_sft_grpo_k8": {"backend": "hf", "mode": "legacy", "model": SFT_MERGED, "family": "ours",
                          "harness": "agents+skills", "tool_access": "tools", "zero_shot": False,
                          "grpo_k": 8, "train_task": TASK},
    "ours_sft_grpo_k16": {"backend": "hf", "mode": "legacy", "model": SFT_MERGED, "family": "ours",
                           "harness": "agents+skills", "tool_access": "tools", "zero_shot": False,
                           "grpo_k": 16, "train_task": TASK},
    # Explicit F4 Z/H GRPO arms: same checkpoints (grpo_k) and base, but tools ONLY in H.
    "ours_sft_grpo_k4_z": {"backend": "hf", "mode": "Z", "model": SFT_MERGED, "family": "ours",
                            "harness": "none", "tool_access": "none", "zero_shot": False, "grpo_k": 4},
    "ours_sft_grpo_k4_h": {"backend": "hf", "mode": "H", "model": SFT_MERGED, "family": "ours",
                            "harness": "agents+skills", "tool_access": "tools", "zero_shot": False, "grpo_k": 4},
    "ours_sft_grpo_k8_z": {"backend": "hf", "mode": "Z", "model": SFT_MERGED, "family": "ours",
                            "harness": "none", "tool_access": "none", "zero_shot": False, "grpo_k": 8},
    "ours_sft_grpo_k8_h": {"backend": "hf", "mode": "H", "model": SFT_MERGED, "family": "ours",
                            "harness": "agents+skills", "tool_access": "tools", "zero_shot": False, "grpo_k": 8},
    "ours_sft_grpo_k16_z": {"backend": "hf", "mode": "Z", "model": SFT_MERGED, "family": "ours",
                             "harness": "none", "tool_access": "none", "zero_shot": False, "grpo_k": 16},
    "ours_sft_grpo_k16_h": {"backend": "hf", "mode": "H", "model": SFT_MERGED, "family": "ours",
                             "harness": "agents+skills", "tool_access": "tools", "zero_shot": False, "grpo_k": 16},
    # --- frontier (frozen replay files; no local weights) --------------------------------
    "frontier_zero": {"backend": "replay", "mode": "Z", "replay": str(WORK / "eval_all.jsonl"),
                      "family": "frontier", "harness": "none", "tool_access": "none", "zero_shot": True},
    "frontier_z": {"backend": "replay", "mode": "Z", "replay": str(WORK / "eval_all.jsonl"),
                   "family": "frontier", "harness": "none", "tool_access": "none", "zero_shot": True},
    # "tools" is the ORIGINAL (historical) arm label, NOT a valid H.MODES prompt mode. The replay
    # generator ignores messages, so the scored prompt is the stored one in the replay file
    # (already augmented by the teacher harness). model_eval normalizes this label to
    # agents+skills for augmentation, keeps "tools" as harness_label, and never passes it to H.augment.
    "frontier_tools": {"backend": "replay", "mode": "legacy",
                       "replay": str(WORK / "harness_eval_completions2.jsonl"),
                       "family": "frontier", "harness": "tools", "tool_access": "tools", "zero_shot": True},
    # Frontier in the LangGraph tool harness (the v6 tools journal, harness_completions.jsonl).
    "frontier_harness": {"backend": "replay", "mode": "legacy", "replay": str(WORK / "harness_completions.jsonl"),
                         "family": "frontier", "harness": "tools", "tool_access": "tools", "zero_shot": True},
    "frontier_h": {"backend": "replay", "mode": "H", "replay": str(WORK / "harness_completions.jsonl"),
                   "family": "frontier", "harness": "tools", "tool_access": "tools", "zero_shot": True},
    # --- other models (placeholders) ----------------------------------------------------
    # The measured domain-model arm is the saved GPT-nano answers, not a live localhost endpoint.
    "other_model": {"backend": "replay", "mode": "legacy", "replay": str(WORK / "eval_gptnano_replay.jsonl"),
                    "family": "domain-model", "harness": "none", "tool_access": "none", "zero_shot": True},
    "other_model_api": {"backend": "openai", "mode": "placeholder", "model": os.environ.get("API_MODEL", ""),
                        "base_url": os.environ.get("API_BASE_URL", "http://127.0.0.1:1234/v1"),
                        "family": "domain-model", "harness": "agents+skills", "tool_access": "none", "zero_shot": True},
}


def run_arm(name):
    spec = SPECS[name]
    out = REPORTS / f"{name}.json"
    if out.exists() and not FORCE:
        print(f"{name}: report exists ({out}); pass FORCE=1 to rerun")
        return out
    cmd = [sys.executable, "-m", "thermo_agent.eval.model_eval",
           *[str(d) for d in DATA_DIRS],
           "--backend", spec["backend"], "--harness", spec["harness"],
           "--limit", str(LIMIT), "--out", str(out), "--model-family", spec["family"],
           "--tool-access", spec["tool_access"], "--display-name", name]
    if spec["backend"] == "hf":
        # model_eval defaults to base Qwen; pass the spec's base explicitly so GRPO arms load the
        # merged SFT weights (SFT_MERGED) instead.
        cmd += ["--model", spec["model"]]
    if spec.get("zero_shot"):
        cmd.append("--zero-shot")
    # One split set for every arm: no arm (including frontier_tools) is restricted to test_id.
    if SPLITS:
        cmd += ["--splits", *SPLITS]
    adapter = spec.get("adapter")
    if spec.get("grpo_k") is not None:
        adapter = GRPO_K_ADAPTERS[spec["grpo_k"]]
    if spec.get("train_task"):
        require_adapter(adapter, f"grpo/{name}")
    if spec["backend"] == "hf" and adapter:
        cmd += ["--adapter", str(adapter), "--dtype", "float16"]
    if spec["backend"] == "replay":
        cmd += ["--replay", spec["replay"]]
    if spec["backend"] == "openai":
        cmd += ["--base-url", spec["base_url"]]
        if spec.get("model"):
            cmd += ["--api-model", spec["model"]]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out

# %%
# Run one arm at a time; set ARMS=frontier_zero to only do the replay arm in a CPU session.
written = {}
for name in ARMS:
    if name not in SPECS:
        raise SystemExit(f"unknown arm {name}; choose from {list(SPECS)}")
    written[name] = run_arm(name)
print("reports:", {k: str(v) for k, v in written.items()})

# %%
# Arm config table. Record which GRPO task produced each checkpoint INSIDE every report as well as
# the manifest, so a report can never be read as a different arm (or as base-model scoring).
def arm_config(name):
    spec = SPECS[name]
    return {"display_name": name, "family": spec["family"], "mode": spec.get("mode"),
            "harness": spec["harness"],
            "tool_access": spec["tool_access"], "zero_shot": spec["zero_shot"],
            "backend": spec["backend"], "model": spec.get("model"),
            "train_task": spec.get("train_task"), "grpo_k": spec.get("grpo_k"),
            "data_dirs": [str(d) for d in DATA_DIRS], "splits": SPLITS}


for name, out in written.items():
    if not out.exists():
        continue
    try:
        report = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    report["arm_config"] = arm_config(name)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

# %%
# Build the paper's comparison asset from whatever reports exist.
manifest = {"arms": {name: {"report": str(REPORTS / f"{name}.json"), **arm_config(name)}
                     for name, spec in SPECS.items() if (REPORTS / f"{name}.json").exists()}}
mpath = ROOT / "paper" / "assets" / "model_results_manifest.json"
mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
comparison.main([str(mpath), "--out", str(ROOT / "paper" / "assets" / "model_results.json")])

# %%
# Tool usage: how often the frontier model called the verifier, with reasoning and cost.
harness = pathlib.Path("/kaggle/working/harness_eval_completions2.jsonl")
if harness.exists():
    summary = tool_usage.summarize(tool_usage.load(harness))
    (REPORTS / "tool_usage.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))

# %%
# Collect a deterministic set of paper examples from every stage and redraw the trace figure.
cmd = [sys.executable, str(ROOT / "paper" / "tools" / "collect_examples.py"),
       "--datagen", str(ROOT / "data" / "processed"), str(ROOT / "data" / "processed_arch"),
       "--sft", str(ROOT / "data" / "sft"), "--frontier", str(ROOT / "data" / "frontier"),
       "--harness", str(harness) if harness.exists() else "",
       "--rollouts", *[str(p) for p in (ROOT / "runs").glob("grpo_*/rollouts.jsonl")],
       "--out", str(ROOT / "paper" / "assets" / "examples.json")]
cmd = [c for c in cmd if c]
subprocess.run(cmd, check=True)
fig = [sys.executable, str(ROOT / "paper" / "tools" / "figures_from_examples.py"),
       "--examples", str(ROOT / "paper" / "assets" / "examples.json"),
       "--out", str(ROOT / "paper" / "figures")]
if (REPORTS / "tool_usage.json").exists():
    fig += ["--tool-usage", str(REPORTS / "tool_usage.json")]
fig += ["--models", str(ROOT / "paper" / "assets" / "model_results.json")]
subprocess.run(fig, check=True)
print("done: download reports/, paper/assets/model_results.json and paper/figures/pdf/fig_trace_walkthrough.pdf")
