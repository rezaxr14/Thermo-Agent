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
# Arms: our base 4B, our SFT model with and without the inference text, the frontier model zero-shot and with our
# tools (replayed from Kaggle answers), and any other model served behind an OpenAI-compatible endpoint.

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

ROOT = setup()
# Pin check in the notebook itself; no source builds in any Kaggle worker.
assert_pinned_environment(ROOT / "kaggle" / "environment.lock.json")
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "30"))
FORCE = os.environ.get("FORCE", "0") == "1"
SPLITS = os.environ.get("SPLITS", "").split(",") if os.environ.get("SPLITS") else None
ARMS = os.environ.get("ARMS", "ours_base,ours_sft,ours_harness,ours_grpo,frontier_zero,frontier_tools,other_model").split(",")
find_input("thermo-teacher-prompts")     # fail early if the prompt dataset is missing

# %%
# Stage the saved teacher answers used by the replay arms. They come from a small private
# dataset; without it the frontier arms have nothing to score in the worker.
REPLAY_FILES = ("eval_all.jsonl", "harness_eval_completions2.jsonl",
                "eval_gptnano_replay.jsonl")
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
from thermo_agent.train import hf_sync
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

# %%
# One spec per arm. `replay` scores saved Kaggle answers with the same verifiers and prompt rows; the frontier
# export already carries its harness text, so zero-shot here means "no extra inference text beyond the prompt".
SPECS = {
    "ours_base": {"backend": "hf", "model": "Qwen/Qwen3.5-4B", "family": "ours", "harness": "none",
                  "tool_access": "none", "zero_shot": True},
    "ours_sft": {"backend": "hf", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER, "family": "ours",
                 "harness": "none", "tool_access": "none", "zero_shot": False},
    "ours_harness": {"backend": "hf", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER, "family": "ours",
                     "harness": "agents+skills", "tool_access": "none", "zero_shot": False},
    "ours_grpo": {"backend": "hf", "model": "Qwen/Qwen3.5-4B", "adapter": GRPO_ADAPTER, "family": "ours",
                  "harness": "agents+skills", "tool_access": "none", "zero_shot": False},
    "frontier_zero": {"backend": "replay", "replay": str(WORK / "eval_all.jsonl"), "family": "frontier",
                      "harness": "none", "tool_access": "none", "zero_shot": True},
    "frontier_tools": {"backend": "replay", "replay": str(WORK / "harness_eval_completions.jsonl"),
                       "family": "frontier", "harness": "tools", "tool_access": "tools", "zero_shot": True},
    # The measured domain-model arm is the saved GPT-nano answers, not a live localhost endpoint.
    "other_model": {"backend": "replay", "replay": str(WORK / "eval_gptnano_replay.jsonl"),
                    "family": "domain-model", "harness": "none", "tool_access": "none", "zero_shot": True},
    "other_model_api": {"backend": "openai", "model": os.environ.get("API_MODEL", ""),
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
           str(ROOT / "data" / "processed"), str(ROOT / "data" / "processed_arch"),
           "--backend", spec["backend"], "--harness", spec["harness"],
           "--limit", str(LIMIT), "--out", str(out), "--model-family", spec["family"],
           "--tool-access", spec["tool_access"], "--display-name", name]
    if spec.get("zero_shot"):
        cmd.append("--zero-shot")
    arm_splits = SPLITS or (["test_id"] if name == "frontier_tools" else None)
    if arm_splits:
        cmd += ["--splits", *arm_splits]
    if spec["backend"] == "hf" and spec.get("adapter"):
        cmd += ["--adapter", spec["adapter"], "--dtype", "float16"]
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
# Build the paper's comparison asset from whatever reports exist.
manifest = {"arms": {name: {"report": str(REPORTS / f"{name}.json"),
                            "family": spec["family"], "harness": spec["harness"],
                            "tool_access": spec["tool_access"], "zero_shot": spec["zero_shot"],
                            "display_name": name}
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
