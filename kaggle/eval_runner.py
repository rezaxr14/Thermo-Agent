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
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

WORK = pathlib.Path("/kaggle/working")
ROOT = WORK / "thermo-agent"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "40"))
FORCE = os.environ.get("FORCE", "0") == "1"
SPLITS = os.environ.get("SPLITS", "").split(",") if os.environ.get("SPLITS") else None
ARMS = os.environ.get("ARMS", "ours_base,ours_harness,frontier_zero,frontier_tools,other_model").split(",")

# %%
def _input_dir(name):
    hits = sorted(p for p in pathlib.Path("/kaggle/input").glob(f"**/{name}") if p.is_dir())
    if not hits:
        raise SystemExit(f"attach the Kaggle dataset {name!r}")
    return hits[0]


code = _input_dir("thermo-agent-code")
bundle = next(code.rglob("kaggle_bundle.zip"), None)
if bundle:
    shutil.rmtree(ROOT, ignore_errors=True)
    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(WORK)
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-lock.txt"], check=True)
corpora = _input_dir("thermo-agent-corpus")
for name in ("sft", "grpo"):
    src = next(corpora.rglob(name), None)
    if src and not (ROOT / "data" / name).exists():
        shutil.copytree(src, ROOT / "data" / name)
prompts = _input_dir("thermo-teacher-prompts")
print("code, corpus and prompts ready")

# %%
from thermo_agent.train import hf_sync
from thermo_agent.eval import comparison, tool_usage

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    print("no HF_TOKEN secret; trained-model arms need local weights:", exc)

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

# %%
# One spec per arm. `replay` scores saved Kaggle answers with the same verifiers and prompt rows; the frontier
# export already carries its harness text, so zero-shot here means "no extra inference text beyond the prompt".
SPECS = {
    "ours_base": {"backend": "hf", "model": "Qwen/Qwen3.5-4B", "family": "ours", "harness": "none",
                  "tool_access": "none", "zero_shot": True},
    "ours_harness": {"backend": "hf", "model": "Qwen/Qwen3.5-4B", "adapter": ADAPTER, "family": "ours",
                     "harness": "agents+skills", "tool_access": "none", "zero_shot": False},
    "frontier_zero": {"backend": "replay", "replay": str(WORK / "completions.jsonl"), "family": "frontier",
                      "harness": "none", "tool_access": "none", "zero_shot": True},
    "frontier_tools": {"backend": "replay", "replay": str(WORK / "harness_completions.jsonl"), "family": "frontier",
                       "harness": "tools", "tool_access": "tools", "zero_shot": True},
    "other_model": {"backend": "openai", "model": os.environ.get("API_MODEL", ""),
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
    if SPLITS:
        cmd += ["--splits", *SPLITS]
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
harness = pathlib.Path("/kaggle/working/harness_completions.jsonl")
if harness.exists():
    summary = tool_usage.summarize(tool_usage.load(harness))
    (REPORTS / "tool_usage.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))

# %%
# Collect a deterministic set of paper examples from every stage and redraw the trace figure.
cmd = [sys.executable, str(ROOT / "paper" / "tools" / "collect_examples.py"),
       "--datagen", str(ROOT / "data" / "processed"), str(ROOT / "data" / "processed_arch"),
       "--sft", str(ROOT / "data" / "sft"), "--frontier", str(ROOT / "data" / "frontier"),
       "--harness", str(harness) if harness.exists(),
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
