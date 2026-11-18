"""Kaggle notebook source: build the Qwen3.5 fused-attention wheels ONCE.

Build with `python kaggle/make_wheel_notebook.py`. The resulting notebook compiles
`causal-conv1d` against the exact CUDA torch Kaggle ships, verifies the import, and leaves
the wheels in /kaggle/working/wheels. Download those wheels with `kaggle kernels output`
and upload them as the private dataset `rezanadimi/thermo-kernel-wheels`; every training
and eval notebook then installs from that dataset with `pip --no-index --find-links`, so no
worker ever recompiles CUDA code.
"""

# %% [markdown]
# # thermo-agent: one-time fused-kernel wheel build
#
# Qwen3.5-4B uses gated delta linear attention. Without `causal-conv1d` and
# `flash-linear-attention`, transformers falls back to a pure-PyTorch implementation that both
# OOMs a T4 at 4096 tokens and makes a training step take ~50 s. Building `causal-conv1d`
# from source costs several minutes of nvcc time, so we build it once here and publish the
# wheel as a private Kaggle dataset.
#
# Run on **GPU T4 x2** so the CUDA toolkit and torch match the training workers exactly.

# %%
import os
import pathlib
import subprocess
import sys

WORK = pathlib.Path("/kaggle/working")
WHEELS = WORK / "wheels"
WHEELS.mkdir(parents=True, exist_ok=True)

import torch
print("[wheel] torch", torch.__version__, "cuda", torch.version.cuda, "python", sys.version.split()[0], flush=True)
assert torch.cuda.is_available(), "build on a GPU worker so nvcc and CUDA torch match training"

# %%
# Build tools. `ninja` is what causal-conv1d's setup.py uses to drive nvcc.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja", "packaging", "wheel"], check=True)

# %%
# Build wheels against the *installed* CUDA torch, without build isolation (which would pull a
# CPU torch from PyPI). fla-core and flash-linear-attention are pure-Python wheels;
# causal-conv1d is the one that needs nvcc.
env = os.environ.copy()
env.update({
    "CUDA_HOME": "/usr/local/cuda",
    "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
    "TORCH_CUDA_ARCH_LIST": "7.5",  # T4 only: smaller compile, faster build
    "MAX_JOBS": "4",
})
packages = ["fla-core==0.6.0", "flash-linear-attention==0.6.0", "causal-conv1d==1.7.0"]
subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                "-w", str(WHEELS), *packages], check=True, env=env)

# %%
# Verify the freshly built wheel imports and that CUDA torch survived.
installed = sorted(str(p) for p in WHEELS.glob("*.whl"))
print("[wheel] built:", installed, flush=True)
assert any("causal_conv1d" in p for p in installed), "causal-conv1d wheel missing"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *installed], check=True)

import torch as verify_torch
import causal_conv1d
import fla
print("[wheel] causal_conv1d", causal_conv1d.__version__, "| fla", getattr(fla, "__version__", "installed"), flush=True)
print("[wheel] torch after install", verify_torch.__version__, "cuda", verify_torch.version.cuda,
      "devices", verify_torch.cuda.device_count(), flush=True)
assert verify_torch.cuda.is_available() and "+cu" in verify_torch.__version__, "CUDA torch was replaced"

# %%
# Manifest for the dataset upload. The wheels themselves are the output files.
import hashlib
hashes = {}
for wheel in sorted(WHEELS.glob("*.whl")):
    hashes[wheel.name] = hashlib.sha256(wheel.read_bytes()).hexdigest()
manifest = {
    "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
    "packages": packages, "wheels": [p.name for p in sorted(WHEELS.glob("*.whl"))],
    "sha256": hashes,
}
(WHEELS / "manifest.json").write_text(__import__("json").dumps(manifest, indent=1) + "\n", encoding="utf-8")
print("[wheel] DONE", manifest, flush=True)
