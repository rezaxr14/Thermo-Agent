"""Build kaggle/wheel_kernel/eval.ipynb from kaggle/wheel_runner.py."""
from __future__ import annotations

import json
import pathlib

from make_notebook import build

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "wheel_kernel"
OUT.mkdir(parents=True, exist_ok=True)
nb = build(HERE / "wheel_runner.py", OUT / "wheel_build.ipynb")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "rezanadimi/thermo-kernel-wheels",
    "title": "thermo-kernel-wheels",
    "code_file": "wheel_build.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}, indent=2) + "\n", encoding="utf-8")
print(nb)
