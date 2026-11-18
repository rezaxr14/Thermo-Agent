"""Shared setup for the generated Kaggle notebooks (train, teacher-with-tools, evaluation).

Keeping these few operations in one module means the three notebooks cannot drift apart in how they find the
uploaded datasets, install dependencies, or read the HF token. The teacher notebook stays self-contained because it
runs in the Kaggle Benchmarks environment without the code dataset.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

WORK = pathlib.Path("/kaggle/working")


def find_input(name: str) -> pathlib.Path:
    import os
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if pathlib.Path(root).name == name:
            return pathlib.Path(root)
    raise SystemExit(f"attach the Kaggle dataset {name!r}")


def setup(root=None, corpora=("sft", "grpo"), install=True):
    """Unpack the code bundle, expose it on sys.path, copy the corpora and set the safe training env."""
    root = pathlib.Path(root or WORK / "thermo-agent")
    import os
    code = find_input("thermo-agent-code")
    code_root, bundle = None, None
    for walk_root, dirs, files in os.walk(code, followlinks=True):
        if "thermo_agent" in dirs and (pathlib.Path(walk_root) / "thermo_agent" / "train" / "sft.py").exists():
            code_root = pathlib.Path(walk_root)
            break
        if "kaggle_bundle.zip" in files:
            bundle = pathlib.Path(walk_root) / "kaggle_bundle.zip"
    if not (root / "thermo_agent" / "train").exists():
        if code_root:
            shutil.copytree(code_root, root, dirs_exist_ok=True)
        elif bundle:
            with zipfile.ZipFile(bundle) as zf:
                zf.extractall(root.parent)
        else:
            raise SystemExit(f"no code at {root}: attach the kaggle_bundle.zip dataset")
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.update({"TOKENIZERS_PARALLELISM": "false", "NCCL_IB_DISABLE": "1", "HF_HUB_DISABLE_XET": "1"})
    corpus_input = find_input("thermo-agent-corpus")
    for name in corpora:
        dst = root / "data" / name
        if dst.exists():
            continue
        import os
        src = None
        for walk_root, _dirs, _files in os.walk(corpus_input, followlinks=True):
            if pathlib.Path(walk_root).name == name:
                src = pathlib.Path(walk_root)
                break
        if src and src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            continue
        archive = None
        for walk_root, _dirs, walk_files in os.walk(corpus_input, followlinks=True):
            if f"{name}.zip" in walk_files:
                archive = pathlib.Path(walk_root) / f"{name}.zip"
                break
        if archive:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dst)
        else:
            raise SystemExit(f"corpus {name} missing under {corpus_input}")
    if install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-lock.txt"], check=True)
    return root


def token() -> str:
    """HF token from a Kaggle Secret; empty string when the arm does not need one."""
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = value
        return value
    except Exception as exc:
        print("no Kaggle secret HF_TOKEN; HF-dependent arms will be skipped:", exc)
        return ""


def run(cmd, cwd=None, **env) -> None:
    """Print and run a subprocess, failing the cell on a non-zero exit code."""
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd, env={**os.environ, **{k: str(v) for k, v in env.items()}})


def write_json(path, obj) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(__import__("json").dumps(obj, indent=1) + "\n", encoding="utf-8")
    return p
