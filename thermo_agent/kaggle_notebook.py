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


def assert_pinned_environment(lock_path) -> None:
    """Fail fast if the worker's pinned stack drifted; never repair it by compiling."""
    import json
    from importlib import metadata

    lock = json.loads(pathlib.Path(lock_path).read_text(encoding="utf-8"))
    expected = lock["python_packages"]
    mismatches = []
    for package, version in expected.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "MISSING"
        if actual != version:
            mismatches.append(f"{package}: expected {version}, found {actual}")
    import torch
    worker = lock["worker"]
    if torch.__version__ != worker["torch"]:
        mismatches.append(f"torch: expected {worker['torch']}, found {torch.__version__}")
    if (torch.version.cuda or "") != worker["cuda"]:
        mismatches.append(f"cuda: expected {worker['cuda']}, found {torch.version.cuda}")
    if mismatches:
        raise SystemExit(
            "[env-pin] FAILED. " + "; ".join(mismatches) + ". "
            "Do NOT install or compile anything in the worker: re-run only the Kaggle build kernel "
            "rezanadimi/thermo-kernel-wheels if the Kaggle image really changed, then update "
            "kaggle/environment.lock.json."
        )
    for name in ("fla", "causal_conv1d"):
        if not __import__("importlib.util", fromlist=["find_spec"]).find_spec(name):
            raise SystemExit(f"[env-pin] fused kernel {name} missing; wheels must come from kernel_sources")
    print(f"[env-pin] OK python {worker['python']}, torch {worker['torch']}, cuda {worker['cuda']}, "
          f"{len(expected)} pinned packages, fla + causal_conv1d importable", flush=True)


def install_cached_kernels(required: bool = True) -> None:
    """Install the pinned fused kernels from the private wheel dataset; NEVER recompile.

    The wheels in `thermo-kernel-wheels` were built once by kernel
    `rezanadimi/thermo-kernel-wheels` against Kaggle's own CUDA torch. Qwen3.5's reference
    linear-attention kernels OOM a T4 at 4096 tokens, so a GPU worker must use these wheels
    or fail loudly. There is deliberately no source-build fallback in any worker.
    """
    import torch
    try:
        source = find_input("thermo-kernel-wheels")
    except SystemExit:
        if required and torch.cuda.is_available():
            raise SystemExit("[kernel] attach rezanadimi/thermo-kernel-wheels; compiling in the worker is forbidden")
        print("[kernel] no wheel dataset attached (CPU-only worker)", flush=True)
        return
    wheels = sorted(str(p) for p in source.rglob("*.whl"))
    if not wheels:
        if required and torch.cuda.is_available():
            raise SystemExit("[kernel] wheel dataset has no wheels; republish it from the build kernel")
        print("[kernel] wheel dataset has no wheels (CPU-only worker)", flush=True)
        return

    # The wheel is ABI-bound to one torch/CUDA build. A mismatch means the Kaggle image
    # changed: rebuild the dataset once, do not install from source in the worker.
    manifest = next((p for p in source.rglob("manifest.json")), None)
    if manifest is not None:
        import json
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        for key, actual in (("torch", torch.__version__), ("cuda", torch.version.cuda)):
            if key in expected and expected[key] != actual:
                raise SystemExit(
                    f"[kernel] pinned wheels expect {key}={expected[key]}, worker has {actual}. "
                    "Rebuild rezanadimi/thermo-kernel-wheels from the build kernel; "
                    "do not compile in the worker."
                )
        print(f"[env-pinned] using wheel manifest {manifest.name}: "
              f"python {expected.get('python')}, torch {expected.get('torch')}, cuda {expected.get('cuda')}", flush=True)
    if required and not torch.cuda.is_available():
        raise SystemExit("[kernel] a GPU worker must see CUDA before installing pinned wheels")

    torch_before = (torch.__version__, torch.version.cuda)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels], check=True)
    if (torch.__version__, torch.version.cuda) != torch_before or not torch.cuda.is_available():
        raise SystemExit("[kernel] CUDA torch was changed by the wheel install")
    try:
        import causal_conv1d
        import fla
        print(f"[kernel] fused kernels from dataset: causal_conv1d {causal_conv1d.__version__}, "
              f"fla {getattr(fla, '__version__', 'installed')}", flush=True)
    except Exception as exc:
        raise SystemExit(f"[kernel] pinned wheels failed to import ({exc}); do not fall back to reference kernels")


def restore_checkpoints(root, stages=("sft", "grpo_repair", "grpo_design", "grpo_audit", "grpo_ledger"), source=None):
    """Restore the newest checkpoint of each stage from the private resume dataset.

    Kaggle Secrets are sometimes unreachable inside a worker, so HF sync cannot be relied on
    for cross-session resume. Between sessions a local CLI step copies the newest checkpoint
    from the previous kernel output into `rezanadimi/thermo-agent-resume`; this helper mounts
    it back into `runs/<stage>/` before the trainer asks for `latest(out_dir)`.
    No credentials are involved: the dataset holds only LoRA weights, optimizer, scheduler,
    scaler and RNG state.
    """
    import json
    root = pathlib.Path(root)
    if source is None:
        try:
            source = find_input("thermo-agent-resume")
        except SystemExit:
            print("[resume] no checkpoint dataset attached; fresh run", flush=True)
            return {}
    source = pathlib.Path(source)
    restored = {}
    for stage in stages:
        base = None
        for walk_root, dirs, _files in os.walk(source, followlinks=True):
            if pathlib.Path(walk_root).name == stage and any(d.startswith("checkpoint-") for d in dirs):
                base = pathlib.Path(walk_root)
                break
        if base is None:
            continue
        candidates = [p for p in base.iterdir()
                      if p.is_dir() and p.name.startswith("checkpoint-") and p.name[11:].isdigit()]
        if not candidates:
            continue
        best, state = None, None
        for candidate in sorted(candidates, key=lambda p: int(p.name[11:]), reverse=True):
            try:
                candidate_state = json.loads((candidate / "trainer_state.json").read_text(encoding="utf-8"))
                has_adapter = any((candidate / name).exists()
                                  for name in ("adapter_model.safetensors", "adapter_model.bin"))
                has_optimizer = (candidate / "optimizer.pt").exists()
                if not (candidate_state.get("global_step", 0) > 0 and has_adapter and has_optimizer):
                    raise ValueError(f"incomplete checkpoint {candidate}")
            except (OSError, ValueError) as exc:
                print(f"[resume] skipping {candidate}: {exc}", flush=True)
                continue
            best, state = candidate, candidate_state
            break
        if best is None:
            continue
        destination = root / "runs" / stage / best.name
        if destination.exists():
            print(f"[resume] {stage}: {best.name} already local", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(best, destination)
        restored[stage] = str(destination)
        print(f"[resume] {stage}: restored {best.name} (step {state['global_step']})", flush=True)
    return restored


def setup(root=None, corpora=("sft", "grpo"), install=True):
    """Unpack the code bundle, expose it on sys.path, copy the corpora and set the safe training env."""
    root = pathlib.Path(root or WORK / "thermo-agent")
    import os
    # Do NOT import numpy/torch here: Kaggle's numpy is replaced on disk below, and importing it
    # first would leave the corrupt module cached in this process. install_cached_kernels imports
    # torch/fla only after the verified reinstall.
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
    os.environ.update({"TOKENIZERS_PARALLELISM": "false", "NCCL_IB_DISABLE": "1", "HF_HUB_DISABLE_XET": "1",
                       # Use the installed fla/causal_conv1d kernels, never a runtime Hub kernel download.
                       "USE_HUB_KERNELS": "NO",
                       # The 4B fp16 model leaves little headroom on a T4; avoid fragmentation OOMs at the limit.
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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
        # Exact versions the chunked-loss path was tested with locally. Kaggle's preinstalled TRL builds the
        # full (tokens x vocabulary) logits, which does not fit a T4 next to the 4B fp16 weights.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "transformers==5.23.0", "trl==1.18.1", "peft==0.21.3",
                        "accelerate==1.16.0", "datasets==5.2.0", "einops==0.8.2"], check=True)
        # Kaggle's preinstalled numpy 2.4.4 is CORRUPT: its compiled extension misses
        # _blas_supports_fpe AND its numpy/_core/umath.py is a stub missing _slice, so any
        # scipy/transformers/fla import fails. pip's mirror keeps serving the same broken wheel
        # and "already satisfied". Force a consistent numpy+scipy pair from PyPI proper (pure
        # C wheels, no CUDA recompile) and verify in a clean subprocess before continuing.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                        "--index-url", "https://pypi.org/simple",
                        "numpy==2.4.4", "scipy==1.17.1"], check=True)
        verify = subprocess.run(
            [sys.executable, "-c",
             "import numpy, scipy.linalg; from scipy.optimize import linear_sum_assignment; "
             "import numpy._core._multiarray_umath as m; assert hasattr(m, '_blas_supports_fpe'); "
             "print('[compat] numpy', numpy.__version__, 'scipy', scipy.__version__, 'import OK')"],
            capture_output=True, text=True)
        print(verify.stdout.strip(), flush=True)
        if verify.returncode != 0:
            raise SystemExit("[compat] numpy/scipy still broken after PyPI reinstall: " + verify.stderr.strip()[-500:])
        # Kaggle's preinstalled torchao 0.10 makes PEFT 0.21 raise on import; this project does not use it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
        install_cached_kernels()
        import importlib.metadata as md
        versions = {p: md.version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")}
        print("[env]", versions, flush=True)
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("[env] the dependency install replaced the CUDA torch with a CPU build")
        print(f"[env] CUDA {torch.version.cuda}, {torch.cuda.device_count()} visible device(s)", flush=True)
        lock = pathlib.Path(root) / "kaggle" / "environment.lock.json"
        if lock.exists() and torch.cuda.is_available():
            assert_pinned_environment(lock)
        elif not lock.exists():
            print("[env-pin] lock file missing in code bundle; pin not verified", flush=True)
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
    """Print and run a subprocess, teeing every line to `/kaggle/working/thermo_run.log`.

    Kaggle's session log endpoint returns only nbconvert warnings, not the notebook's
    stdout, so the file is the only way to read a run's output afterwards:
    `kaggle kernels output <slug> -p OUT` downloads it.
    """
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    log = WORK / "thermo_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("+ " + " ".join(cmd) + "\n")
        fh.flush()
        process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, **{k: str(v) for k, v in env.items()}},
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            fh.write(line)
            fh.flush()
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def status(**fields) -> pathlib.Path:
    """Merge a few fields into `/kaggle/working/status.json`, the run's progress record."""
    import datetime
    import json
    path = WORK / "status.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data.update(fields)
    data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print("[status]", fields, flush=True)
    return path


def write_json(path, obj) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(__import__("json").dumps(obj, indent=1) + "\n", encoding="utf-8")
    return p
