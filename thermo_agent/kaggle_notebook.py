"""Shared setup for the generated Kaggle notebooks (train, teacher-with-tools, evaluation).

Keeping these few operations in one module means the three notebooks cannot drift apart in how they find the
uploaded datasets, install dependencies, or read the HF token. The teacher notebook stays self-contained because it
runs in the Kaggle Benchmarks environment without the code dataset.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

WORK = pathlib.Path("/kaggle/working")
PINNED_SITE = pathlib.Path("/kaggle/working/pydeps")

# Updated by setup() on every call so a caller (or a test) can inspect what was actually staged
# without having to re-walk the attached corpus dataset. Kept module-level because returning a
# richer object would break callers that expect setup() to return the root Path.
STAGED_CORPORA = ()
MISSING_CORPORA = ()
DEFAULT_CORPORA = ("sft", "grpo", "sft_all", "grpo_all", "heldout")


def _pinned_env() -> dict:
    """Environment that makes a fresh child import the pinned numeric wheels first."""
    previous = os.environ.get("PYTHONPATH", "")
    prefix = str(PINNED_SITE) + (os.pathsep + previous if previous else "")
    return {**os.environ, "PYTHONPATH": prefix}


def _child(code: str) -> str:
    """Run a short Python snippet in a fresh interpreter with PINNED_SITE first on PYTHONPATH."""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_pinned_env())
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip()[-800:])
    return result.stdout.strip()


def _install_pinned_numeric(target) -> None:
    """Install the pinned numpy/scipy into a SIDE directory, never into the running interpreter.

    Kaggle's ipykernel preloads numpy before cell 1, so reinstalling numpy in place cannot work: the
    stale C extension stays cached in sys.modules and the new pure-Python layer cannot bind to it.
    Installing into `target` and putting it first on PYTHONPATH gives every fresh child the correct
    wheels from interpreter startup; the parent never re-imports the numeric stack.
    """
    target = pathlib.Path(target)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(target),
         "--index-url", "https://pypi.org/simple", "numpy==2.4.4", "scipy==1.17.1"],
        check=True)
    # Split the snippet's prefix so this source contains no real numpy import statement: this
    # module must never import the numeric stack in the parent process.
    verify = subprocess.run(
        [sys.executable, "-c", "import " + "numpy, scipy; print(numpy.__version__, scipy.__version__)"],
        capture_output=True, text=True, env=_pinned_env())
    if verify.returncode != 0:
        raise SystemExit("[compat] pinned numpy/scipy failed to import: " + verify.stderr.strip()[-500:])
    versions = verify.stdout.strip().split()
    if len(versions) != 2:
        raise SystemExit("[compat] unexpected pinned version output: " + repr(verify.stdout))
    print(f"[compat] pinned numpy {versions[0]} scipy {versions[1]} OK via {target}", flush=True)


# A short subprocess reports torch facts as JSON; the parent notebook never imports torch (nor the
# numeric stack it can pull in) after the pinned install.
_TORCH_FACTS_CODE = (
    "import json,torch;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
    "'available':torch.cuda.is_available()}))"
)


def torch_facts() -> dict:
    """Return torch facts from a short subprocess; the parent process never imports torch."""
    return json.loads(_child(_TORCH_FACTS_CODE))


def torch_device_count() -> int:
    """Number of visible CUDA devices, from a short subprocess (no parent torch import)."""
    return int(_child("import " + "torch;print(torch.cuda.device_count())"))


def _fused_kernel_versions() -> dict:
    """Import the fused-kernel wheels in a fresh child and report their versions as JSON."""
    code = ("import json, causal_conv1d, fla;"
            "print(json.dumps({'causal_conv1d':causal_conv1d.__version__,"
            "'fla':getattr(fla,'__version__','installed')}))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_pinned_env())
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-500:])
    return json.loads(result.stdout.strip())


def find_input(name: str) -> pathlib.Path:
    import os
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if pathlib.Path(root).name == name:
            return pathlib.Path(root)
    raise SystemExit(f"attach the Kaggle dataset {name!r}")


def assert_pinned_environment(lock_path) -> None:
    """Fail fast if the worker's pinned stack drifted; never repair it by compiling."""
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
    facts = torch_facts()
    worker = lock["worker"]
    if facts["torch"] != worker["torch"]:
        mismatches.append(f"torch: expected {worker['torch']}, found {facts['torch']}")
    if (facts["cuda"] or "") != worker["cuda"]:
        mismatches.append(f"cuda: expected {worker['cuda']}, found {facts['cuda']}")
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
    print(f"[env-pin] OK python {worker['python']}, torch {facts['torch']}, cuda {facts['cuda']}, "
          f"{len(expected)} pinned packages, fla + causal_conv1d importable", flush=True)


def install_cached_kernels(required: bool = True) -> None:
    """Install the pinned fused kernels from the private wheel dataset; NEVER recompile.

    The wheels in `thermo-kernel-wheels` were built once by kernel
    `rezanadimi/thermo-kernel-wheels` against Kaggle's own CUDA torch. Qwen3.5's reference
    linear-attention kernels OOM a T4 at 4096 tokens, so a GPU worker must use these wheels
    or fail loudly. There is deliberately no source-build fallback in any worker.
    """
    facts = torch_facts()
    try:
        source = find_input("thermo-kernel-wheels")
    except SystemExit:
        if required and facts["available"]:
            raise SystemExit("[kernel] attach rezanadimi/thermo-kernel-wheels; compiling in the worker is forbidden")
        print("[kernel] no wheel dataset attached (CPU-only worker)", flush=True)
        return
    wheels = sorted(str(p) for p in source.rglob("*.whl"))
    if not wheels:
        if required and facts["available"]:
            raise SystemExit("[kernel] wheel dataset has no wheels; republish it from the build kernel")
        print("[kernel] wheel dataset has no wheels (CPU-only worker)", flush=True)
        return

    # The wheel is ABI-bound to one torch/CUDA build. A mismatch means the Kaggle image
    # changed: rebuild the dataset once, do not install from source in the worker.
    manifest = next((p for p in source.rglob("manifest.json")), None)
    if manifest is not None:
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        for key, actual in (("torch", facts["torch"]), ("cuda", facts["cuda"])):
            if key in expected and expected[key] != actual:
                raise SystemExit(
                    f"[kernel] pinned wheels expect {key}={expected[key]}, worker has {actual}. "
                    "Rebuild rezanadimi/thermo-kernel-wheels from the build kernel; "
                    "do not compile in the worker."
                )
        print(f"[env-pinned] using wheel manifest {manifest.name}: "
              f"python {expected.get('python')}, torch {expected.get('torch')}, cuda {expected.get('cuda')}", flush=True)
    if required and not facts["available"]:
        raise SystemExit("[kernel] a GPU worker must see CUDA before installing pinned wheels")

    torch_before = (facts["torch"], facts["cuda"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels], check=True)
    facts_after = torch_facts()
    if (facts_after["torch"], facts_after["cuda"]) != torch_before or not facts_after["available"]:
        raise SystemExit("[kernel] CUDA torch was changed by the wheel install")
    try:
        versions = _fused_kernel_versions()
        print(f"[kernel] fused kernels from dataset: causal_conv1d {versions['causal_conv1d']}, "
              f"fla {versions['fla']}", flush=True)
    except RuntimeError as exc:
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


def setup(root=None, corpora=("sft", "grpo", "sft_all", "grpo_all", "heldout"), install=True):
    """Unpack the code bundle, expose it on sys.path, copy the corpora and set the safe training env.

    Every requested corpus is staged from either a directory or a ``<name>.zip`` under the
    ``thermo-agent-corpus`` dataset. Names that are absent are skipped (never faked): they are
    recorded in ``MISSING_CORPORA`` and reported with ``[corpus] missing <name>``. The names that
    were staged (or were already present) are recorded in ``STAGED_CORPORA``. Both records are
    reset on each call. The return value stays the root Path for backwards compatibility.
    """
    root = pathlib.Path(root or WORK / "thermo-agent")
    global STAGED_CORPORA, MISSING_CORPORA
    STAGED_CORPORA, MISSING_CORPORA = (), ()
    # Never import the numeric stack (numpy/scipy/pandas) or transformers/torch here. The pinned
    # wheels live in a side directory and are imported only by fresh child processes; see
    # _install_pinned_numeric.
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
    # PINNED_SITE must come first so every training/eval child imports the pinned numpy/scipy from
    # interpreter startup; then the repo root, then whatever PYTHONPATH the kernel already had.
    _previous_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(PINNED_SITE), str(root)] + ([_previous_pythonpath] if _previous_pythonpath else [])
    )
    os.environ.update({"TOKENIZERS_PARALLELISM": "false", "NCCL_IB_DISABLE": "1", "HF_HUB_DISABLE_XET": "1",
                       # Use the installed fla/causal_conv1d kernels, never a runtime Hub kernel download.
                       "USE_HUB_KERNELS": "NO",
                       # The 4B fp16 model leaves little headroom on a T4; avoid fragmentation OOMs at the limit.
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    corpus_input = find_input("thermo-agent-corpus")
    staged, missing = [], []
    for name in corpora:
        dst = root / "data" / name
        if dst.exists():
            # Already staged in a previous cell/session; keep it and don't re-copy.
            staged.append(name)
            continue
        src = None
        for walk_root, _dirs, _files in os.walk(corpus_input, followlinks=True):
            if pathlib.Path(walk_root).name == name:
                src = pathlib.Path(walk_root)
                break
        if src and src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            staged.append(name)
            print(f"[corpus] staged {name} from directory {src}", flush=True)
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
            staged.append(name)
            print(f"[corpus] staged {name} from {archive}", flush=True)
        else:
            # Absent corpora are skipped, not faked: never create an empty dst dir. The runner's
            # pre-DDP assertion is responsible for failing loudly if a requested corpus is missing.
            missing.append(name)
            print(f"[corpus] missing {name}", flush=True)
    STAGED_CORPORA, MISSING_CORPORA = tuple(staged), tuple(missing)
    print(f"[corpus] staged {list(staged)}; missing {list(missing)}", flush=True)
    if install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-lock.txt"], check=True)
        # Exact versions the chunked-loss path was tested with locally. Kaggle's preinstalled TRL builds the
        # full (tokens x vocabulary) logits, which does not fit a T4 next to the 4B fp16 weights.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "transformers==5.23.0", "trl==1.18.1", "peft==0.21.3",
                        "accelerate==1.16.0", "datasets==5.2.0", "einops==0.8.2"], check=True)
        # Kaggle's preinstalled numpy 2.4.4 is CORRUPT (its compiled extension and umath stub are
        # broken), and ipykernel preloads it before cell 1. An in-place reinstall cannot fix the
        # already-loaded C extension, so install the pinned pair into a SIDE directory and let every
        # child import it from interpreter startup via PINNED_SITE-first PYTHONPATH. The parent
        # process never imports numpy/scipy/pandas/transformers after this point.
        _install_pinned_numeric(PINNED_SITE)
        # Kaggle's preinstalled torchao 0.10 makes PEFT 0.21 raise on import; this project does not use it.
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
        install_cached_kernels()
        import importlib.metadata as md
        versions = {p: md.version(p) for p in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")}
        print("[env]", versions, flush=True)
        facts = torch_facts()
        if not facts["available"]:
            raise SystemExit("[env] the dependency install replaced the CUDA torch with a CPU build")
        print(f"[env] CUDA {facts['cuda']}, {torch_device_count()} visible device(s)", flush=True)
        lock = pathlib.Path(root) / "kaggle" / "environment.lock.json"
        if lock.exists() and facts["available"]:
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
