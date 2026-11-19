"""Upload/download ONLY this project's Kaggle datasets, under ``data/kaggle_datasets/``.

Hard rule: the dataset id must be one of ALLOWED_DATASETS. A foreign dataset (including another
owner's copy of the same slug, or the adjacent ``thermo-kernel-wheels`` dataset) is refused before any
network call. Nothing here creates, deletes or modifies any other dataset or model repo.

All Kaggle CLI traffic from this machine goes through the local SOCKS proxy. The proxy is scoped to the
kaggle subprocess only -- os.environ is never mutated, so no non-Kaggle call is ever proxied. In PowerShell
the equivalent scoping is:

    $env:HTTP_PROXY='socks5h://localhost:10808'; $env:HTTPS_PROXY='socks5h://localhost:10808'
    .venv/Scripts/kaggle.exe datasets <...>
    Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY -ErrorAction SilentlyContinue

(KAGGLE_API_TOKEN is read silently from the environment or .env and passed only to the kaggle process.)

    python scripts/sync_kaggle_datasets.py upload   --dataset thermo-agent-code --dry-run
    python scripts/sync_kaggle_datasets.py download --dataset thermo-agent-corpus --dry-run
    python scripts/sync_kaggle_datasets.py upload   --datasets thermo-agent-code thermo-agent-corpus \
        --message "code+corpus rebuild" --tag 2026-11-19
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

OWNER = "rezanadimi"
# Hard allow-list of this project's datasets only. Never expand this from a CLI arg or env var.
ALLOWED_DATASETS = frozenset({
    "thermo-agent-code",
    "thermo-agent-corpus",
    "thermo-agent-resume",
    "thermo-teacher-prompts",
    "thermo-teacher-prompts-big",
    "thermo-agent-eval-replays",
})
PROXY = "socks5h://localhost:10808"
DATA_ROOT = Path("data/kaggle_datasets")
# Legacy local folder name for one slug. This is a directory alias, NOT an allow-list expansion.
DIR_ALIASES = {"thermo-teacher-prompts-big": "teacher_big_prompts"}


# --------------------------------------------------------------------------------------------- pure helpers
def resolve_dataset(value: str) -> str:
    """Return ``rezanadimi/<slug>`` for an allow-listed id; refuse everything else, before any call."""
    raw = (value or "").strip()
    if "/" in raw:
        owner, slug = raw.split("/", 1)
        if owner != OWNER:
            raise ValueError(f"refusing Kaggle dataset {raw!r}: only {OWNER!r} datasets are allowed")
    else:
        slug = raw
    if slug not in ALLOWED_DATASETS:
        raise ValueError(f"refusing Kaggle dataset {value!r}: not in this project's dataset allow-list")
    return f"{OWNER}/{slug}"


def dataset_dir(full_id: str) -> Path:
    """Resolve the local directory for ``rezanadimi/<slug>``; refuse anything outside DATA_ROOT."""
    full_id = resolve_dataset(full_id)  # re-check the allow-list even for direct callers
    slug = full_id.split("/", 1)[1]
    candidate = DATA_ROOT / DIR_ALIASES.get(slug, slug)
    root = DATA_ROOT.resolve()
    target = candidate.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"refusing dataset path {candidate!r}: outside {DATA_ROOT.as_posix()!r}")
    return candidate


def read_kaggle_token() -> str | None:
    """KAGGLE_API_TOKEN from env or .env; value is never printed. None is allowed in dry-run."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "KAGGLE_API_TOKEN":
                token = value.strip().strip('"').strip("'")
                if token:
                    return token
    return None


def kaggle_bin() -> str:
    return os.environ.get("KAGGLE_BIN") or (".venv/Scripts/kaggle.exe" if os.name == "nt" else "kaggle")


def kaggle_env(token: str | None = None) -> dict:
    """Env for the kaggle subprocess ONLY. Never mutate os.environ: non-Kaggle calls stay unproxied."""
    env = os.environ.copy()
    env["HTTP_PROXY"] = PROXY
    env["HTTPS_PROXY"] = PROXY
    env["http_proxy"] = PROXY
    env["https_proxy"] = PROXY
    env["PYTHONIOENCODING"] = "utf-8"
    if token:
        env["KAGGLE_API_TOKEN"] = token
    return env


def _quote(arg: str) -> str:
    return arg if all(c not in arg for c in " \t\"'") else f'"{arg}"'


def describe_command(cmd: list[str]) -> str:
    """Human-readable form showing the proxy is scoped to this command (PowerShell syntax)."""
    return (f"$env:HTTP_PROXY='{PROXY}'; $env:HTTPS_PROXY='{PROXY}'; "
            + " ".join(_quote(c) for c in cmd))


def build_command(action: str, dataset: str, directory: str, message: str,
                  overwrite: bool = False, create: bool = False) -> list[str]:
    """Build the exact kaggle CLI argv. Uploads are additive versions, never deletes."""
    k = kaggle_bin()
    if action == "upload":
        if create:
            return [k, "datasets", "create", "-p", directory, "-m", message]
        # Flags copied from docs/RUNBOOK.md (the command used for this project's code dataset).
        return [k, "datasets", "version", "-p", directory, "-m", message, "-t", "-r", "skip"]
    if action == "download":
        cmd = [k, "datasets", "download", dataset, "-p", directory, "--unzip"]
        return cmd + (["-w"] if overwrite else [])
    raise ValueError(f"unknown action {action!r}")


def plan(action: str, dataset: str, message: str = "refresh thermo-agent dataset", tag: str = "",
         overwrite: bool = False, create: bool = False) -> dict:
    """Pure planning helper used by both the CLI and the tests. No network, no token read."""
    full = resolve_dataset(dataset)
    directory = dataset_dir(full)
    if not directory.is_dir():
        raise SystemExit(f"dataset directory {directory.as_posix()!r} does not exist locally")
    note = f"{message} [tag {tag}]" if tag else message
    cmd = build_command(action, full, str(directory), note, overwrite=overwrite, create=create)
    return {"dataset": full, "slug": full.split("/", 1)[1], "dir": directory.as_posix(),
            "command": cmd, "display": describe_command(cmd)}


def run_plan(item: dict, dry_run: bool) -> int:
    if dry_run:
        print(f"[dry-run] {item['display']}")
        print("[dry-run] proxy scoped to the kaggle command; no non-Kaggle call is proxied; nothing was sent")
        return 0
    token = read_kaggle_token()
    if not token:
        raise SystemExit("no KAGGLE_API_TOKEN in the environment or .env (value is never printed)")
    result = subprocess.run(item["command"], env=kaggle_env(token), check=False)
    return int(result.returncode)


# --------------------------------------------------------------------------------------------- command line
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("upload", "download"))
    ap.add_argument("--dataset", help="one allow-listed slug or rezanadimi/<slug>")
    ap.add_argument("--datasets", nargs="+", help="several allow-listed datasets (default: all of them)")
    ap.add_argument("--message", default="refresh thermo-agent dataset")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the scoped command; no network")
    ap.add_argument("--overwrite", action="store_true", help="download only: pass -w to kaggle")
    ap.add_argument("--create", action="store_true", help="upload only: first-time create instead of version")
    a = ap.parse_args()

    datasets = a.datasets or ([a.dataset] if a.dataset else sorted(ALLOWED_DATASETS))
    if not datasets:
        raise SystemExit("nothing to sync: pass --dataset or --datasets")

    rc = 0
    for ds in datasets:
        # resolve_dataset refuses a foreign id before plan() (and therefore before any token/network).
        item = plan(a.action, ds, a.message, a.tag, overwrite=a.overwrite, create=a.create)
        print(f"[kaggle] {a.action} {item['dataset']} <- {item['dir']}", flush=True)
        code = run_plan(item, a.dry_run)
        if code:
            rc = code
    return rc


if __name__ == "__main__":
    sys.exit(main())
