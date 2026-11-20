"""Mirror the CURRENT training corpora + served memory to one fixed private HF dataset repo.

This is the Hugging Face half of the Kaggle/HF sync. Kaggle datasets carry the code bundle and the
corpus zips; this pushes the exact bytes a training run used to one fixed private *dataset* repo so a
second machine (or a rerun) can pull them. It is deliberately separate from the *checkpoint* repo
``Reza-Nadimi/thermo-agent-checkpoints`` managed by ``thermo_agent/train/hf_sync.py``: that repo holds
optimizer/LoRA state under distinct ``sft/`` and ``grpo_*`` stage prefixes and can never be written by
this script.

Safety rules (same spirit as thermo_agent/train/hf_sync.py):
  * the repo id is fixed to PROJECT_REPO; any other id is refused before a network call;
  * the repo must be private (created private if missing, refused if public);
  * the token is read from the environment or .env and is never printed or uploaded;
  * only allow-listed directories are uploaded, so .env, models, runs and logs can never be included
    by a glob accident;
  * ``data/heldout_v2`` is uploaded PROMPTS ONLY: a per-field filter removes completions, reports,
    verdicts, ground truth and every other solution-bearing field, at every nesting depth;
  * remote files are never deleted. Every refresh is an additive commit with an explicit message/tag
    and a manifest (paths, sha256, sizes, repo id, timestamp) written locally and uploaded to the repo.

    python scripts/sync_corpora_hf.py --dry-run                         # plan + manifest, no network
    python scripts/sync_corpora_hf.py --message "v2 corpus rebuild" --tag 2026-11-19   # real sync
    python scripts/sync_corpora_hf.py --dirs data/heldout_v2 --dry-run  # one corpus
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fixed. Never take a dataset id from an env var or a notebook cell.
PROJECT_REPO = "Reza-Nadimi/thermo-agent-corpora"

# The current corpora that matter (H1-H3): the v2 tool/mix corpora, the compact-harness variants, the
# A slice, the frontier teacher sets, served memory, and the frozen held-out prompts.
# Reza (2026-11-20): only the data we will actually use again — the ground-truth
# generators, hosted journals, Kaggle bundles, provenance, all frontier trace dirs,
# the current v2 training corpora, memory and the frozen held-out prompts. Legacy
# corpora (sft, grpo, sft_all, grpo_all, _pre_leakfix, old tool corpora) are NOT synced.
NEEDED = (
    "data/processed", "data/processed_arch", "data/processed_big",
    "data/teacher_runs", "data/kaggle_datasets", "data/provenance",
    "data/frontier", "data/frontier_eval_export", "data/frontier_export",
    "data/frontier_prov", "data/frontier_tools", "data/frontier_tools_clean",
    "data/frontier_tools_clean_v2", "data/frontier_tools_cohort",
    "data/frontier_tools_accepted", "data/frontier_tools_cohort_prompts",
    "data/frontier_tools_pilot",
    "data/sft_tools_v2", "data/grpo_tools_v2",
    "data/sft_tools_v2_h", "data/grpo_tools_v2_h",
    "data/sft_mix_v2", "data/grpo_mix_v2",
    "data/a_slice", "data/memory", "data/heldout_v2",
)
DEFAULT_DIRS = tuple(d for d in NEEDED if Path(d).is_dir())
ALLOWED_PREFIXES = NEEDED
HELDOUT_DIRS = ("data/heldout_v2",)

# Prompts-only projection for held-out data. Only these fields may leave the machine; anything not
# listed here (in particular the solution) is dropped. Defense in depth: FORBIDDEN_FIELDS is removed
# recursively even from within a kept field.
HELDOUT_KEEP_FIELDS = frozenset({
    "id", "record_id", "split", "task", "topology", "fault_type", "tier", "gen_seed", "prompt",
})
FORBIDDEN_FIELDS = frozenset({
    "completion", "answer", "answers", "report", "reports", "verdict", "verdicts",
    "solution", "solutions", "reference", "references", "ground_truth", "truth",
    "label", "labels", "target", "targets", "gold", "expected", "original",
    "plant_original", "corrected", "grade", "grading", "score", "reward",
    "rationale", "edits", "diagnosis", "feedback", "memory_notes", "notes",
})
SKIP_PARTS = frozenset({".git", "__pycache__", ".ipynb_checkpoints"})


def _is_secret_file(path: Path) -> bool:
    """Never upload credentials, even when they live inside a data directory."""
    name = path.name.lower()
    return name.startswith(".env") or name == "env" or name.endswith(".env")


# --------------------------------------------------------------------------------------------- credentials
def read_token() -> str:
    """HF_TOKEN from env, else from a KEY=value line in .env. The value is never printed."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                token = value.strip().strip('"').strip("'")
                if token:
                    return token
    raise SystemExit("no HF token in the environment or .env (value is never printed)")


# --------------------------------------------------------------------------------------------- pure helpers
def resolve_repo(repo_id: str) -> str:
    """Accept only this project's private dataset repo. Pure: no network call, no token read."""
    if repo_id != PROJECT_REPO:
        raise ValueError(f"refusing HF dataset {repo_id!r}: the only project dataset is {PROJECT_REPO!r}")
    return repo_id


def _rel(path: Path) -> str:
    """Repo-relative POSIX form. Absolute paths under cwd are normalised so filters still apply."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(Path.cwd().resolve())
        except (ValueError, OSError):
            pass  # outside cwd: keep the absolute form so it can never match an allow-list prefix
    return p.as_posix()


def _under(path: Path, roots) -> bool:
    rel = _rel(path)
    return any(rel == r or rel.startswith(r + "/") for r in roots)


def check_dirs(dirs: list[str]) -> list[Path]:
    """Refuse any path outside the hard allow-list; skip (never guess) missing directories."""
    out = []
    for d in dirs:
        p = Path(d)
        if p.is_absolute() or ".." in p.parts:
            raise SystemExit(f"refusing to upload {d!r}: use a repo-relative data/ path")
        if not p.is_dir():
            print(f"[skip] {p} is not a directory", flush=True)
            continue
        if not _under(p, ALLOWED_PREFIXES):
            raise SystemExit(f"refusing to upload {_rel(p)!r}: not in the corpus/memory allow-list")
        out.append(p)
    if not out:
        raise SystemExit("nothing to upload")
    return out


def scrub_obj(obj):
    """Recursively drop any solution-bearing field name, at any depth."""
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items() if str(k).lower() not in FORBIDDEN_FIELDS}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    return obj


def filter_heldout_row(row: dict) -> dict:
    """Project one held-out JSONL row down to prompt + identifying metadata only."""
    return {k: scrub_obj(v) for k, v in row.items()
            if k in HELDOUT_KEEP_FIELDS and str(k).lower() not in FORBIDDEN_FIELDS}


def transform_bytes(path: Path, raw: bytes) -> bytes:
    """Apply the prompts-only filter to held-out files; return other corpora untouched."""
    if not _under(path, HELDOUT_DIRS):
        return raw
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        lines = []
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row = filter_heldout_row(row) if isinstance(row, dict) else scrub_obj(row)
            lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    if suffix == ".json":
        obj = json.loads(raw.decode("utf-8"))
        return (json.dumps(scrub_obj(obj), ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    # Never leak an unknown file type from the held-out directory by uploading it verbatim.
    raise SystemExit(f"refusing held-out file {path}: prompts-only filter cannot apply to {suffix!r}")


def plan_files(root: Path) -> list[dict]:
    """Return the upload plan for one directory, with sha256/size of the bytes actually uploaded."""
    records = []
    for f in sorted(root.rglob("*")):
        if (not f.is_file() or any(part in SKIP_PARTS for part in f.parts)
                or _is_secret_file(f)):
            continue
        raw = f.read_bytes()
        data = transform_bytes(f, raw)
        records.append({
            "path": f"{_rel(root)}/{f.relative_to(root).as_posix()}",
            "local_path": _rel(f),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "_data": data,
        })
    return records


def build_manifest(repo: str, dirs: list[Path], records: list[dict], message: str,
                   tag: str, mode: str) -> dict:
    return {
        "repo_id": repo,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": message,
        "tag": tag,
        "mode": mode,
        "additive_only": True,
        "heldout_prompts_only": True,
        "dirs": [_rel(p) for p in dirs],
        "file_count": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records),
        "files": [{k: r[k] for k in ("path", "sha256", "size_bytes", "local_path")} for r in records],
    }


def write_manifest(manifest: dict, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def default_manifest_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"reports/hf_corpora_manifests/corpora_{ts}.json"


def dataset_card() -> str:
    return (
        "# thermo-agent corpora\n\n"
        "Private mirror of the CURRENT additive training corpora used by the thermo-agent project.\n"
        "Checkpoints live in a different repo (`thermo-agent-checkpoints`) under distinct stage prefixes.\n\n"
        "- `data/sft_tools_v2` / `data/grpo_tools_v2`: deterministic multi-turn tool traces.\n"
        "- `data/sft_tools_v2_h` / `data/grpo_tools_v2_h`: same completions with the compact\n"
        "  harness-tool system prompt used by the `ours_harness` eval arm.\n"
        "- `data/sft_mix_v2` / `data/grpo_mix_v2`: mixed tool + plain corpora.\n"
        "- `data/a_slice`, `data/frontier_tools_clean_v2`, `data/frontier_tools_pilot`,\n"
        "  `data/frontier_tools_cohort`: A-slice and verifier-graded frontier teacher traces.\n"
        "- `data/memory`: served success/failure notes.\n"
        "- `data/heldout_v2`: **prompts only** — completions, reports, verdicts and ground truth are\n"
        "  filtered out before upload; never use these prompts as training targets.\n\n"
        "No `.env`, tokens, model weights or run logs are uploaded. Refreshes are additive: nothing is\n"
        "ever deleted; see `manifests/` for the sha256/size manifest of each commit.\n"
    )


# --------------------------------------------------------------------------------------------- command line
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=PROJECT_REPO, help=f"must be {PROJECT_REPO}")
    ap.add_argument("--dirs", nargs="+", default=list(DEFAULT_DIRS))
    ap.add_argument("--dry-run", action="store_true", help="plan + manifest only; no network, no token")
    ap.add_argument("--message", default="sync thermo-agent corpora", help="explicit commit message")
    ap.add_argument("--tag", default="", help="short version tag recorded in the manifest and commit")
    ap.add_argument("--manifest-out", default=None, help="where to write the audit manifest (JSON)")
    a = ap.parse_args()

    # Refuse a foreign repo BEFORE reading a token or importing huggingface_hub.
    try:
        repo = resolve_repo(a.repo)
    except ValueError as exc:
        raise SystemExit(str(exc))

    dirs = check_dirs(a.dirs)
    records: list[dict] = []
    for p in dirs:
        files = plan_files(p)
        heldout = _under(p, HELDOUT_DIRS)
        records.extend(files)
        print(f"[plan] {_rel(p)}: {len(files)} files, "
              f"{sum(f['size_bytes'] for f in files)/2**20:.2f} MiB"
              + (" (prompts only)" if heldout else ""), flush=True)

    manifest = build_manifest(repo, dirs, records, a.message, a.tag,
                              "dry-run" if a.dry_run else "upload")
    manifest_out = Path(a.manifest_out) if a.manifest_out else Path(default_manifest_path())
    written = write_manifest(manifest, manifest_out)
    print(f"[manifest] {written} ({manifest['file_count']} files, {manifest['total_bytes']} bytes)", flush=True)

    if a.dry_run:
        print(f"[dry-run] would upload to private dataset {repo}; nothing was sent")
        return 0

    from huggingface_hub import HfApi  # imported only after all refusals

    token = read_token()
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo, token=token)
    if not getattr(info, "private", False):
        raise SystemExit(f"refusing {repo}: the existing dataset is public")

    suffix = f" [tag {a.tag}]" if a.tag else ""
    commit = f"{a.message}{suffix}"
    for p in dirs:
        target = _rel(p)
        if _under(p, HELDOUT_DIRS):
            # Filtered content: upload the transformed bytes, never the original file.
            for r in records:
                if r["path"].startswith(target + "/"):
                    api.upload_file(path_or_fileobj=io.BytesIO(r["_data"]), path_in_repo=r["path"],
                                    repo_id=repo, repo_type="dataset",
                                    commit_message=f"{commit}: {r['path']} (prompts only)")
            print(f"[hf] uploaded {target} (prompts only)")
        else:
            api.upload_folder(folder_path=str(p), path_in_repo=target, repo_id=repo,
                              repo_type="dataset", commit_message=f"{commit}: {target}")
            print(f"[hf] uploaded {target}")

    api.upload_file(path_or_fileobj=io.BytesIO(dataset_card().encode("utf-8")),
                    path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message=f"{commit}: dataset card")
    api.upload_file(path_or_fileobj=io.BytesIO((json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")),
                    path_in_repo=f"manifests/{written.name}", repo_id=repo, repo_type="dataset",
                    commit_message=f"{commit}: sync manifest")
    print(f"[hf] done: https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
