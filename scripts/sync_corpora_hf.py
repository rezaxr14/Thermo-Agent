"""Mirror the training corpora and memory to a private Hugging Face dataset repo.

This is the Hugging Face half of the Kaggle/HF sync: Kaggle datasets already carry the
code bundle and the corpus zips; this pushes the same additive corpora plus the served
memory to one fixed private dataset repo so a second machine (or a rerun) can pull the
exact bytes that a training run used.

Safety rules (same spirit as thermo_agent/train/hf_sync.py):
  * the repo id is fixed to PROJECT_REPO; any other id is refused before a network call;
  * the repo must be private (created private if missing, refused if public);
  * the token is read from the environment or .env and is never printed or uploaded;
  * only allow-listed directories are uploaded, so .env, models, runs and logs can never
    be included by a glob accident.

    python scripts/sync_corpora_hf.py --dry-run
    python scripts/sync_corpora_hf.py              # upload
    python scripts/sync_corpora_hf.py --dirs data/sft_tools data/grpo_tools
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_REPO = "Reza-Nadimi/thermo-agent-corpora"
DEFAULT_DIRS = (
    "data/sft_tools",
    "data/grpo_tools",
    "data/sft_tools_h",
    "data/grpo_tools_h",
    "data/frontier",
    "data/memory",
)
# Hard allow-list: a typo or an absolute path outside data/ is refused.
ALLOWED_PREFIXES = ("data/sft_tools", "data/grpo_tools", "data/frontier", "data/memory")


def read_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no HF token in the environment or .env (value is never printed)")


def check_dirs(dirs: list[str]) -> list[Path]:
    out = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[skip] {p} is not a directory", flush=True)
            continue
        rel = p.as_posix()
        if not any(rel == a or rel.startswith(a + "/") for a in ALLOWED_PREFIXES):
            raise SystemExit(f"refusing to upload {rel!r}: not in the corpus/memory allow-list")
        out.append(p)
    if not out:
        raise SystemExit("nothing to upload")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=PROJECT_REPO)
    ap.add_argument("--dirs", nargs="+", default=list(DEFAULT_DIRS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="sync thermo-agent corpora")
    a = ap.parse_args()

    if a.repo != PROJECT_REPO:
        raise SystemExit(f"refusing HF dataset {a.repo!r}: the only project dataset is {PROJECT_REPO!r}")

    dirs = check_dirs(a.dirs)
    for p in dirs:
        files = [f for f in p.rglob("*") if f.is_file()]
        print(f"[plan] {p.as_posix()}: {len(files)} files, {sum(f.stat().st_size for f in files)/2**20:.2f} MiB")
    if a.dry_run:
        print(f"[dry-run] would upload to private dataset {a.repo}")
        return 0

    from huggingface_hub import HfApi

    token = read_token()
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(a.repo, token=token)
    if not getattr(info, "private", False):
        raise SystemExit(f"refusing {a.repo}: the existing dataset is public")

    for p in dirs:
        target = p.as_posix()
        api.upload_folder(
            folder_path=str(p), path_in_repo=target, repo_id=a.repo,
            repo_type="dataset", commit_message=f"{a.message}: {target}",
        )
        print(f"[hf] uploaded {target}")

    readme = (
        "# thermo-agent corpora\n\n"
        "Private mirror of the additive training corpora used by the thermo-agent project.\n\n"
        "- `data/sft_tools` / `data/grpo_tools`: deterministic multi-turn tool traces (plain prompt).\n"
        "- `data/sft_tools_h` / `data/grpo_tools_h`: same completions with the harness-tool system prompt\n"
        "  that the `ours_harness` eval arm uses (AGENTS.md + skills + shared `<tool_call>` contract).\n"
        "- `data/frontier`: verifier-graded frontier teacher traces.\n"
        "- `data/memory`: served success/failure notes (no model ids, timestamps or costs in the rendered text).\n\n"
        "No `.env`, tokens, model weights or run logs are uploaded.\n"
    )
    import io
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")), path_in_repo="README.md", repo_id=a.repo,
        repo_type="dataset", commit_message="dataset card",
    )
    print(f"[hf] done: https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
