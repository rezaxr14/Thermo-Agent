"""Kaggle kernel: pull the private HF corpus repo into the single Kaggle dataset.

No data is uploaded from the local machine. Credentials come from the private
Kaggle dataset `rezanadimi/thermo-agent-env` (file `env`, generated from the
repo-root .env). The dataset `rezanadimi/thermo-agent-data` receives a new
version built from the HF bytes. Tokens are never printed.
"""
# %% [markdown]
# # thermo-agent: Hugging Face -> Kaggle dataset bridge
#
# Downloads `Reza-Nadimi/thermo-agent-corpora` with the private `env` dataset and
# publishes it as the single Kaggle dataset `rezanadimi/thermo-agent-data`.
# No local data upload; no Kaggle Secrets are required.

# %%
import os
import pathlib

ENV_PATH = None
for candidate in pathlib.Path("/kaggle/input").rglob("*"):
    if candidate.is_file() and candidate.name in ("env", ".env"):
        ENV_PATH = candidate
        break
if ENV_PATH is None:
    raise SystemExit("attach the private dataset rezanadimi/thermo-agent-env (file 'env')")

for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
for key in ("HF_TOKEN", "KAGGLE_API_TOKEN"):
    if not os.environ.get(key):
        raise SystemExit(f"{key} missing from attached env file")
print(f"credentials loaded from {ENV_PATH.name}; values redacted", flush=True)

# %%
from huggingface_hub import snapshot_download

HF_REPO = "Reza-Nadimi/thermo-agent-corpora"
OUT = pathlib.Path("/kaggle/working/hf_corpora")
OUT.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=HF_REPO, repo_type="dataset", token=os.environ["HF_TOKEN"],
                  local_dir=str(OUT))
files = sorted(p for p in OUT.rglob("*") if p.is_file())
print(f"downloaded {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.1f} MB from {HF_REPO}", flush=True)

# %%
import kagglehub

KAGGLE_DATASET = "rezanadimi/thermo-agent-data"
kagglehub.dataset_upload(KAGGLE_DATASET, str(OUT))
print(f"published {KAGGLE_DATASET}: https://www.kaggle.com/datasets/{KAGGLE_DATASET}", flush=True)
