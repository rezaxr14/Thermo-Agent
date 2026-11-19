"""Kaggle kernel: pull the private HF corpus repo into the single Kaggle dataset.

No data is uploaded from the local machine. HF_TOKEN (and, if needed,
KAGGLE_API_TOKEN) are read from Kaggle Secrets. The dataset
``rezanadimi/thermo-agent-data`` receives a new version built from the HF bytes.
"""
# %% [markdown]
# # thermo-agent: Hugging Face -> Kaggle dataset bridge
#
# Pulls `Reza-Nadimi/thermo-agent-corpora` (private) and publishes it as the single
# Kaggle dataset `rezanadimi/thermo-agent-data`. Data never leaves HF except into
# Kaggle. Tokens come from Kaggle Secrets, never from notebooks or datasets.

# %%
import os
import pathlib

from kaggle_secrets import UserSecretsClient

HF_REPO = "Reza-Nadimi/thermo-agent-corpora"
KAGGLE_DATASET = "rezanadimi/thermo-agent-data"
OUT = pathlib.Path("/kaggle/working/hf_corpora")

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
try:
    os.environ["KAGGLE_API_TOKEN"] = secrets.get_secret("KAGGLE_API_TOKEN")
except Exception as exc:  # kagglehub may already have kernel credentials
    print("KAGGLE_API_TOKEN secret unavailable:", type(exc).__name__)

# %%
from huggingface_hub import snapshot_download

OUT.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=HF_REPO, repo_type="dataset", token=os.environ["HF_TOKEN"],
                  local_dir=str(OUT))
files = sorted(p for p in OUT.rglob("*") if p.is_file())
print(f"downloaded {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.1f} MB from {HF_REPO}", flush=True)

# %%
import kagglehub

kagglehub.dataset_upload(KAGGLE_DATASET, str(OUT))
print(f"published {KAGGLE_DATASET}: https://www.kaggle.com/datasets/{KAGGLE_DATASET}", flush=True)
