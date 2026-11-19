"""The generated Kaggle smoke kernel must carry the corpus-staging setup call.

`scripts/make_smoke_kernel.py` copies the maintained `kaggle/train.ipynb` and prepends one cell
of small-run env vars. These tests fail if the checked-in smoke notebook is stale or if it lost
the explicit `setup(corpora=...)` call that stages `sft_all`/`grpo_all` (the v25 staging bug).
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
SMOKE = KAGGLE / "train_kernel_smoke" / "train.ipynb"


def _smoke_nb():
    return json.loads(SMOKE.read_text(encoding="utf-8"))


def test_smoke_notebook_contains_new_setup_call():
    code = "".join("".join(c["source"]) for c in _smoke_nb()["cells"] if c["cell_type"] == "code")
    assert "setup(corpora=tuple(dict.fromkeys(" in code, "smoke kernel lost the explicit setup(corpora=...) call"
    assert "SFT_DATA, GRPO_DATA" in code
    assert '"sft_all", "grpo_all"' in code
    assert 'require_train_corpus(SFT_DATA, "SFT_DATA")' in code


def test_smoke_notebook_matches_generator():
    """The smoke notebook is exactly make_smoke_kernel.py's transform of the current train.ipynb."""
    nb = json.loads((KAGGLE / "train.ipynb").read_text(encoding="utf-8"))
    env = {"STEPS": "20", "ROWS": "64", "TASK": "repair", "NGPU": "1",
           "SFT_BATCH": "1", "SFT_ACCUM": "8",
           "SFT_DATA": "sft_tools", "GRPO_DATA": "grpo_tools"}
    source = ["import os\n"] + [f'os.environ["{k}"]="{v}"\n' for k, v in env.items()]
    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": source})
    assert nb == _smoke_nb(), "kaggle/train_kernel_smoke/train.ipynb is stale: run scripts/make_smoke_kernel.py"
