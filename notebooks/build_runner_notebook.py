"""Regenerate `notebooks/kaggle_model_runner.ipynb` from the canonical `.py`.

The Kaggle task is pushed from `notebooks/kaggle_model_runner.py`, so that file is
the source of truth. The notebook is a convenience view; this script keeps it from
drifting (there is a test that fails when they disagree).

Usage:
    python notebooks/build_runner_notebook.py
"""
from __future__ import annotations

import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
PY = HERE / "kaggle_model_runner.py"
IPYNB = HERE / "kaggle_model_runner.ipynb"
SPLIT_AT = "if IS_KAGGLE:"

INTRO = (
    "# thermo-agent generic Kaggle model runner\n"
    "\n"
    "Reusable for tracegen, evaluation, zero-shot and cohort passes. Push once, then\n"
    "run explicitly with `-m gpt-6-luna -m claude-haiku-5-5`; no other model is\n"
    "called. Gemini 3.8 / 3.5 are hard-blocked.\n"
)
KAGGLE_NOTE = (
    "## Kaggle-only bootstrap and execution\n"
    "The cells below execute only on Kaggle; local import/py_compile is safe."
)


def split_source(text: str) -> tuple[str, str]:
    """Strip the module docstring and split before the Kaggle bootstrap."""
    body = re.sub(r'\A""".*?"""\n', "", text, count=1, flags=re.S)
    index = body.index(SPLIT_AT)
    return body[:index].rstrip() + "\n", body[index:]


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(py_path: pathlib.Path = PY, out_path: pathlib.Path = IPYNB) -> pathlib.Path:
    head, tail = split_source(py_path.read_text(encoding="utf-8"))
    notebook = {
        "cells": [cell("markdown", INTRO), cell("code", head),
                  cell("markdown", KAGGLE_NOTE + "\n"), cell("code", tail)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return out_path


def notebook_code(path: pathlib.Path = IPYNB) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")


if __name__ == "__main__":
    print(build())
