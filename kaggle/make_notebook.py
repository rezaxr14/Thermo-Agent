"""Build `kaggle/teacher.ipynb` from `kaggle/teacher_benchmark.py`.

The script is the single source of truth: it runs as a plain Python file and splits on `# %%` cell markers into
the notebook, so the two can never drift. Kaggle's "File -> Import notebook" accepts the result directly, which
means the teacher pass needs a browser and a dataset upload, not local Kaggle API credentials.

    python kaggle/make_notebook.py          # writes kaggle/teacher.ipynb
"""
from __future__ import annotations

import json
import pathlib
import re

MARK = "# %%"
HERE = pathlib.Path(__file__).resolve().parent


def cells(text: str) -> list[dict]:
    out, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                             for line in body.splitlines())
            out.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": body.splitlines(keepends=True)})

    for line in text.splitlines():
        if line.startswith(MARK):
            flush()
            buf = []
            kind = "markdown" if "markdown" in line else "code"
            continue
        buf.append(line)
    flush()
    return out


def build(py_path=None, out_path=None) -> pathlib.Path:
    py = pathlib.Path(py_path or HERE / "teacher_benchmark.py")
    out = pathlib.Path(out_path or HERE / "teacher.ipynb")
    notebook = {"cells": cells(re.sub(r'\A""".*?"""\n', "", py.read_text(encoding="utf-8"), count=1, flags=re.S)),
                "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    pairs = [(HERE / "teacher_benchmark.py", HERE / "teacher.ipynb"),
             (HERE / "train_runner.py", HERE / "train.ipynb"),
             (HERE / "teacher_harness.py", HERE / "teacher_harness.ipynb"),
             (HERE / "eval_runner.py", HERE / "eval.ipynb")]
    built = [build(*pair) for pair in pairs]
    print("\n".join(str(p) for p in built))
