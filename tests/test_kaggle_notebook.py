"""The Kaggle notebook must stay in sync with the script it is built from, and it must not carry an answer.

`kaggle/teacher.ipynb` is generated, not hand-edited: `kaggle/make_notebook.py` splits `teacher_benchmark.py` on its
`# %%` markers. These tests fail if the checked-in notebook is stale, run the notebook's own cells against a stub
model to prove the loop is resumable and records failures, and re-check that the export carries no completions.
"""
import importlib.util
import json
import re
import pathlib
import sys
import types

from thermo_agent.teacher import transfer

from _rows import make_row

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"


def _load_make_notebook():
    """By path: the repo's `kaggle/` directory must not shadow the installed `kaggle` CLI package."""
    spec = importlib.util.spec_from_file_location("thermo_make_notebook", KAGGLE / "make_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells():
    nb = json.loads((KAGGLE / "teacher.ipynb").read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_notebook_is_up_to_date(tmp_path):
    make_notebook = _load_make_notebook()
    built = make_notebook.build(KAGGLE / "teacher_benchmark.py", tmp_path / "teacher.ipynb")
    checked_in = KAGGLE / "teacher.ipynb"
    assert checked_in.exists(), "run `python kaggle/make_notebook.py` and commit kaggle/teacher.ipynb"
    assert json.loads(built.read_text(encoding="utf-8")) == json.loads(checked_in.read_text(encoding="utf-8")), (
        "kaggle/teacher.ipynb is stale: run `python kaggle/make_notebook.py`")


def test_notebook_is_valid_and_has_no_answer_column():
    cells = _cells()
    assert len(cells) >= 4 and any(kind == "markdown" for _, kind in cells)
    code = "".join(src for src, kind in cells if kind == "code")
    assert "kbench.chats.new" in code and "completions.jsonl" in code
    assert 'row["completion"]' not in code, "the notebook must never read a ground-truth completion"


class _StubLLM:
    """Answers every prompt but one, so the failure path is exercised too."""

    def prompt(self, text: str) -> str:
        if "FAIL" in text:
            raise RuntimeError("quota exhausted")
        return "stub answer"


class _Chats:
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    def new(self, name="Chat", system_instructions=None, orphan=False):
        assert system_instructions is None or isinstance(system_instructions, str)
        return self._Ctx()


def test_notebook_writes_one_line_per_attempt_and_records_errors(tmp_path, monkeypatch):
    # cell 1 pins OUT to /kaggle/working; redirect it before the loop cell runs
    out = tmp_path / "completions.jsonl"
    srcs = [(s, k) for s, k in _cells() if k == "code"]
    ns = {"__name__": "__notebook__"}
    export = tmp_path / "input" / "thermo-teacher-prompts"
    export.mkdir(parents=True)
    good, bad = make_row("wrong_fluid"), dict(make_row("unit_slip"), id="row-b")
    rows = [dict(good, id="row-a", task="repair", harness="none", topology="dc_chp"),
            dict(bad, id="row-b", task="repair", harness="none", topology="dc_chp")]
    for i, r in enumerate(rows):
        r["prompt"] = [{"role": "system", "content": "sys"}, {"role": "user", "content": f"user {i}"}]
    rows[1]["prompt"][1]["content"] = "FAIL please"
    (export / "repair_train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (export / "export_manifest.json").write_text("{}", encoding="utf-8")

    stub = types.ModuleType("kaggle_benchmarks")
    stub.llms = {"stub/model": _StubLLM()}
    stub.llm = _StubLLM()
    stub.chats = _Chats()
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", stub)
    monkeypatch.chdir(tmp_path)

    exec(compile(srcs[0][0], "<notebook>", "exec"), ns)          # noqa: S102
    ns["OUT"] = out                                             # the only line the notebook hard-codes
    for src, _ in srcs[1:]:
        exec(compile(src, "<notebook>", "exec"), ns)            # noqa: S102

    written = [json.loads(l) for l in out.open(encoding="utf-8")]
    assert [r["id"] for r in written] == ["row-a", "row-b"], "one line per attempt, ids preserved"
    assert written[0]["completion"] == "stub answer"
    assert "quota exhausted" in written[1]["error"], "a failed call is recorded, not fatal"
    # The zero-shot baseline notebook default is now the user-selected cheap model (D7).
    assert all(r["model"] == "claude-haiku-5-5" for r in written)

    # second run: everything already answered, nothing duplicated
    ns2 = {"__name__": "__notebook__"}
    exec(compile(srcs[0][0], "<notebook>", "exec"), ns2)         # noqa: S102
    ns2["OUT"] = out
    for src, _ in srcs[1:]:
        exec(compile(src, "<notebook>", "exec"), ns2)            # noqa: S102
    assert len([json.loads(l) for l in out.open(encoding="utf-8")]) == 2, "resumable: no duplicate rows"


def test_export_never_carries_a_completion(tmp_path):
    row = make_row("wrong_fluid")
    src = tmp_path / "sft"
    src.mkdir()
    sft = dict(row, id="row-a", task="repair", source="ground_truth", harness="none", memory_notes=[], tokens=100)
    (src / "repair_train.jsonl").write_text(json.dumps(sft) + "\n", encoding="utf-8")
    transfer.export(src, tmp_path / "export", tasks=("repair",), per_task=5)
    exported = [json.loads(l) for l in (tmp_path / "export" / "repair_train.jsonl").open(encoding="utf-8")]
    assert exported and all("completion" not in r for r in exported)

def test_train_notebook_is_generated_and_has_no_token():
    """The training notebook comes from train_runner.py and resumes from the private HF repo."""
    make_notebook = _load_make_notebook()
    out = pathlib.Path(make_notebook.build(KAGGLE / "train_runner.py", KAGGLE / "_train_check.ipynb"))
    try:
        assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
            (KAGGLE / "train.ipynb").read_text(encoding="utf-8")), (
            "kaggle/train.ipynb is stale: run `python kaggle/make_notebook.py`")
        code = "".join("".join(c["source"]) for c in json.loads(out.read_text(encoding="utf-8"))["cells"]
                       if c["cell_type"] == "code")
        assert "torch.distributed.run" in code and "--hf-sync" in code and "hf:merged/sft" in code
        assert "hf_sync.ensure_private_repo" in code
        assert "kaggle_notebook import" in code and "token()" in code, "the token must come from the shared Kaggle Secret helper"
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", code), "no token literal in the notebook"
    finally:
        out.unlink(missing_ok=True)


def test_tools_notebook_is_generated_and_exposes_the_physics_tool():
    """The tool-using CoT arm must run the LangGraph harness with real tools, skills, memory and keep the trajectory."""
    make_notebook = _load_make_notebook()
    out = pathlib.Path(make_notebook.build(KAGGLE / "teacher_harness.py", KAGGLE / "_harness_check.ipynb"))
    try:
        assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
            (KAGGLE / "teacher_harness.ipynb").read_text(encoding="utf-8")), (
            "kaggle/teacher_harness.ipynb is stale: run `python kaggle/make_notebook.py`")
        code = "".join("".join(c["source"]) for c in json.loads(out.read_text(encoding="utf-8"))["cells"]
                       if c["cell_type"] == "code")
        assert "run_graph" in code and "agent_graph" in code, "must use the LangGraph harness"
        assert "tool_names" in code and "fresh_notes" in code, "must expose real tools and fresh memory"
        assert "trajectory" in code and "sft_completion" in code, "must persist the tool-using trajectory for SFT"
        assert "kbench_adapter" in code and "make_respond" in code, "must deliver the full prompt via the shared adapter"
        assert "harness_completions.jsonl" in code
    finally:
        out.unlink(missing_ok=True)


def test_eval_notebook_runs_every_arm_and_writes_comparison():
    """The eval notebook must actually invoke each arm, then build the comparison and tool-use assets."""
    make_notebook = _load_make_notebook()
    out = pathlib.Path(make_notebook.build(KAGGLE / "eval_runner.py", KAGGLE / "_eval_check.ipynb"))
    try:
        assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
            (KAGGLE / "eval.ipynb").read_text(encoding="utf-8")), (
            "kaggle/eval.ipynb is stale: run `python kaggle/make_notebook.py`")
        code = "".join("".join(c["source"]) for c in json.loads(out.read_text(encoding="utf-8"))["cells"]
                       if c["cell_type"] == "code")
        for arm in ("ours_base", "ours_harness", "frontier_zero", "frontier_tools", "other_model"):
            assert arm in code
        assert "thermo_agent.eval.model_eval" in code and "comparison.main" in code
        assert "tool_usage.summarize" in code and "figures_from_examples.py" in code
    finally:
        out.unlink(missing_ok=True)


def test_shared_notebook_setup_has_no_token_literal():
    """The three code-bearing notebooks share one setup module; it reads the secret, never embeds a token."""
    code = (ROOT / "thermo_agent" / "kaggle_notebook.py").read_text(encoding="utf-8")
    assert "UserSecretsClient().get_secret(\"HF_TOKEN\")" in code
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", code)
    for name in ("train_runner.py", "teacher_harness.py", "eval_runner.py"):
        src = (KAGGLE / name).read_text(encoding="utf-8")
        assert "from thermo_agent.kaggle_notebook import" in src, f"{name} must use the shared setup"
