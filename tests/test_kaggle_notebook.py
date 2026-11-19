"""The Kaggle notebook must stay in sync with the script it is built from, and it must not carry an answer.

`kaggle/teacher.ipynb` is generated, not hand-edited: `kaggle/make_notebook.py` splits `teacher_benchmark.py` on its
`# %%` markers. These tests fail if the checked-in notebook is stale, run the notebook's own cells against a stub
model to prove the loop is resumable and records failures, and re-check that the export carries no completions.
"""
import importlib.util
import json
import os
import re
import pathlib
import sys
import types
import zipfile

import pytest

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
    # Isolate the notebook's shared-progress input: a real HF_TOKEN in the environment flips SYNC on,
    # so progress.sync() pulls the live corpus (4149 rep-70-… rows) and writes it into OUT.
    monkeypatch.delenv("HF_TOKEN", raising=False)

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
        # P0-10: GRPO is initialised from the LOCAL merged SFT weights (runs/sft_merged), not a remote alias.
        assert "torch.distributed.run" in code and "--hf-sync" in code and "sft_merged" in code
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


def test_setup_exports_pinned_site_first_and_parent_never_imports_numeric(tmp_path, monkeypatch):
    """PINNED_SITE must lead the child PYTHONPATH, and the parent module must never import numpy."""
    import thermo_agent.kaggle_notebook as kn

    source = (ROOT / "thermo_agent" / "kaggle_notebook.py").read_text(encoding="utf-8")
    assert kn.PINNED_SITE == pathlib.Path("/kaggle/working/pydeps")
    assert "_purge_numeric_modules" not in source
    assert "_blas_supports_fpe" not in source
    assert not re.search(r"(?m)^\s*(?:import\s+numpy\b|from\s+numpy\b)", source), (
        "the parent setup module must never import numpy")

    # Fake just enough of the two Kaggle datasets to run setup() without installing anything.
    code = tmp_path / "code"
    (code / "thermo_agent" / "train").mkdir(parents=True)
    (code / "thermo_agent" / "train" / "sft.py").write_text("", encoding="utf-8")
    corpus = tmp_path / "corpus"
    for name in ("sft", "grpo"):
        (corpus / name).mkdir(parents=True)
    root = tmp_path / "thermo-agent"
    monkeypatch.setattr(kn, "find_input", lambda name: code if name == "thermo-agent-code" else corpus)
    monkeypatch.setenv("PYTHONPATH", "/existing/entry")

    before = pathlib.Path.cwd()
    try:
        got = kn.setup(root=root, install=False)
    finally:
        os.chdir(before)
    assert got == root
    entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(kn.PINNED_SITE), "PINNED_SITE must be first on the child PYTHONPATH"
    assert entries[1] == str(root), "the repo root must still be present"
    assert entries[2] == "/existing/entry", "the pre-existing PYTHONPATH must be preserved"


# --- corpus staging + pre-DDP guards (the v25 "no training rows in data/sft_all" failure) ---

def _fake_code_and_corpus(tmp_path, dirs=(), zips=()):
    """Minimal code bundle + corpus dataset: dirs are directories, zips are <name>.zip archives."""
    code = tmp_path / "code"
    (code / "thermo_agent" / "train").mkdir(parents=True)
    (code / "thermo_agent" / "train" / "sft.py").write_text("", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    for name in dirs:
        d = corpus / name
        d.mkdir(parents=True)
        (d / f"{name}_train.jsonl").write_text("{}\n", encoding="utf-8")
    for name in zips:
        payload = tmp_path / f"_{name}_payload"
        payload.mkdir()
        row = payload / "repair_train.jsonl"
        row.write_text("{}\n", encoding="utf-8")
        with zipfile.ZipFile(corpus / f"{name}.zip", "w") as zf:
            zf.write(row, "repair_train.jsonl")
    return code, corpus


def _setup_with(tmp_path, monkeypatch, code, corpus, **kwargs):
    """Run setup() against fake Kaggle inputs, restoring cwd/PYTHONPATH afterwards."""
    import thermo_agent.kaggle_notebook as kn

    monkeypatch.setattr(kn, "find_input", lambda name: code if name == "thermo-agent-code" else corpus)
    monkeypatch.setenv("PYTHONPATH", "/existing/entry")
    root = tmp_path / "thermo-agent"
    before = pathlib.Path.cwd()
    try:
        got = kn.setup(root=root, install=False, **kwargs)
    finally:
        os.chdir(before)
    return kn, root, got


def test_setup_default_stages_merged_and_heldout_from_zips(tmp_path, monkeypatch, capsys):
    """The default corpora now include sft_all/grpo_all/heldout, extracted from <name>.zip."""
    code, corpus = _fake_code_and_corpus(
        tmp_path, dirs=("sft", "grpo"), zips=("sft_all", "grpo_all", "heldout"))
    kn, root, got = _setup_with(tmp_path, monkeypatch, code, corpus)
    assert got == root
    for name in ("sft_all", "grpo_all", "heldout"):
        assert (root / "data" / name / "repair_train.jsonl").is_file(), f"{name} was not staged"
    for name in ("sft", "grpo", "sft_all", "grpo_all", "heldout"):
        assert name in kn.STAGED_CORPORA
    assert kn.MISSING_CORPORA == ()
    assert "[corpus] staged" in capsys.readouterr().out


def test_setup_missing_corpus_is_skipped_not_faked(tmp_path, monkeypatch, capsys):
    """An absent corpus is skipped, printed as missing, and no empty dir is created for it."""
    code, corpus = _fake_code_and_corpus(tmp_path)
    kn, root, got = _setup_with(tmp_path, monkeypatch, code, corpus, corpora=("ghost",))
    assert got == root
    assert "[corpus] missing ghost" in capsys.readouterr().out
    assert not (root / "data" / "ghost").exists(), "a missing corpus must not create an empty dir"
    assert kn.MISSING_CORPORA == ("ghost",)
    assert kn.STAGED_CORPORA == ()


def _runner_cell(make_notebook, name, needle, tmp_path):
    """Build a runner notebook and return the source of the cell defining `needle`."""
    nb = pathlib.Path(make_notebook.build(KAGGLE / name, tmp_path / f"{name}.ipynb"))
    for cell in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and needle in src:
            return src
    raise AssertionError(f"no code cell in {name} contains {needle!r}")


def test_train_runner_pre_ddp_corpus_guard(tmp_path):
    """require_train_corpus() fails before DDP on an empty corpus, and passes with one train row."""
    make_notebook = _load_make_notebook()
    src = _runner_cell(make_notebook, "train_runner.py", "def require_train_corpus", tmp_path)
    ns = {"ROOT": tmp_path}
    exec(compile(src, "<train-corpus-guard>", "exec"), ns)  # noqa: S102
    guard = ns["require_train_corpus"]

    d = tmp_path / "data" / "sft_all"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        guard("sft_all", "SFT_DATA")
    msg = str(exc.value)
    assert "sft_all" in msg and "SFT_DATA" in msg
    assert str(d) in msg, "the error must name the missing corpus dir"

    (d / "repair_train.jsonl").write_text("{}\n", encoding="utf-8")
    assert guard("sft_all", "SFT_DATA") == d


def test_eval_runner_heldout_test_guard(tmp_path):
    """require_test_corpus() fails with the missing path when heldout has no *_test_*.jsonl."""
    make_notebook = _load_make_notebook()
    src = _runner_cell(make_notebook, "eval_runner.py", "def require_test_corpus", tmp_path)
    ns = {"ROOT": tmp_path}
    exec(compile(src, "<eval-corpus-guard>", "exec"), ns)  # noqa: S102
    guard = ns["require_test_corpus"]

    d = tmp_path / "data" / "heldout"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        guard("heldout")
    msg = str(exc.value)
    assert "heldout" in msg and "*_test_*.jsonl" in msg
    assert str(d) in msg, "the error must name the missing corpus dir"

    (d / "repair_test_ood_fault.jsonl").write_text("{}\n", encoding="utf-8")
    assert guard("heldout") == d


def test_grpo_batch_math_for_k_4_8_16(tmp_path, monkeypatch):
    """The GRPO cell must resolve a k-divisible per-device batch so ddp.assert_grpo_batches accepts k=4/8/16."""
    make_notebook = _load_make_notebook()
    src = _runner_cell(make_notebook, "train_runner.py", "--num-generations", tmp_path)
    from thermo_agent.train.ddp import grpo_batches_divisible

    calls = []
    ns = {
        "os": os, "PY": "python", "NGPU": 2, "ROOT": tmp_path,
        "merged": tmp_path / "sft_merged", "GRPO_DATA": "grpo_all", "SYNC": "auto", "LIMITS": [],
        "status": lambda **kw: calls.append(("status", kw)),
        "require_train_corpus": lambda *a, **kw: None,
        "hf_sync": types.SimpleNamespace(local_steps=lambda *a, **kw: []),
        "run": lambda argv: calls.append(("run", argv)),
    }

    for k in (4, 8, 16):
        monkeypatch.setenv("GRPO_K", str(k))
        monkeypatch.delenv("GRPO_BATCH", raising=False)
        monkeypatch.delenv("GRPO_ACCUM", raising=False)
        calls.clear()
        exec(compile(src, "<train-grpo>", "exec"), ns)  # noqa: S102

        argv = next(a for name, a in calls if name == "run")
        flag = lambda name: argv[argv.index(name) + 1]  # noqa: E731
        batch, accum = int(flag("--per-device-batch")), int(flag("--grad-accum"))
        assert flag("--num-generations") == str(k)
        assert batch == 2, f"k={k}: decided per-device batch is 2"
        assert accum == k, "decided: grad-accum = k, so every k sees 4 prompts per optimizer step"
        assert grpo_batches_divisible(batch, accum, 2, k), "effective batch must be a whole group too"

    # Explicit env overrides win over the k-derived default.
    monkeypatch.setenv("GRPO_K", "8")
    monkeypatch.setenv("GRPO_BATCH", "32")
    monkeypatch.setenv("GRPO_ACCUM", "4")
    calls.clear()
    exec(compile(src, "<train-grpo>", "exec"), ns)  # noqa: S102
    argv = next(a for name, a in calls if name == "run")
    assert argv[argv.index("--per-device-batch") + 1] == "32"
    assert argv[argv.index("--grad-accum") + 1] == "4"
