"""The corpus token audit must flag rows the trainer would truncate, and pass the ones it would not."""
import json

import pytest

from thermo_agent.synth import combine
from thermo_agent.train import token_audit

from _rows import make_row


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    if combine.tokenizer() is None:
        pytest.skip("no tokenizer available (set THERMO_TOKENIZER or cache the Qwen3.5 tokenizer)")
    row = make_row("wrong_fluid")
    row = dict(row, tokens=None)
    row["prompt"] = [dict(m) for m in row["prompt"]]
    write(tmp_path / "repair_train.jsonl", [row])
    return tmp_path


def test_audit_reports_the_measured_lengths(corpus):
    r = token_audit.audit(corpus)
    f = r["files"]["repair_train"]
    assert f["n"] == 1 and f["total_max"] >= f["prompt_max"] > 0
    assert r["rows"] == 1 and r["fits"] is True and r["over_total"] == 0
    assert r["tokenizer"].endswith("tokenizer.json")


def test_audit_fails_a_corpus_over_the_sequence_length(corpus):
    r = token_audit.audit(corpus, max_length=10)
    assert r["fits"] is False and r["over_total"] == 1
    assert r["over"][0]["tokens"] > 10 and r["over"][0]["task"] == "repair"
