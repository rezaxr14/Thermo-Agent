# Tiny committed fixtures (fresh-clone safe)

These are verbatim, trimmed copies of real rows so prompt / parse / tool-shape
tests can run on a clone that has none of the untracked `data/processed*`,
`data/sft*`, `data/grpo*` or `data/frontier*` corpora.

- `corpus/<task>_train.jsonl` — 2 rows per task, copied from `data/sft/*_train.jsonl`.
- `heldout/<task>_test_id.jsonl` — 1 row per task, copied from `data/heldout/*_test_id.jsonl`.
- `heldout/meta_<task>.json` — matching split metadata.

They are intentionally small (no full dataset).  They are exposed as pytest
fixtures in `tests/conftest.py` (`corpus_fixture_dir`, `heldout_fixture_dir`,
`corpus_fixture_rows`, `heldout_fixture_rows`).  Real-data tests should be
marked `@pytest.mark.corpus`; the conftest guard skips those cleanly when the
untracked corpora are absent and leaves them untouched when present.

`tests/test_contamination.py` never scans this directory — the held-out copies
here are fixtures, not training artifacts.
