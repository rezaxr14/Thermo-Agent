# Second-Law-Verified Language-Model Agents for On-Site Power, Cooling and Heat Recovery in AI Data Centers

Draft paper (LaTeX, 11 pt article, natbib author-year). System name macro: `\sys` = `thermo-agent`.

## Build and check

    python tools/verify_paper.py          # latexmk build, then all checks; prints the inventory below
    python tools/generate_paper_assets.py --project PATH/TO/thermo-agent --scratch DIR --tests-log PYTEST_LOG [--stage ...] [--force]

`tools/generate_paper_assets.py` writes `assets/*.json` (one study each: environment, crosscheck, example_plant, mechanisms, repair_baselines, audit_baselines),
`assets_summary.json` (value, TeX form, source asset, command, seed and sample size for every `\val{key}`), `generated/values.tex` and `generated/table_*.tex`.
The other files in `assets/` and everything in `figures/` come from the figure build (`figures/figure_manifest.json` lists each figure with its data source and pending items).

## Rules kept

- Every measured number is a `\val{key}`; every key is in `assets_summary.json`.
- Pending results are red `\todo{...}`; no model has been trained and no figure shows a trained-model result.
- The work is simulation-driven: published load and coolant data set operating points of a physics model; there is no measured on-site-generation data center.

## Pending items (`\todo`)

- Authors, affiliations and date, repository, dataset and checkpoint links, dataset licence.
- `headline model result`; model results and ablations (`tab:results`, `tab:ablation`); names of the compared models; GPU hours and costs.
- Corpus: variants per row (target 20 or more) for Repair and the other tasks; final corpus sizes per source, task and split; teacher models and quota; teacher acceptance statistics (`tab:corpusstatus`); number and families of teacher-authored diagrams; final catalog (families, diagrams, held-out diagrams); leakage check for memory notes.
- Reward diagnostic re-run with the training group size (4); `Frontier trace results`.
- Related work: citations for rejection-sampling fine-tuning, retrieval-augmented generation and failure memory; second-law column of six prior works in the positioning table.

## Described in the paper, to be built in the repository

The paper describes the intended system. These parts are not in the repository at the time of writing and each has a matching `\todo`:
teacher-authored diagrams (plants written by the teacher, checked, then turned into fault, ledger and audit rows); catalog families beyond the data-center plants and the steam-and-HRSG plant (Rankine, ORC-only, further cogeneration, combined cycles) with the components they need; failure-memory retrieval in the evaluator and in GRPO; paraphrase variants for Ledger, Audit and Design rows and 20 or more variants for Repair; export and ingest helpers for the Kaggle teacher pass.

## Placeholder figures

None. All 13 figures are vector figures (SVG with a PDF twin) from the figure build.

## References marked TODO-verify

`qwen2026qwen35` (licence taken from derivative model cards), `smith1999grimech` (project web page not opened), `jadhav2026frontierml` (author list from the first PDF page only).
