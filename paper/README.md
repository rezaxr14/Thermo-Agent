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

- Authors, affiliations and date (title block), repository, dataset and checkpoint links (data and code availability), dataset licence (appendix D).
- `headline model result` (abstract, conclusion).
- Model results: base, SFT and SFT+GRPO per task, with harness none and agents+skills (table `tab:results`); ablations (table `tab:ablation`); GPU hours and costs (appendix F); runbook steps 4 and 5.
- `Frontier trace results` (section 5): the Frontier file has not been downloaded.
- Positioning table (section 2): the second-law column of six prior works is `\todo{check}`.

## Placeholder figures

None. All 13 figures are the final vector figures (SVG with a PDF twin) from the figure build.

## References marked TODO-verify

`qwen2026qwen35` (licence taken from derivative model cards), `smith1999grimech` (project web page not opened), `jadhav2026frontierml` (author list from the first PDF page only).
