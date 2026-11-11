# Data archive: contents

The data archive `paper1_data.zip` (116 MiB, 17,744 files) is attached to the GitHub release of this
repository; the same census, hierarchy, instruments and silver labels are browsable as parquet on the
Hugging Face hub at https://huggingface.co/datasets/alex2awesome/CreativePreferences.
Paths below are relative to the unzipped `data/` folder. The repository layout is preserved under
`data/outputs/...` so that the analysis code runs unchanged once `ROOT` points at the unzipped `data`
folder. Domain ids used in file names: code-review = Software Code, creative-writing,
grant-funding = Grant Proposals, humor, legal-outcome-prediction = Legal Arguments,
math-stackexchange = Math, news-homepages = Journalism, notice-and-comment = Regulatory Comments,
patents, peer-review = Academic Articles, press-releases.
## data/

**census/<domain>.criteria.jsonl.gz** (53,381 criteria; the paper's 53,413 includes 32 records whose
parse record was dropped at ingest). One record per extracted criterion:
`key` (domain::layer::document::index), `domain`, `task`, `layer` (raw = fetched page,
crp = later collection waves), `doc`, `name`, `description`, `guidance`, `canonical` (one-sentence
restatement), `specificity` (general / specific / hyper-specific), `source_id` (host and path),
`url` (where the fetch log records it), `strata` (subfield, page type, intended audience),
`codification_rung` (1 official ... 5 folk; Appendix H.4) and `page_type`, `evidence_quote` and
`evidence_status` (verbatim quote located in the source by the quote pass, where run), and the
hierarchy ids `form_id`, `form_name`, `construct_id`, `theme_id`, `category_id`.
Page text is not released; `url`/`source_id` identify the source.

**outputs/lexicon/** (repository layout):
- Hierarchy (Appendix C, D): `partition_<domain>_L0v4.json` (criterion key -> surface-form id),
  `cluster_names_<domain>_L0v4.json` (form id -> name, gloss), `partition_<domain>_R1.json` (form ->
  construct), `partition_<domain>_R2.json` (construct -> theme), `partition_<domain>_R3.json` (theme ->
  category), `node_names_<domain>_R*.json`, `derive_then_classify_v1/<domain>/{R2,R3}/taxonomy_*.json`
  (theme and category names), `cluster_gold_validation_20260724/` (external validation, Appendix D),
  `crp_ingest/` (seating ledgers of the later collection waves, Appendix B.3),
  `prereg23/` and `prereg23_*.json` (subtrees of Appendix K, per-form formality).
- Provenance and codability (Appendix A, H.3, H.4): `provenance_rungs_<domain>.jsonl`,
  `codability/`, `census_*.json`, `coverage_census_*.json`, `subfield_*`, `codability_sampling_*.json`,
  `census_secondary_analyses_20260828.json`, `subfield_age_analyses_20260828.json`.
- Instruments (Appendix F): `register_bank_20260722.jsonl` (per-name formality, etymological
  stratum, nominalization, metaphoricity, transparency), `register_height_judgments.jsonl`,
  `axis_metaphoricity_*.jsonl`, `axis_transparency_*.jsonl`, `specificity_newmetrics_20260723.jsonl`,
  `instrument_calibration_20260723.json` (Table 11), `register_corpus_20260723/` (lay-critic and
  community-rule criteria with their judgments; commenter names and affiliations are replaced by
  `[redacted]`), `community_rule_criteria_20260723.jsonl`, `lay_construct_matches_20260723.jsonl`,
  `lay_corpus_gt_20260722.json`.
- Behavioral data (Appendix E): `../silver_label_splits/` (document ids per scoring construct,
  positive and negative), `hier_nodes_<domain>.json` (the scoring partition), and the item sets with
  texts inside the experiment folders below. The raw commentary corpora (1.01M parsed norm
  applications) are not in this archive; the seated document ids and the scored item sets are.
- Judge experiments (Appendix G): `prereg15v2_20260828/` (register matching, 85 constructs, 340
  runs), `prereg25_20261012/` (matched-meaning name swap, humor; adjudication, calibrated
  re-adjudication, 420 runs), `prereg26_cw_20261012/`, `prereg27_{math,news}_20261012/`,
  `prereg28_{code-review,peer-review,press-releases}_20261012/`,
  `prereg29_{grant-funding,legal-outcome-prediction,notice-and-comment,patents}_20261012/`,
  `prereg30_humor_20261012/` (name swaps in the other ten domains and humor under the construct-level
  standard: prompts, item files, every raw judge output, per-construct tables, results),
  `review_rearticulation_20261012/` (rearticulation robustness, Table 17),
  `review_reanalyses_20261012/` (register, specificity, diffuseness, codability and norm-age checks),
  `review_extraction_20261012/` (extractor checks of Appendix A.3; source page text excluded),
  `review_repro_20261012/`, `extraction_validity_2026072*/`, `frame_calibration_20260724/`
  (Appendix B.3), `paper1_figdata.json`, `prereg15_results_20260723.json`.
  Answer keys of the adjudication anchors (`*_private`) are withheld so the gates stay reusable.

**datasets/<domain>/online-rubrics/**: fetch logs of the web search (`urls-visited.csv`, per-wave
logs); no page content.

Preregistration texts are summarized in Appendix J; the registry itself is not in this archive. Paths in logs and result files are rewritten relative to `ROOT`, account and host names are replaced by placeholders, and e-mail addresses inside item texts are replaced by `[email redacted]`.

## Code (`methods/` in this repository)

`methods/codability/lexicon/`: the collection ingest (`sources.py`, `crp_ingest.py`, `crp_seat.py`),
extraction (`extract.py`, `census.py`), hierarchy construction (`build_level.py`, `repair.py`,
`level_naming.py`, `hierarchy.py`; procedure in `PIPELINE.md`), external validation
(`external_gold_*.py`), instruments (`build_register_bank.py`, `axes_judge_*.py`,
`prereg13_metaphoricity.py`, `latinate_detector.py`), saturation and frame calibration
(`frame_calibration_airules.py`, `coverage_census.py`, `agreement_vs_missing_mass.py`), and the
experiment builders, runners and analyses (`prereg15v2_*.py`, `prereg25_*.py`, `prereg26_*.py`,
`prereg28_*.py`, `prereg29_*.py`, `prereg30_*.py`, `naming_elicitation.py`), plus the figure scripts
(`fig_*.py`). Judge calls go through the Codex CLI (GPT-5.6), Claude Code (Claude Sonnet 5), vLLM
(Gemma-4-31B, Llama-3.1-8B) and the Z.ai API (GLM-4.7); the harness used for each run is recorded in
`HARNESS.txt` inside the experiment folders. Set `ROOT` at the top of each script to the unzipped
`data` folder. Python 3.11; package list in `methods/requirements.txt`.
The web search agents that fetched the pages were LLM agents driven from the seeded queries recorded
in the wave logs.
