# One Construct, Many Names

**Synonyms Yield Divergent Judgements**

Alexander Spangher, Daniel E. Ho, Sanmi Koyejo, Diyi Yang (Stanford University)

Paper: [`paper/Spangher_Ho_Koyejo_Yang__One_Construct_Many_Names__2026-11-10.pdf`](paper/Spangher_Ho_Koyejo_Yang__One_Construct_Many_Names__2026-11-10.pdf), also at https://www.alexander-spangher.com/papers/One_Construct_Many_Names.pdf

LLM judges are guided by prompts, constitutions and rubrics that verbalize normative criteria, and the
same criterion is named differently by different communities. We present a large-scale analysis of
evaluative lexicons across 11 creative domains: 53,413 criteria mined to saturation from online
rubrics, professional guidance and community rules, clustered into surface forms, constructs, themes and
categories. We find that (1) community-written rules are worded less formally than institutional
criteria, (2) LLMs asked to name criteria speak at or above the institutional register, so criteria that
one LLM writes for another arrive in the formal register, and (3) swapping only the name of a construct
between register variants disrupts a judge's rankings while paraphrasing the surrounding prompt with the
name held fixed does not.

## What is here

| Path | Contents |
|---|---|
| `methods/codability/lexicon/` | Collection, extraction, hierarchy construction, instruments, saturation checks, the judge experiments (`prereg*`), and the figure scripts. Procedure in `PIPELINE.md`. |
| `methods/codability/tests/` | Tests for the hierarchy and instrument code. |
| `methods/requirements.txt` | Python 3.11 package list. |
| `DATA.md` | Field-by-field description of the data archive and of the experiment folders. |
| `paper/` | The paper (preprint, November 2026). |

## Data

- **Archive** (`paper1_data.zip`, attached to the GitHub release): the criteria census, hierarchy
  partitions, provenance, instruments, silver-label splits, and the full materials of every judge
  experiment whose runs were persisted (prompts, item sets, every raw judge output, per-construct
  tables). `DATA.md` lists each file.
- **Hugging Face**: https://huggingface.co/datasets/alex2awesome/CreativePreferences holds the same
  census, hierarchy, instruments, source pages and silver labels as parquet tables under
  `metrics/found/<domain>/`, next to the text corpora and preference labels of the companion papers.

To run the analysis code, unzip the archive and set `ROOT` at the top of each script to the unzipped
`data` folder. Judge calls were made through the Codex CLI (GPT-5.6), Claude Code (Claude Sonnet 5),
vLLM (Gemma-4-31B, Llama-3.1-8B) and the Z.ai API (GLM-4.7); the harness used for each run is recorded
in `HARNESS.txt` inside each experiment folder. Account names, hosts and local paths in logs were
replaced by placeholders before release; commenter names in the register corpora are `[redacted]`.

## Citation

```bibtex
@misc{spangher2026oneconstruct,
  title  = {One Construct, Many Names: Synonyms Yield Divergent Judgements},
  author = {Spangher, Alexander and Ho, Daniel E. and Koyejo, Sanmi and Yang, Diyi},
  year   = {2026},
  note   = {Preprint},
  url    = {https://www.alexander-spangher.com/papers/One_Construct_Many_Names.pdf}
}
```
