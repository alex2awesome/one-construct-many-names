# Author-lexicon census: 4-domain expansion + verification (2026-07-09)

Canonical record of the census expansion (humor / creative-writing / news-homepages /
math-stackexchange) and the verification pass that followed. **Everything here must be re-run
when the L0→R3 groupings are rebuilt** — see the rerun runbook at the bottom. All analysis
methods live in this package (`dialect.py`, `census_checks.py`, `census.py` via `run_lexicon.py`);
this file records the numbers those methods produced against the 2026-07-06 partitions.

## Instruments and certification

- Extraction: GLM-4.7 verbatim author-lexicon extraction (`extract.py`), quote must appear
  verbatim-normalized in source, key_terms ⊆ source. 8 blinded anchors in every batch.
- Extraction runs (all anchor-certified, pass_rate 1.00):
  humor 5,885 keys (Jul 6, 0 anchors — retroactively certified by CW run) · CW 4,958 (91.4% ok)
  · **news-homepages 3,018 (89.5% ok, 317 rej)** · **math-stackexchange 5,253 (93.5% ok, 340 rej)**.
- Construct partition: `outputs/lexicon/partition_<task>.json` (judge-grounded repaired L0
  concept grain, built Jul 6 by `run_lexicon partition`). R1 grain (humor/CW only):
  `outputs/lexicon/codability/partition_key2R1_<task>.json`.
- Irreplaceable inputs (GLM quota to regenerate, ~16MB total, gitignored):
  `outputs/lexicon/extract_<task>_glm-4.7.jsonl` ×4. Consider `git add -f` or a backup copy.

## Census (concept grain; `run_lexicon census`; outputs/lexicon/census_<task>.json)

| domain | n_keys | concepts (multi-src) | unnamed | agreement | entropy | synonymy |
|---|---|---|---|---|---|---|
| humor | 5,885 | 2,449 (586) | **.684** | .468 | 1.39 b | .925 |
| creative-writing | 4,958 | 2,155 (543) | .550 | .479 | 1.41 b | .902 |
| news-homepages | 3,030 | 1,062 (246) | .552 | .487 | 1.45 b | .915 |
| math-stackexchange | 5,265 | 1,531 (327) | .554 | **.526** | **1.18 b** | .872 |

Readings: (1) unnamed rate splits humor (.68) vs everyone else (~.55 dead heat) — the
named-jargon hypothesis for journalism was half-right (names more than humor, not more than CW).
(2) math is the most conventionalized lexicon *conditional on naming* (highest agreement,
lowest entropy). (3) R1-grain census (humor/CW, `codability/census_<task>_R1.json`): unnamed
.694/.624, agreement .351/.233, entropy 2.51/3.57 b — grain changes levels, not orderings.
Scope: the census universe is author-articulated (explicit, found=true) criteria only —
"unnamed" = explicit-but-unlexicalized, NOT tacit. No TASTE/CRAFT/MECH slicing (types exist
only per-R1 humor/CW).

## Dialect contrast (`dialect.py`) — ★ MIRROR CONFOUND, partial retraction

Construct-matched author key-term Jaccard, within vs cross coarse sub-community bucket,
source→bucket permutation null. **Unguarded numbers are inflated 4–8× by mirrored canonical
texts** (SPJ/Reuters ethics codes, Aristotle's Poetics in CW drama_stage, Lean/mathlib style
guides; 62–88% of quote-mirror pairs are same-bucket; top pairs jac=1.0 verbatim). Guard =
drop within-construct pairs with quote token-Jaccard ≥ .5.

Concept grain:

| domain | unguarded Δ (p) | mirror≥.5 Δ (p) | mirror≥.3 Δ (p) | verdict |
|---|---|---|---|---|
| news-homepages | +.0547 (.000) | +.0037 (.12) | +.0030 (.11) | **DEAD** — mirror artifact |
| math-stackexchange | +.0354 (.001) | +.0094 (.07) | +.0033 (.19) | **DEAD** (marginal) |
| creative-writing | +.0418 (.000) | +.0120 (.002) | +.0022 (.17) | survives at .5 |
| humor | +.0260 (.000) | +.0072 (.002) | +.0058 (.004) | **survives both** |

R1 grain (the previously published instrument): humor +.0533 → guarded **+.0097 (p=0)**;
CW +.0470 → guarded **+.0065 (p=0)** — both survive both thresholds ⇒ the published humor/CW
dialect claims stand at 5–8× smaller amplitude. Journalism/math show shared-canonical-text
adoption (a community phenomenon, but not lexical dialect). The .3 threshold is aggressive
(stopword tokens count), weight the .5 arm.

Bucket provenance: humor/CW buckets from `census_strata_buckets.py`; journalism buckets written
2026-07-09; math buckets are **v2** (v1 left 61% in 'other' and gave Δ+.0153 p=.049; the rewrite
followed a peek at v1's p — disclosed, justified by the objective coverage defect).

## Unnamed-rate robustness (`census_checks.py`) — VERIFIES

| domain | headline | junk-excluded | mirror-collapsed | by size bin (2 / 3-4 / 5-9 / 10+) |
|---|---|---|---|---|
| humor | .684 | .688 | .693 | .71 / .67 / .69 / .66 |
| creative-writing | .550 | .553 | .546 | .60 / .51 / .53 / .50 |
| news-homepages | .552 | .552 | .545 | .56 / .54 / .59 / .53 |
| math-stackexchange | .554 | .563 | .564 | .57 / .54 / .54 / .49 |

Humor ≥ .64 in every size bin; median construct size 3 in all domains; all guards move ≤ .01.
(Note: `census_checks.py` groups best-per-source, so bin values can differ ≤.01 from
census_<task>.json-derived numbers, which pool all records per source.)

## Lessons locked

1. **URL-level independence is insufficient for any lexical-overlap statistic.** Normalized
   source URL was the census independence unit; mirrored canonical texts defeat it. Any
   pairwise surface-overlap measure needs a text-reuse (quote-overlap) guard first — this
   applies to every future dialect/agreement readout, at every grain, with any similarity
   measure (semantic measures score copied text as perfectly similar too).
2. Bucket coverage is power: check the 'other' fraction (should be ≲ 35%) before reading p.
3. Measurement upgrades discussed (pending sign-off, see session 2026-07-09): classifier-AUC
   bucket probe (translationese-detection design, threshold-free), Fightin' Words log-odds
   term lists, two-axis lexical-vs-semantic contrast (BERTScore/WMD-style) to separate dialect
   from sub-concept heterogeneity.

## Dialect battery (2026-07-09 pm, user-approved expansion; `dialect_battery.py`)

Five literature-standard instruments + exact Jaccard on one mirror-guarded footing (see module
docstring for designs). Model instruments (classifier/FW/LM) run on globally mirror-deduped,
grab-bag-free records; their null is bucket labels permuted WITHIN construct with full refit —
it preserves all topic→bucket structure, so null AUC sits at .68-.74 and only form-beyond-topic
clears it. p=.005 = 0/200 permutations ≥ observed (resolution floor).

Concept grain (outputs/lexicon/dialect_battery_<task>.json):

| instrument | humor | creative-writing | news-homepages | math-se |
|---|---|---|---|---|
| classifier AUC (null) | .785 (.710) ✓ | .781 (.707) ✓ | .797 (.716) ✓ | .811 (.743) ✓ |
| — strict-mirror arm | .771 (.690) ✓ | .766 (.698) ✓ | .771 (.684) ✓ | .790 (.723) ✓ |
| community-LM bits (null) | +.66 (.36) ✓ | +.62 (.39) ✓ | +.65 (.40) ✓ | +.64 (.39) ✓ |
| — strict-mirror arm | +.57 (.29) ✓ | +.54 (.35) ✓ | +.49 (.27) ✓ | +.59 (.37) ✓ |
| exact Jaccard Δ | +.0072 p=.002 | +.0107 p=.002 | +.0037 p=.11 ✗ | +.0096 p=.07 ✗ |
| chrF3 Δ | +.0088 p=.005 | +.0208 p=.000 | +.0093 p=.053 | +.0151 p=.022 ✓ |
| semantic Δ (bge) | +.0070 p=.023 | +.0189 p=.000 | +.0110 p=.024 | +.0091 p=.09 |
| semantic, disjoint pairs | +.0016 p=.30 | +.0125 p=.001 | +.0100 p=.018 | +.0066 p=.16 |

✓/✗ at α=.05. Strict-mirror arm (dialect_battery_<task>_strictmirror.json) = drop EVERY source
in any within-construct quote-mirror pair (journalism loses 120 sources) — all classifier/LM
results survive at the permutation floor. R1 grain (humor/CW): classifier .772/.772 vs nulls
.553/.528 (much lower nulls — R1 constructs are bucket-mixed), Jaccard +.0097 p=0 / +.0052
p=.069, semantic-disjoint POSITIVE both (+.016/+.015, p≤.002).

**Synthesis (instrument ladder from strict surface → soft form → meaning):**
1. Community-distinctive FORM exists in ALL FOUR domains (classifier + LM, mirror-strict robust):
   sub-communities are identifiable from how they phrase the same constructs. The pairwise
   exact-term instrument is the least sensitive — it needs per-pair convergence on identical
   canonical terms and only fires for humor/CW.
2. chrF3 adds math (morphological variants: word-Jaccard misses "newsworthy/newsworthiness"-
   style kinship; math Δ+.0151 p=.022).
3. The semantic two-axis SPLITS the interpretation: humor concept-grain = the classic dialect
   signature (lexical gap WITH disjoint-pairs semantic null p=.30 — same meaning, different
   words). CW, journalism, and R1-grain humor keep a semantic gap on lexically-DISJOINT pairs
   ⇒ their within-community similarity is partly sub-concept/aspect differentiation (communities
   emphasize different facets of the shared construct), not only word choice.
4. Fightin' Words community lexicons are face-valid: improv "game / game of the scene /
   heightening", joke_writing "punchline / setup / surprise / violation", drama_stage "pity /
   fear / catharsis / spectacle", newsworthiness "timeliness / framing / conflict / impact",
   exposition "economy / elegance / simplicity" (descriptive instrument; topic+dialect mixed).

## Battery convergence (`battery_convergence.py` → dialect_battery_convergence.json)

Do the instruments agree on WHERE dialect lives? Construct level (per-construct deltas, pooled
n=715/539 with within-task rank normalization) and bucket level (per-community signal, pooled
n=45/37):

| level | pair | pooled ρ | read |
|---|---|---|---|
| construct | jaccard ~ chrf3 | +.33 | surface family coheres |
| construct | chrf3 ~ semantic | +.56 | soft family coheres |
| construct | jaccard ~ semantic | +.25 | " |
| construct | any pairwise ~ LM advantage | **+.01 to +.08** | families ~orthogonal |
| bucket | classifier AUC ~ LM advantage | +.56 (humor .81, journ .93, math .61, CW .10) | register instruments agree |
| bucket | classifier/LM ~ pairwise jaccard | +.29 / +.25 | weak positive |
| bucket | FW n_sig ~ classifier/LM | −.28 / −.35 | SIZE CONFOUND — see below |

**Reading: the battery measures TWO distinct phenomena, and they are near-orthogonal at the
construct level.** (1) *Lexical convergence dialect* — same-community authors converging on
shared terms for a specific construct (pairwise family: jaccard/chrf3/semantic deltas, which
inter-correlate .25–.56). (2) *Register dialect* — a record being written in its community's
general vocabulary (classifier + community-LM; the LM's construct-excluded training makes this
explicit — it can only use community-general language). Construct-level ρ between families ≈ 0;
bucket-level +.25–.29 (weak). This refines the domain verdicts: **all four domains have register
dialect; only humor/CW additionally have per-construct lexical convergence.** CW is the
bucket-level outlier (15 small buckets, clf~lm only +.10). FW n_sig is count-powered (bigger
bucket → more significant terms) and anti-correlates with per-class AUC — the agreement~size
lesson again; do not use FW term counts as a signal-strength measure, only the term lists.

## Rerun runbook (after the L0→R3 rebuild)

The new groupings replace `partition_<task>.json` (and eventually new key→R1 maps). Then, per
task (all CPU, no GLM — extractions are reused as-is):

```bash
# 1. rebuild judge-grounded partitions (writes outputs/lexicon/partition_<task>.json + meta)
python -m methods.codability.lexicon.run_lexicon partition --tasks <tasks>

# 2. census (writes outputs/lexicon/census_<task>.json)
python -m methods.codability.lexicon.run_lexicon census --tasks <tasks>

# 3. dialect with guards — REPORT THE GUARDED ARM, unguarded is known-inflated
python -m methods.codability.lexicon.dialect <tasks> --arms
python -m methods.codability.lexicon.dialect humor,creative-writing --r1   # when new R1 maps land
#   (point --partition at the new key→R1 maps if the path changes)

# 4. unnamed-rate robustness
python -m methods.codability.lexicon.census_checks <tasks>

# 5. dialect battery (classifier/FW/LM/semantic/chrF; ~30-60 min for 4 tasks, CPU)
python -m methods.codability.lexicon.dialect_battery <tasks> [--r1]
python -m methods.codability.lexicon.dialect_battery <tasks> --strict   # mirror-strict arm

# 6. instrument convergence (construct/bucket-level correlation structure)
python -m methods.codability.lexicon.battery_convergence
```

New tasks additionally need: a BUCKETS entry in `dialect.py` (inspect the subtask_short
distribution first; keep 'other' ≲ 35%) and contexts + extraction via `run_lexicon build/glm`
with anchors blended (see `extract_input_<task>.jsonl` construction, session 2026-07-08).

Archaeology: exploratory copies of these scripts (plus the R1-era one-offs
`census_strata_buckets.py`, `subtask_dialect.py`, fidelity/extractiveness checks) are in
`outputs/lexicon/codability_audit/` — this package is canonical from 2026-07-09 on.
