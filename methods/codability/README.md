# Codability of preferences — the stratified Codability Profile

Implementation of `notes/2026-07-01__codability-audit-and-proposal.md` (Brown–Lenneberg codability,
upgraded to behavioral communication-success measures over the R1/R2/R3 metric clusters), for the
anthropological/linguistic study of preference articulability (`notes/2026-07-01__articulability-anthropology-reframe.md`,
theory `notes/2026-06-18__prompt-optimality-theory.md` §12.6).

**Two binding constraints, enforced throughout:**

1. **No embedding spaces in any reported number.** Everything is verdict-space (Cohen's κ on
   binarized executor verdicts), information-theoretic functionals of verdicts, categorical judge
   decisions, or corpus metadata. (Learned Euclidean geometry shifts with model training — the one
   residual embedding dependence is the R2/R3 cluster *definitions* themselves; the transfer matrix
   here re-validates membership behaviorally and flags FRAGMENTED clusters as a by-product.)
2. **Codability is never measured by asking a model whether something is describable** (the
   retraction lesson, `project_subtask_codability_result`): always by whether an articulation,
   **re-executed**, reproduces the verdicts.

## The subfield control

Pooling probes over subfields (CW: horror/adventure/romance/…) makes an *indexical* metric — codable
given the frame, different realization per frame — read as tacit (mixture masquerading as tacitness,
the Simpson pattern). The design therefore stratifies everything:

```
Δ_context  = mean_g R_g − R_global        — INDEXICALITY: codable given the frame
A_g        = T_g − R_g                    — the within-frame articulation gap
R_ig       = μ + a_i + b_g + (ab)_ig      — a_i = subfield-ADJUSTED codability of metric i,
                                            Var[(ab)] = indexicality variance
M[g→g']    = κ( exec(r_g, ·), m̄ )  on held-out g'   — the transfer matrix (diag = R_g);
                                            row structure = the heterogeneity read
```

## The ordinal levels (replacing a single scalar)

| level | name | operational criterion |
|---|---|---|
| **L0** | COMPILABLE | code implementation converges with judge; recovery via program |
| **L1** | UNIVERSALLY CODABLE | `R_global ≈ R_g ≈ T_g` ∀g; one short rubric; high κ across families |
| **L2** | INDEXICALLY CODABLE | `R_g ≈ T_g` but `R_global ≪ mean R_g`; transfer diagonal-dominant |
| **L3** | OSTENSIVELY TRANSMISSIBLE | rules plateau below `T_g`, exemplars close the gap |
| **L4** | TACIT-WITHIN-FRAME | `R_g ≪ T_g` in every defined stratum, exemplars included, `T_g` materially > 0, and a preregistered articulation-search horizon reached; explicitly within that tested horizon |

Exclusion **gates** (not levels): UNDEFINED (probe imbalance), FORM-DOMINATED, NO-SIGNAL,
FRAGMENTED (not one concept — routed to the re-clustering audit), UNDERSAMPLED. The withdrawn §12.6
epsilon diagnostic cannot waive the sampling gate or certify L4.

L1→L4 is the anthropological gradient: fully tellable → tellable-in-context → showable → only
learnable by immersion. The per-task **codability map** (fraction of metrics at each level) is the
headline deliverable; `Δ_context` per task is the indexicality of that community's evaluative language.

## Modules

| module | contents |
|---|---|
| `strata.py` | metadata/categorical-judge stratum assignment (never topic models/embeddings), frozen stratified splits, probe-imbalance guard (codability *undefined* ≠ low) |
| `transfer.py` | verdict-space κ, the transfer matrix `M[g→g']`, diagonal dominance, 2-block structure detection (FRAGMENTED evidence) |
| `decompose.py` | `Δ_context`, per-stratum gaps `A_g`, the two-way mixed model with bootstrap CIs, reliability attenuation correction |
| `levels.py` | the gate + L0–L4 verdict router (`profile_level`) over a per-metric profile dict |
| `controls.py` | the four planted controls of §4.3 (universal / genre-indexed / exemplar-only / noise) + a fragmented world — **mandatory positive controls before any real claim**; the genre-indexed control landing L2-not-L4 is the proof the design separates indexicality from tacitness |
| `run_codability.py` | driver: `--controls` (offline discipline check) or assembly of live profiles from recon_channel + per-stratum `value_certificate` outputs |
| `tests/` | planted-ground-truth tests, all CPU/offline |

## Live wiring (sk3)

Per metric × stratum, the live quantities come from existing instruments — this package only
assembles and adjudicates:

- `R_g`, `R_global`, exemplar channel → `metric_implementer/recon_channel.py` (induce on stratum-g
  pairs only, execute on held-out g′; `mode=free` vs few-shot for the ostension channel)
- `T_g` → per-stratum test–retest ceiling of `m̄_ω` (orbit-averaged target,
  `alpha_probe.orbit_metric_verdict`); task-level: per-stratum deconfounded dense ceiling `C_g`
- `eps_frac_g` → descriptive-only `value_certificate` flux diagnostic. It never gates a level;
  `upper_bound_valid=False` is preserved. L4 instead requires separately recorded,
  preregistered `search_horizon_reached_g` evidence.
- form gate → `alpha_probe.form_invariance`; code convergence → scorecard #8
- stratum tags → corpus metadata, else `strata.make_stratum_judge` (GLM — **quota is binding, be
  sparing**: one batched call per corpus, cache the labels)

Pilot per §4.4: CW, 3 R3 metrics (craft / generic-taste / genre-specific) × 4 genre strata ×
~120 probes/stratum × 2 reconstructor families; share the stratified probe pool across metrics of
the same task (the stratum count is the main cost multiplier).

---

## The text-only leg: `lexicon/` (2026-07-06, notes/2026-07-06__metric-lexicon-codability-census.md)

> **L0→R3 hierarchy reconstruction pipeline of record:** `lexicon/PIPELINE.md` (stages, gates,
> reproduce commands, and the empirical caveats — arbiter-truth generosity, net-band cap, TF-IDF≥BGE).
> Runbook/results: `notes/2026-07-06__hierarchy-reconstruction-ledger.md`; rules: `BEST-PRACTICES.md`.

The OBSERVATIONAL companion to the behavioral profile above — no behavior, no re-execution: how
the original human authors wrote the metrics. Key-term extraction from ORIGINAL source docs
(verbatim-validated, anchors in every batch), judge-grounded hierarchy trust audit ("beyond v6"),
and the conventionalization census (naming agreement / name entropy / synonymy rate / per-R3
subtree lexicalizations), subfield-controlled via `strata.py`/`decompose.py`.

| module | role |
|---|---|
| `lexicon/sources.py` | canon key → original doc text (raw HTML / claude-parsed md) + source-URL identity + windowed extraction contexts |
| `lexicon/anchors.py` | blinded planted anchors + scoring (catches synonym-substituting annotators) |
| `lexicon/extract.py` | extraction prompt, mechanical quote/term validation, GLM batch runner (retry-by-seed) |
| `lexicon/compare.py` | GLM-4.7 vs Sonnet equivalence gate (user rule: GLM preferred iff equivalent) |
| `lexicon/audit.py` | trusted-edge tiers (T1 fresh-majority / T2 triangle / T3 untrusted), chain-proof repaired partition, leaf FP/FN certificates, lexeme-overlap under-merge net |
| `lexicon/census.py` | naming agreement, name entropy, synonymy rate, R3-subtree census, strata decomposition |
| `lexicon/run_lexicon.py` | driver: build / sample / glm / ingest / partition / census |

Superseded-for-this-purpose legacy (kept, read-only): `scripts/cluster_canon.py`,
`scripts/leaf_name_clusters.py`, `scripts/cluster_pairs.py` (embedding/canonical-form era);
`scripts/adopt_tail_clusters.py` (its ≥2-edge + chain-proof rules live on in `lexicon/audit.py`);
sk3 `sk3_match_pipeline.py` family artifacts consumed via `.cache/norm_embed/`.

Discipline: judge decisions PARTITION, never score (retraction lesson); no verdict-trained
encoder or blend-cluster co-membership in any reported number (S⊥R); independence unit = source
URL; repairs append-only.
