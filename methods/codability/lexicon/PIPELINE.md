# L0→R3 concept-hierarchy reconstruction — pipeline of record

Consolidated description of the *current* process that reconstructs the metric-statement hierarchy
(L0 same-criterion → R1 same-construct → R2 same-theme → R3 same-category) with measured
recall/precision at every level, per task. This is the method description; the two living records are:

- **Runbook + chronological results:** `notes/2026-07-06__hierarchy-reconstruction-ledger.md` (read
  it before resuming — every improvised deviation so far has lost to it).
- **Discipline/rules (do & don't):** `BEST-PRACTICES.md` → *"hierarchy L0→R3 relabeling (codability)"*.

Engine: `methods/codability/lexicon/build_level.py` (R-levels), `repair.py` (L0), `level_naming.py`
(group naming between levels). All artifacts live in `outputs/lexicon/`. Reconstruction-only: judges
PARTITION, never see labels; no human subjects.

---

## 0. Data model

- **Node at level L** = a *named group* from level L−1. L0 nodes are the deduped rubric criteria;
  R1 nodes are L0 clusters; R2 nodes are R1 groups; etc. `nodes_from_level(task, level)` returns
  `[{node_id, name, gloss}]` and composes membership down to the original L0 keys for later census.
- **Relation per level** (`RELATIONS` in `build_level.py`), each strictly broader than the last:
  - **L0** = *same criterion* — would give the same verdict on real work; differ only in wording.
  - **R1** = *same construct* — facets of ONE underlying quality; a reader treats them as one dimension.
  - **R2** = *same theme* — belong to one broad evaluative area/family.
  - **R3** = *same category* — top-level subsumption.
- **Final measurement** at each level = a post-candidate-freeze sample of node pairs independently
  labeled 0/1/2 by Sonnet and GPT-5, with a third frontier pass only on disagreements. Preserve each
  judge's label as well as the adjudicated label; no single model family defines truth. Historical
  preselected evals remain immutable secondary tests, not the primary promotion instrument.
  `pair_id = sha1(sorted node keys joined by "||")[:16]`.
- **Gemma scorer input** = `pair_id`, task, level, protocol ID, the two named/glossed node texts,
  and source-node hashes. **Output** = DIFFERENT/RELATED/SAME probabilities in both input orders,
  winning label, order-consistency flag, adapter hash, and protocol hash. It emits no cluster ID,
  hierarchy name, free-form rationale, or ground-truth label.

## 1. Stage order (load-bearing)

```
L0 repair → freeze + name
  → R1 (freeze lineage → retrieve → Gemma score → candidate freeze → audit → promote → name)
  → R2 (same, from promoted R1)
  → R3 (same, from promoted R2)
```
Building an R-level on *unrepaired* L0 measures the repair debt, not the relation. Each level prints
BOTH recall and precision; neither may silently sag (L0 repair is a deliberate recall-ward move whose
precision cost compounds up-tree). R2 cannot begin until R1 is promoted and manifest-frozen; R3
cannot begin until R2 is. Parent drift is an error, never an implicit rebuild.

## 2. L0 build + repair (`repair.py`)

L0 starts from the deduped rubric partition and is *repaired* (under-merge recovery) to dryness:

1. **Candidate net** — union of three signals so paraphrases aren't missed:
   `TF-IDF(min_df=2) ∪ shared-name ∪ v6-BGE semantic`. Measured ceiling 1.0 on adjudicated eval.
2. **Screen** (permissive: keep at ≥1) → **Confirm** (independent LLM, strict `==2`, frozen
   `CONFIRM_PROTOCOL_L0_V2.txt` = the L0 "same criterion" relation) → **≥2-edge STAR merge**.
   Historical repairs used `CONFIRM_PROTOCOL_R1.txt`, whose wording was the broader R1
   "same construct" relation despite earlier notes calling it L0. Preserve those artifacts but do
   not reuse that prompt; the cross-task L0 coherence audit measures the resulting precision debt.
3. **STAR merge invariant** — a head must never also become a tail (else A→B, B→C chains leave A→B
   un-realized). `apply_merges` locks BOTH tails and heads (`heads` set); **union-find is forbidden**
   (single-bridge chaining). Vote parsing is strict-int (`type(s) is int and s in (0,1,2)` — bool is
   an int subclass, so `True==1` would otherwise slip through).
4. **Widen-to-dryness** — sweep candidate bands (rank ~2500→8000, deeper if still wet) until yield
   dries; miss-by-band triage first; net-vintage parity. Only then freeze.
5. Write `partition_<task>_L0vN.json` (NEW file each version; never overwrite), `score_vs_truth`,
   ledger row. Applying a new L0vN does NOT rebuild R1/R2/R3 — `apply_merges` reuses existing cluster
   ids, so merged nodes inherit their parents by composition; upper levels just GROW → a cheap rescore.

**Exception — precision regroup/splits:** `l0_regroup.py` can split an overbroad L0v3 cluster back
into strict same-criterion groups of its v6 source clusters. A split changes the parent node inventory;
R1 and all descendants MUST be rebuilt against the new manifest-frozen L0v4. Never use the cheap
composition rule for splits.

## 3. R-level build (`build_level.py`) — the ONLY validated R-build

For each level R1→R3, in order:

1. **Freeze lineage.** Run `lineage-freeze` before emitting or applying a build. It recursively pins
   the exact parent partition and semantic-name artifacts for every ancestor by path and SHA-256.
   Missing, partial, mismatched, or drifted R1/R2/R3 manifests fail closed.
2. **Retrieve a high-recall proposal net.** Semantic embeddings and/or lexical kNN expose economical
   candidate pairs. Retrieval proposes what to score; it never supplies a semantic judgment. Record
   the retriever, node inventory, parameters, and pair IDs in the candidate manifest.
3. **Score with the level-specific Gemma LoRA.** Use the pooled R1, R2-v2.1, or R3 scorer as the
   high-volume builder. Score both A→B and B→A and retain all DIFFERENT/RELATED/SAME probabilities,
   winning labels, order consistency, model hash, and protocol hash. Task LoRAs are exceptions that
   require a fresh paired audit; they are not selected because their training loss is lower.
4. **`apply_pairwise` writes an immutable candidate**, never canonical state. It is fail-closed:
   - **Completeness check**: every non-anchor net pair must have ≥1 well-formed vote (by pair_id, not
     shard count) → else `IncompleteVotesError`.
   - **Anchor gate**: positive-anchor accuracy ≥ 0.8, negative ≥ 0.9 → else `AnchorGateFailure`.
     Anchors never contribute a build edge regardless of outcome. `exclude_from_gate={pids}` drops a
     *confirmed-bad-gold* anchor from the gate math ONLY (it stays edge-blocked) — document every use.
   - **Louvain** community detection (`res=1.0`) on the verified-SAME graph. **No fallback** (a silent
     connected-components fallback once rotted R1 precision .80→.43 undetected). The default
     `related_weight=0` exactly preserves this hard-edge graph. A preregistered positive value may
     use LLM score-1 RELATED pairs as weaker weighted structure evidence, never as SAME labels; that
     candidate still requires the independent uniform LLM precision audit and every >30-member
     whole-group certification. `--output-path` is mandatory and canonical-looking destinations are
     rejected.
5. **Freeze the candidate, then draw its final audit.** The candidate partition, parent, node
   inventory, edge probabilities, thresholds, and graph configuration receive SHA-256 pins before
   any final pair is sampled. The audit stratifies co-clustered pairs, nearest cross-cluster
   boundaries, high-Gemma-SAME cross-cluster pairs, and random pairs. Sampling continues until about
   150 adjudicated SAME pairs are obtained or the preregistered cap is reached; retain sampling
   weights for population estimates.
6. **Frontier LLMs measure and certify.** Sonnet and GPT-5 judge independently; a third frontier pass
   adjudicates disagreements. Near-threshold and order-inconsistent Gemma pairs may be sent to these
   judges for a *new candidate*, but final-audit labels can never mutate the candidate they evaluate.
   Every cluster over 30 members requires a complete-member Sonnet/GPT-5 certification artifact.
7. **Central score and promotion.** `build_level.score()` is the sole scorer allowed to place a
   number in the ledger. Record the candidate, parent, audit, protocol, judge, and scorer hashes plus
   Cohen κ, SAME precision/recall/F1, macro-F1, confidence intervals, and sampling strata. Do not
   report a rescue module's private definition of “precision,” and do not select on the deprecated
   recall-only quantity. `partition-promote` is the sole canonical writer: it validates exact node
   coverage, atomically replaces the canonical file, and banks the previous canonical by content
   hash when `--replace-canonical` is explicitly supplied.
8. **Name only the promoted partition.** `level_naming` emits complete member inventories for the
   Sonnet naming pass and writes `node_names_<task>_<level>.json`; singletons inherit their member's
   name. Freeze these names before building the next rung.

**No confirm stage at R-levels** — the R1 bridge-confirm judged bridges at ≈chance (60% anchor) and
killed 94% of them (recall .673→.537 flat precision); REVERTED. (The L0 screen→confirm is a different,
sanctioned confirm — don't conflate.)

**Gemma's authority boundary is strict.** Gemma retrieves/scores enough edges to build economically.
It does not define truth, certify a cluster, name a group, or support a reported comparison by itself.
All final measurements and model/partition comparisons are based on persisted frontier-LLM judgments.

## 4. Certification gates (measured at EVERY level)

- **Scorer gate:** on frozen held-out LLM labels, SAME precision and recall must each be ≥ .50.
  Calibrate the edge threshold on development data to maximize recall subject to a target SAME
  precision ≥ .60; never tune it on the final audit.
- **Retrieval gate:** after the audit is labeled, report what fraction of adjudicated SAME pairs the
  retriever exposed. This diagnoses a missed-candidate problem but does not authorize inserting those
  audit edges into the evaluated candidate.
- **Partition gate:** promote only when paired ΔCohen κ and ΔSAME-F1 beat the current canonical on
  the same audit with a 95% paired interval above zero, neither precision nor recall regresses by
  >.02, and cold-concept performance does not materially regress. A tie retains the current
  canonical. A task LoRA additionally needs Δκ and ΔSAME-F1 ≥ +.03.
- **Judge-ceiling gate:** “at ceiling” requires the absolute precision/recall floors and a κ deficit
  no larger than .03 from the reliable-judge benchmark, with paired uncertainty excluding a material
  deficit. Low test–retest reliability is a finding, not permission to quote an unstable point score.
- **Large-cluster gate:** every group over 30 members needs its complete-member independent
  Sonnet/GPT-5 certification before promotion. Size alone is not evidence that a cluster is wrong.

## 5. Judge placement & throughput (fixed by measurement, not cost)

- Gemma LoRAs on **sk2 only** are the high-volume R-level builders. sk3 is out of scope. The pooled
  level scorer is the default; task-specific continuations require the paired promotion evidence in
  §4. **GLM remains L0-only** because it over-merged R1 in the historical runs.
- Sonnet and GPT-5 independently label calibration, disputed boundaries, cluster certifications, and
  final comparisons. A third frontier pass sees only their disagreements. Keep individual and
  adjudicated labels; never silently replace one pass with a consensus field.
- **Blinded anchors in every judging batch** are the tripwire — every failure caught so far (resumed-
  judge drift; 60%-anchor confirm) was caught by anchors. Use the 0/1/2 scale + calibrated exemplars.
- TERSE output (score only) validated at ~400 pairs/shard; consolidated multi-file agents hit a 64k
  output ceiling → 1 file/agent, no reasoning field.
- **JUDGE-AGENT EFFICIENCY (2026-07-11) — go straight to scoring, no overhead.** Observed waste: agents
  spending 20-40 tool calls on self-validation, re-reading, cross-checking pair_ids, and even spawning
  their own subagents (one caused a shard content-mixup). Standard judge prompt now mandates: (a) do the
  work YOURSELF — never spawn/delegate to subagents; (b) exactly ONE Read per input file and ONE Write
  per output file; (c) process files ONE AT A TIME (read file → judge all lines → write file → next),
  never hold multiple files' content together (juggling caused the mixup); (d) do NOT re-read, validate,
  verify, count, or cross-check your output — the orchestrator runs the completeness + per-shard 0-fraction
  distribution checks downstream, so agent self-QC is pure wasted latency; (e) no repo search/exploration;
  (f) snap-judge each pair, terse {pair_id,score} only. This cut ~4-5× the tool calls per shard.
  The scale lever is now the persisted Gemma builder: frontier calls are reserved for calibration,
  uncertain boundaries, certification, and final measurement rather than repeated full-net judging.

## 6. Reproduce (news R1 example)

```bash
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level lineage-freeze \
  --task news-homepages --level R1

# Retrieval + pooled Gemma-R1 inference writes the frozen candidate-pair scores consumed below.
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level pairwise-apply \
  --task news-homepages --level R1 \
  --output-path outputs/lexicon/partition_news-homepages_R1_candidate_gemma_pooled_v1.json

# Freeze the candidate hash, draw/register the audit afterward, persist both judge passes, and set
# ADJUDICATED_VOTES to the corresponding consensus/third-pass vote artifact.
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level score \
  --task news-homepages --level R1 \
  --partition-path outputs/lexicon/partition_news-homepages_R1_candidate_gemma_pooled_v1.json \
  --arbiter-vote-path "$ADJUDICATED_VOTES"

# Run only after the central ledger's paired promotion gates and >30-member certifications pass.
PYTHONPATH=. python3 -m methods.codability.lexicon.build_level partition-promote \
  --task news-homepages --level R1 \
  --partition-path outputs/lexicon/partition_news-homepages_R1_candidate_gemma_pooled_v1.json \
  --replace-canonical
# Then name the promoted R1, freeze those names, and begin R2.
```

## 7. Historical preselected-eval workflow (legacy, do not resume)

Runs from 2026-07-10 through 2026-07-12 selected and labeled the eval before construction, then
deleted every eval pair from `emit_verify_net`. Preserve those payloads, votes, partitions, and notes
for reproducibility, but do not use that workflow for new candidates. It protected against direct
train/test reuse at the cost of grading the build on pairs it was forbidden to propose: “recall” then
measured whether Louvain happened to reconnect an excluded pair through transitive closure. Sparse R2
graphs suffered most, while coarse R3 partitions reconnected pairs almost automatically. The old R2
trough is therefore a mixture of semantic difficulty, retrieval, verifier strictness, and this
measurement design—not an identified property of themes.

The legacy findings below remain useful diagnostics, not current promotion evidence:

- **The single-pass R-level arbiter was systematically generous (over-called SAME).** Cross-family blind
  validation (Codex gpt-5.6-sol, a different family, 120 held-out pairs/task) agrees with the single-
  pass Sonnet arbiter's "same construct = 2" only **28% (news) / 33% (math)** of the time; the verify
  fleet shows the same strictness. ⇒ **low R-level recall is deflated by truth generosity, not (only)
  build failure**; precision stays high because what the strict build merges is clean. A de-noised
  truth (multi-pass / cross-family arbiter) would raise apparent recall — a methodology change that
  needs sign-off. Read every R-recall with this caveat.
- **Legacy transitive recall was net-band-capped as node count grew.** The diagnostic ceiling was
  approximately `f(#nodes, cap)`. math R1
  (2893 nodes, full net 27,826) at cap=9000 ceils recall at **0.623**; widening cap→21,000 lifts it to
  **0.891** (~40 more verify shards). news R1 (1225 nodes) at cap=9000 already ceils at 0.926. Log the
  cap (no silent caps).
- **TF-IDF beat the tested generic BGE retriever on these node reps.** BGE-large kNN on math R1 nodes reached only 0.52 full-
  net ceiling vs TF-IDF's 0.913 (and adds 1/138 truth-SAME pairs TF-IDF misses): short jargon-dense
  criteria collapse into a generic neighborhood under a general-English embedder, losing the
  discriminating rare terms TF-IDF keys on. Only add embedding nets if TF-IDF *measurably* decays.

## 8. Legacy results snapshot (not commensurable with the current audit)

These values came from the preselected-eval/excluded-edge workflow and several historical build
variants. They must not enter the current ledger as headline comparisons or be compared directly
with post-freeze audit estimates. They are retained solely to reproduce earlier notes.

| task | L0v3 | R1 | R2 | R3 |
|---|---|---|---|---|
| news-homepages | .913 / .587 | .477 / .583 | .176 / .417 | .753 / .732 |
| math-stackexchange | .881 / .742 | .438 / .750 | .365 / .547 | .746 / .815 |
| humor | ~.86 | .673 / .689 | .425 / .703 | .649 / .689 (anchored) |
| creative-writing | ~.90 | .620 / .678 | .384 / .682 | .729 / .670 (anchored) |
| peer-review | .850 / .748 | .587 / .677 | .259 / .658 | .756 / .749 (classify-derive) |

**Legacy chance-corrected recall** (historical key `recall_kappa`, retained only as a deprecated alias) —
peer-review: L0v3 .850 → R1 **.576** → R2 **.241** → R3 **.691**. This is
`(recall-p0)/(1-p0)`, NOT Cohen's κ; `score()` now reports actual binary Cohen κ separately plus
Wilson uncertainty and high-sim/random eval strata.

**The legacy R2 run showed a trough, but its cause is not identified by the old ceiling diagnostic.** The
eval set is deliberately TF-IDF enriched and the candidate net uses the same representation. For
peer-review R2, the `.829` aggregate ceiling consists of 157/157 high-similarity positives but only
3/36 random positives. Therefore it does not establish `.829` population findability. The candidate
graph has 1,103 edges (avg degree 5.7, 43 isolates); the LLM verifier retains 291 (avg degree 1.5,
141 isolates). Loss is a mixture of candidate retrieval, conservative LLM verification, forced
disjoint partitioning, and Louvain fragmentation. `recall/ceiling` is not a clean stage-conditional
probability because held-out eval pairs never contribute build edges.

**Historical blind R2 cross-family audit (2026-07-12):** 120 pairs/task, balanced 40/40/40 over Sonnet
scores 0/1/2; independent Codex-family LLM judges saw no Sonnet labels. `P(Codex SAME | Sonnet SAME)`:
math `.525`, creative-writing `.800`, humor `.850`, news `.775`, peer-review `.850`. Binary SAME/not
agreement: `.758/.892/.858/.758/.850`; binary Cohen κ on this deliberately balanced diagnostic sample:
`.424/.752/.691/.491/.675`. Thus R2 positives are reproducible in CW/humor/peer, moderately so in news,
and materially unstable in math. These replace the older nonuniform `.925/.75` shorthand; frozen
payload manifests and complete reports live under `outputs/lexicon/codex_val/*_R2_*reaudit*`.

**Interpretation discipline:** the R1→R2 decline uses the same code family but not a measurement-
invariant instrument: graph exposure, semantic representations, relation transitivity, and verified
edge density all change. R3 classify-derive also avoids fragmentation and its anchored arbiter shares
the proposed taxonomy. Report the observed R2 trough, but do not call it an intrinsic property of
themes until an independent LLM-proposed candidate comparison and an evaluation sample not selected by
the build representation reproduce it. Themes may also overlap; Louvain's disjoint equivalence
partition is itself a hypothesis to test.
Do not update this table with new cells. Current results belong only in the centralized ledger and
must be reproducible by `build_level.score()` from the frozen candidate and persisted final-audit
judgments. A ledger row without partition, parent, audit, protocol, judge, and scorer hashes is invalid.
