#!/usr/bin/env python
"""PREREG-23 baselines: what does a similarity model score on the SAME pairs?

The prompts are judged against human gold links in external_gold_score.py. That number is
only interpretable next to what a cheaper method gets on the identical pairs, so we run:
  tfidf   -- character+word TF-IDF cosine. The lexical control. Our protocols instruct the
             judge NOT to infer sameness from shared vocabulary, so this measures how much of
             the link structure is recoverable from surface overlap alone.
  minilm / mpnet / bge -- sentence-embedding cosine, small to large.
  agglom  -- average-linkage agglomerative clustering on embeddings, cut at the GOLD number of
             parents (which the LLM pipeline is never told), scored by ARI. This is the
             partition-level baseline for the partition-level claim.

Every baseline sees exactly the pairs the judges saw, with the same hard/easy strata, so the
AUCs are directly comparable. Cutting agglomerative at the true K is deliberately generous:
the baseline is handed a parameter our pipeline has to infer.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

from .external_gold_score import OUT, auc, boot_ci

MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2",
          "mpnet": "sentence-transformers/all-mpnet-base-v2",
          "bge": "BAAI/bge-large-en-v1.5"}


def load_cells(judged_only=True):
    """Restrict to the pairs a judge actually scored. The judges only saw batch 0 of each
    cell, so running the baselines over the full pair file would compare the two arms on
    different samples and make every delta uninterpretable."""
    cells = defaultdict(list)
    for fn in glob.glob(f"{OUT}/pairs_*.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            cells[(r["corpus"], r["rung"])].append(r)
    if not judged_only:
        return cells
    from .external_gold_score import score_all  # reuse the legacy-id shim + anchor gate
    import hashlib
    judged = defaultdict(set)
    manifest = json.load(open(f"{OUT}/batch_manifest.json"))
    for m in manifest:
        if not os.path.exists(m["votes"]):
            continue
        for l in open(m["votes"]):
            try:
                judged[(m["corpus"], m["rung"])].add(json.loads(l)["pair_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    out = {}
    for key, rows in cells.items():
        keep = []
        for r in rows:
            legacy = hashlib.sha1(
                f"{r['corpus']}|{r['rung']}|{r['a_id']}|{r['b_id']}".encode()).hexdigest()[:16]
            if r["pair_id"] in judged[key] or legacy in judged[key]:
                keep.append(r)
        if len(keep) >= 30:
            out[key] = keep
    return out


def tfidf_scores(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [r["a"] for r in rows] + [r["b"] for r in rows]
    n = len(rows)
    out = []
    for kw in ({"analyzer": "word", "ngram_range": (1, 2)},
               {"analyzer": "char_wb", "ngram_range": (3, 5)}):
        V = TfidfVectorizer(sublinear_tf=True, min_df=1, **kw).fit_transform(texts)
        A, B = V[:n], V[n:]
        out.append(np.asarray(A.multiply(B).sum(1)).ravel() /
                   (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel() *
                            np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9))
    return np.maximum(*out)          # best of word / char views


def embed_scores(rows, model):
    texts = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    idx = {t: i for i, t in enumerate(texts)}
    E = model.encode(texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    return np.array([float(E[idx[r["a"]]] @ E[idx[r["b"]]]) for r in rows])


def cell_auc(scores, rows):
    g = np.array([r["gold"] for r in rows])
    st = np.array([r["stratum"] for r in rows])
    res = {"auc": auc(scores, g), "ci": boot_ci(scores, g)}
    for tag, keep in (("hard", "neg_hard"), ("easy", "neg_easy")):
        m = (st == "pos") | (st == keep)
        res[f"auc_{tag}"] = auc(scores[m], g[m]) if (g[m] == 0).any() else float("nan")
    return res


def agglom_ari():
    """Partition-level baseline: cluster the items of each corpus at the gold K, score by ARI."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELS["bge"])
    out = []
    for fn in sorted(glob.glob(f"{OUT}/nodes_*.json")):
        c = json.load(open(fn))
        for lvl, pk in (("L0", "p1"), ("R1", "p2"), ("R2", "p3")):
            items = [it for it in c["items"] if it["level"] == lvl and it.get(pk)]
            if len(items) < 20:
                continue
            if len(items) > 4000:
                items = items[:4000]
            gold = [it[pk] for it in items]
            K = len(set(gold))
            E = m.encode([it["text"] for it in items], batch_size=64,
                         normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            lab = AgglomerativeClustering(n_clusters=K, metric="cosine",
                                          linkage="average").fit_predict(E)
            out.append({"corpus": c["corpus"], "level": lvl, "n": len(items), "K_gold": K,
                        "ari": float(adjusted_rand_score(gold, lab))})
            print(f"  agglom {c['corpus']:<11}{lvl:<4}n={len(items):>5} K={K:>4} ARI={out[-1]['ari']:.3f}")
    return out


def main():
    from sentence_transformers import SentenceTransformer
    cells = load_cells()
    loaded = {}
    results = defaultdict(dict)
    for (corpus, rung), rows in sorted(cells.items()):
        results[(corpus, rung)]["tfidf"] = cell_auc(tfidf_scores(rows), rows)
        for name, path in MODELS.items():
            if name not in loaded:
                loaded[name] = SentenceTransformer(path)
            results[(corpus, rung)][name] = cell_auc(embed_scores(rows, loaded[name]), rows)

    print(f"\n{'corpus':<12}{'rung':<5}{'n':>5} | " +
          "".join(f"{m:>16}" for m in ["tfidf", "minilm", "mpnet", "bge"]))
    print(" " * 25 + "   " + "".join(f"{'AUC   hard':>16}" for _ in range(4)))
    flat = []
    for (corpus, rung), r in sorted(results.items()):
        line = f"{corpus:<12}{rung:<5}{len(cells[(corpus,rung)]):>5} | "
        for m in ("tfidf", "minilm", "mpnet", "bge"):
            line += f"{r[m]['auc']:>8.3f}{r[m]['auc_hard']:>8.3f}"
        print(line)
        flat.append({"corpus": corpus, "rung": rung,
                     **{f"{m}_{k}": r[m][k] for m in r for k in ("auc", "auc_hard", "auc_easy")}})
    print("\nAgglomerative (bge, cut at gold K):")
    ari = agglom_ari()
    json.dump({"pairwise": flat, "agglomerative": ari},
              open(f"{OUT}/baseline_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
