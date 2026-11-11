"""Dialect battery: five literature-standard dialectalization instruments + exact Jaccard,
run over the author-lexicon census on one shared, mirror-guarded footing (user-approved
2026-07-09; expands dialect.py's single exact-Jaccard instrument).

Instruments
  1. classifier   Translationese-detection design (Volansky/Ordan/Wintner; Koppel & Ordan):
                  predict a record's sub-community bucket from its author key_terms; out-of-fold
                  macro one-vs-rest AUC. Null = bucket labels permuted WITHIN construct, full
                  refit (preserves construct->bucket base rates, i.e. topic structure — any AUC
                  above this null is FORM, not topic). Expect null mean > .5; report obs vs null.
  2. fightin      Monroe, Colaresi & Quinn 2008 log-odds with informative Dirichlet prior:
                  per-bucket distinctive canonical terms (z >= 1.96). Descriptive; corpus-level,
                  so topic and dialect are mixed — read as "what marks this community's talk".
  3. lm           Community language models (Danescu-Niculescu-Mizil et al. 2013): word
                  uni+bigram add-k LM per bucket; self-advantage = mean CE under other buckets'
                  LMs minus CE under own (bits/token, positive = community-specific language).
                  Training EXCLUDES the eval record's construct (decrement trick) so concept
                  vocabulary cannot leak. Null = within-construct permutation, full recount.
  4. semantic     Two-axis soft matching (BERTScore lineage; NOT METEOR — hand-built paraphrase
                  tables rejected by user): bge-small cosine between records' term strings,
                  construct-matched within/cross contrast on the IDENTICAL pair set as the
                  lexical axes. Dialect signature = lexical delta > 0 with semantic delta ~ 0;
                  both > 0 = bucket-correlated sub-concept structure, not (only) dialect.
                  Extra arm: lexically-disjoint pairs only (Jaccard = 0) — shared meaning with
                  zero shared surface.
  5. chrf3        Dialectometry-style soft surface (Nerbonne/Wieling; chrF lineage): char
                  3-gram Jaccard between term strings — catches morphological variants
                  (newsworthy/newsworthiness) invisible to word-level Jaccard.
  0. jaccard      Exact canonical-term Jaccard recomputed on the same pair set (axis anchor).

Hygiene (locked lessons): pairwise instruments use the per-pair quote-overlap mirror guard
(>= .5, as verified in dialect.py); pooled/model instruments (1-3) additionally require GLOBAL
source-level mirror dedup (union-find over per-source quote-token signatures) — mirrored
documents otherwise leak across CV folds / into LM training. Buckets 'other'/'junk_doc' are
excluded from model instruments (grab-bag classes), kept in pairwise ones (matches dialect.py).

Rerun after any L0->R3 rebuild:
  python -m methods.codability.lexicon.dialect_battery <tasks> [--r1] [--partition P]
Writes outputs/lexicon/dialect_battery_<task>[_R1].json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from .dialect import OUT, bucket_of, canon, load_groups, qtok

MODEL_EXCLUDE = {"other", "junk_doc"}
MIN_CLASS = 20          # min records per bucket for model instruments
MIRROR = 0.5            # quote token-Jaccard mirror threshold (verified 2026-07-09)


# ---------------------------------------------------------------- shared data prep

def flatten(groups: dict, task: str) -> list[dict]:
    """construct/source/bucket/terms/quote records (best-per-source per construct)."""
    rows = []
    for cid, by_src in groups.items():
        for src, r in by_src.items():
            terms = sorted({canon(t) for t in (r.get("key_terms") or []) if canon(t)})
            if not terms:
                continue
            rows.append({"construct": cid, "source": src,
                         "bucket": bucket_of(task, (r.get("strata") or {}).get("subtask_short") or ""),
                         "terms": terms, "qtok": qtok(r.get("quote"))})
    return rows


def mirror_components(rows: list[dict], thresh: float = MIRROR) -> dict:
    """source -> mirror-component id, via union-find over per-source quote signatures."""
    sig, nrec = defaultdict(set), Counter()
    for r in rows:
        sig[r["source"]] |= r["qtok"]
        nrec[r["source"]] += 1
    srcs = sorted(sig, key=lambda s: len(sig[s]))
    parent = {s: s for s in srcs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(srcs):
        sa = sig[a]
        if len(sa) < 10:
            continue
        for b in srcs[i + 1:]:
            sb = sig[b]
            if len(sa) / len(sb) < thresh:   # size window: jac >= thresh impossible beyond it
                break
            if len(sa & sb) / len(sa | sb) >= thresh:
                parent[find(a)] = find(b)
    comp = {s: find(s) for s in srcs}
    reps = {}
    for s in sorted(srcs, key=lambda s: -nrec[s]):   # representative = most records
        reps.setdefault(comp[s], s)
    return {"comp": comp, "rep": reps}


def model_records(rows: list[dict], mc: dict) -> tuple[list[dict], list[str]]:
    """Mirror-deduped (representative source only), grab-bag buckets dropped,
    small classes dropped. Returns (records, kept_classes)."""
    dedup = [r for r in rows if mc["rep"][mc["comp"][r["source"]]] == r["source"]
             and r["bucket"] not in MODEL_EXCLUDE]
    counts = Counter(r["bucket"] for r in dedup)
    classes = sorted(b for b, c in counts.items() if c >= MIN_CLASS)
    return [r for r in dedup if r["bucket"] in classes], classes


def strict_records(rows: list[dict]) -> tuple[list[dict], list[str], int]:
    """Strictest mirror hygiene for model instruments: drop EVERY source implicated in any
    within-construct quote-mirror pair (>= MIRROR), then the usual class filtering. Partial
    mirrors (shared boilerplate, otherwise-different docs) survive global dedup; this doesn't
    let them survive. Verified 2026-07-09: all 4 domains' classifier/LM survive this arm."""
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["construct"]].append(r)
    bad = set()
    for rs in by_c.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                qa, qb = rs[i]["qtok"], rs[j]["qtok"]
                if qa and qb and len(qa & qb) / len(qa | qb) >= MIRROR:
                    bad.add(rs[i]["source"]); bad.add(rs[j]["source"])
    kept = [r for r in rows if r["source"] not in bad and r["bucket"] not in MODEL_EXCLUDE]
    counts = Counter(r["bucket"] for r in kept)
    classes = sorted(b for b, c in counts.items() if c >= MIN_CLASS)
    return [r for r in kept if r["bucket"] in classes], classes, len(bad)


def perm_within_construct(rows: list[dict], rng) -> list[str]:
    """Bucket labels shuffled within construct (topic-preserving null)."""
    by_c = defaultdict(list)
    for i, r in enumerate(rows):
        by_c[r["construct"]].append(i)
    lab = [r["bucket"] for r in rows]
    out = list(lab)
    for idx in by_c.values():
        sh = rng.permutation(len(idx))
        for k, i in enumerate(idx):
            out[i] = lab[idx[sh[k]]]
    return out


# ---------------------------------------------------------------- 1. classifier probe

def _feats(r: dict) -> list[str]:
    f = [f"P:{t}" for t in r["terms"]]
    for t in r["terms"]:
        f += t.split()
    return f


def _oof_auc(X, labels, classes, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    y = np.array([classes.index(b) for b in labels])
    proba = np.zeros((len(y), len(classes)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        for j, cls in enumerate(clf.classes_):
            proba[te, cls] = p[:, j]
    aucs = {}
    from sklearn.metrics import roc_auc_score
    for k in range(len(classes)):
        pos = y == k
        if 0 < pos.sum() < len(y):
            aucs[classes[k]] = roc_auc_score(pos, proba[:, k])
    return float(np.mean(list(aucs.values()))), aucs


def classifier_probe(recs: list[dict], classes: list[str], B: int = 200, seed: int = 0) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=_feats, min_df=2)
    X = vec.fit_transform(recs)
    obs, per_class = _oof_auc(X, [r["bucket"] for r in recs], classes, seed)
    rng = np.random.default_rng(seed)
    null = np.array([_oof_auc(X, perm_within_construct(recs, rng), classes, seed)[0]
                     for _ in range(B)])
    return {"auc": round(obs, 4), "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4),
            "p": round(float(np.mean(null >= obs) + 1 / (B + 1)), 4),
            "per_class_auc": {c: round(a, 4) for c, a in per_class.items()},
            "n": len(recs), "n_classes": len(classes), "n_features": X.shape[1], "B": B}


# ---------------------------------------------------------------- 2. fightin' words

def fightin_words(recs: list[dict], classes: list[str], alpha0: float = 100.0,
                  topk: int = 8) -> dict:
    cnt = {b: Counter() for b in classes}
    for r in recs:
        cnt[r["bucket"]].update(r["terms"])
    total = Counter()
    for c in cnt.values():
        total.update(c)
    N = sum(total.values())
    a = {w: alpha0 * c / N for w, c in total.items()}
    a0 = alpha0
    out, nsig = {}, {}
    for b in classes:
        ni = sum(cnt[b].values())
        nj = N - ni
        zs = []
        for w, tw in total.items():
            yi, yj = cnt[b][w], tw - cnt[b][w]
            d = (math.log((yi + a[w]) / (ni + a0 - yi - a[w]))
                 - math.log((yj + a[w]) / (nj + a0 - yj - a[w])))
            v = 1.0 / (yi + a[w]) + 1.0 / (yj + a[w])
            zs.append((w, d / math.sqrt(v)))
        zs.sort(key=lambda t: -t[1])
        sig = [(w, round(z, 2)) for w, z in zs if z >= 1.96]
        nsig[b] = len(sig)
        out[b] = sig[:topk]
    return {"top_terms": out, "n_sig_terms_total": sum(nsig.values()),
            "n_sig_per_bucket": nsig, "vocab": len(total), "alpha0": alpha0}


# ---------------------------------------------------------------- 3. community LMs

def _tok(r: dict) -> tuple[list[str], list[tuple]]:
    uni, bi = [], []
    for t in r["terms"]:
        ws = t.split()
        uni += ws
        bi += list(zip(ws, ws[1:]))
    return uni, bi


def _ce(uni, bi, U, cU, Bg, cB, utot, V, k=0.5, lam=0.7):
    """bits/token, interpolated add-k unigram + bigram-backoff-to-unigram.
    Construct-exclusion via subtract-on-lookup (cU/cB = the eval construct's counts) —
    no Counter copies, this runs inside the permutation loop."""
    if not uni:
        return None
    tot = 0.0
    for w in uni:
        pu = (U.get(w, 0) - cU.get(w, 0) + k) / (utot + k * V)
        tot += -math.log2(max(pu, 1e-12)) * lam
    for a, b in bi:
        pu = (U.get(b, 0) - cU.get(b, 0) + k) / (utot + k * V)
        pb = ((Bg.get((a, b), 0) - cB.get((a, b), 0) + k * pu * 10)
              / (U.get(a, 0) - cU.get(a, 0) + k * 10))
        tot += -math.log2(min(max(pb, 1e-12), 1.0)) * (1 - lam)
    return tot / max(len(uni), 1)


def community_lm(recs: list[dict], classes: list[str], B: int = 200, seed: int = 0) -> dict:
    toks = [_tok(r) for r in recs]
    vocab = {w for u, _ in toks for w in u}
    V = max(len(vocab), 1)
    by_construct_buckets = defaultdict(set)
    for r in recs:
        by_construct_buckets[r["construct"]].add(r["bucket"])
    eval_idx = [i for i, r in enumerate(recs) if len(by_construct_buckets[r["construct"]]) >= 2]

    def advantage(labels):
        U = {b: Counter() for b in classes}
        Bg = {b: Counter() for b in classes}
        cU = defaultdict(lambda: {b: Counter() for b in classes})
        cB = defaultdict(lambda: {b: Counter() for b in classes})
        for (uni, bi), lab, r in zip(toks, labels, recs):
            U[lab].update(uni); Bg[lab].update(bi)
            cU[r["construct"]][lab].update(uni); cB[r["construct"]][lab].update(bi)
        Utot = {b: sum(U[b].values()) for b in classes}
        cUtot = {(c, b): sum(d[b].values()) for c, d in cU.items() for b in classes}
        adv = []
        for i in eval_idx:
            r, lab = recs[i], labels[i]
            uni, bi = toks[i]
            c = r["construct"]
            ces = {}
            for b in classes:
                utot = Utot[b] - cUtot.get((c, b), 0)
                if utot < 50:
                    continue
                ces[b] = _ce(uni, bi, U[b], cU[c][b], Bg[b], cB[c][b], utot, V)
            if lab in ces and len(ces) >= 2:
                others = [v for b, v in ces.items() if b != lab]
                adv.append((c, lab, float(np.mean(others) - ces[lab])))
        return adv

    obs = advantage([r["bucket"] for r in recs])
    vals = [v for _, _, v in obs]
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean([v for _, _, v in advantage(perm_within_construct(recs, rng))]
                                   or [0])) for _ in range(B)])
    om = float(np.mean(vals)) if vals else None
    by_c, by_b = defaultdict(list), defaultdict(list)
    for c, b, v in obs:
        by_c[c].append(v); by_b[b].append(v)
    return {"advantage_bits": round(om, 4) if om is not None else None,
            "frac_positive": round(float(np.mean([v > 0 for v in vals])), 3) if vals else None,
            "n_eval": len(vals), "null_mean": round(float(null.mean()), 4),
            "p": round(float(np.mean(null >= om) + 1 / (B + 1)), 4) if om is not None else None,
            "per_construct": {str(c): round(float(np.mean(v)), 4) for c, v in by_c.items()},
            "per_bucket": {b: round(float(np.mean(v)), 4) for b, v in by_b.items()},
            "B": B}


# ---------------------------------------------------------------- 4/5/0. pairwise axes

def _chr3(s: str) -> set:
    s = " ".join(s.split())
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def embed_records(rows: list[dict]) -> dict:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    keys = sorted({"; ".join(r["terms"]) for r in rows})
    E = model.encode(keys, batch_size=256, normalize_embeddings=True,
                     show_progress_bar=False)
    return dict(zip(keys, E))


def paired_contrast(rows: list[dict], stat: str, emb: dict | None = None,
                    mirror: float = MIRROR, disjoint_only: bool = False,
                    B: int = 1000, seed: int = 0) -> dict | None:
    """Construct-matched within/cross-bucket contrast for an arbitrary pair statistic,
    with the verified quote-overlap mirror guard. stat in {jaccard, chrf3, cosine}."""
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["construct"]].append(r)
    rng = np.random.default_rng(seed)
    per, perm = [], np.zeros(B)
    npw = npc = 0
    for cid, rs in by_c.items():
        n = len(rs)
        if n < 2:
            continue
        tsets = [set(r["terms"]) for r in rs]
        joined = ["; ".join(r["terms"]) for r in rs]
        bks = np.array([r["bucket"] for r in rs], dtype=object)
        iu = np.triu_indices(n, k=1)
        val = np.zeros(len(iu[0]))
        keep = np.zeros(len(iu[0]), bool)
        for p, (i, j) in enumerate(zip(*iu)):
            a, b = tsets[i], tsets[j]
            if not (a or b):
                continue
            qa, qb = rs[i]["qtok"], rs[j]["qtok"]
            if qa and qb and len(qa & qb) / len(qa | qb) >= mirror:
                continue
            lex = len(a & b) / len(a | b)
            if disjoint_only and lex > 0:
                continue
            keep[p] = True
            if stat == "jaccard":
                val[p] = lex
            elif stat == "chrf3":
                ca, cb = _chr3(joined[i]), _chr3(joined[j])
                val[p] = len(ca & cb) / max(len(ca | cb), 1)
            elif stat == "cosine":
                val[p] = float(emb[joined[i]] @ emb[joined[j]])
        eq = (bks[:, None] == bks[None, :])[iu]
        w, c = keep & eq, keep & ~eq
        if w.any() and c.any():
            per.append((cid, float(val[w].mean()), float(val[c].mean())))
            npw += int(w.sum()); npc += int(c.sum())
            for bi in range(B):
                pb = bks[rng.permutation(n)]
                peq = (pb[:, None] == pb[None, :])[iu]
                pw, pc = keep & peq, keep & ~peq
                if pw.any() and pc.any():
                    perm[bi] += val[pw].mean() - val[pc].mean()
    if not per:
        return None
    wm = float(np.mean([x[1] for x in per]))
    cm = float(np.mean([x[2] for x in per]))
    obs = sum(w - c for _, w, c in per)
    return {"n_constructs": len(per), "pairs_w": npw, "pairs_c": npc,
            "within": round(wm, 4), "cross": round(cm, 4), "delta": round(wm - cm, 4),
            "p": round(float(np.mean(perm >= obs)), 4),
            "per_construct": {str(cid): round(w - c, 4) for cid, w, c in per}}


# ---------------------------------------------------------------- driver

def run_task(task: str, partition_path: str, B_model: int = 200, B_pairs: int = 1000,
             strict: bool = False) -> dict:
    groups = load_groups(task, partition_path)
    rows = flatten(groups, task)
    if strict:
        recs, classes, n_bad = strict_records(rows)
        out = {"task": task, "partition": os.path.basename(partition_path), "mode": "strict",
               "excluded_mirror_sources": n_bad, "n_model_records": len(recs),
               "model_classes": classes}
        if len(classes) >= 2:
            out["classifier"] = classifier_probe(recs, classes, B=B_model)
            out["community_lm"] = community_lm(recs, classes, B=B_model)
        return out
    mc = mirror_components(rows)
    recs, classes = model_records(rows, mc)
    out = {"task": task, "partition": os.path.basename(partition_path),
           "n_records": len(rows), "n_sources": len(mc["comp"]),
           "n_mirror_components_collapsed": len(mc["comp"]) - len(set(mc["rep"])),
           "n_model_records": len(recs), "model_classes": classes}
    if len(classes) >= 2:
        out["classifier"] = classifier_probe(recs, classes, B=B_model)
        out["fightin_words"] = fightin_words(recs, classes)
        out["community_lm"] = community_lm(recs, classes, B=B_model)
    emb = embed_records(rows)
    out["pairwise"] = {
        "jaccard": paired_contrast(rows, "jaccard", B=B_pairs),
        "chrf3": paired_contrast(rows, "chrf3", B=B_pairs),
        "semantic": paired_contrast(rows, "cosine", emb=emb, B=B_pairs),
        "semantic_disjoint": paired_contrast(rows, "cosine", emb=emb,
                                             disjoint_only=True, B=B_pairs),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--r1", action="store_true")
    ap.add_argument("--B-model", type=int, default=200)
    ap.add_argument("--B-pairs", type=int, default=1000)
    ap.add_argument("--strict", action="store_true",
                    help="strict-mirror arm: classifier+LM only, all mirror-pair sources dropped")
    args = ap.parse_args()
    for task in args.tasks.split(","):
        task = task.strip()
        pp = args.partition or (
            os.path.join(OUT, "codability", f"partition_key2R1_{task}.json") if args.r1
            else os.path.join(OUT, f"partition_{task}.json"))
        res = run_task(task, pp, args.B_model, args.B_pairs, strict=args.strict)
        suffix = ("_R1" if args.r1 else "") + ("_strictmirror" if args.strict else "")
        o = os.path.join(OUT, f"dialect_battery_{task}{suffix}.json")
        json.dump(res, open(o, "w"), indent=1)
        hyg = (f"{res['excluded_mirror_sources']} mirror sources excluded" if args.strict
               else f"{res['n_mirror_components_collapsed']} mirror sources collapsed")
        print(f"\n===== {task}{suffix}  ({res['n_model_records']} model recs, {hyg}) =====")
        if "classifier" in res:
            c, l = res["classifier"], res["community_lm"]
            print(f"  classifier AUC {c['auc']} vs null {c['null_mean']}±{c['null_sd']} p={c['p']}")
            print(f"  community-LM adv {l['advantage_bits']} bits/tok "
                  f"(frac+ {l['frac_positive']}, null {l['null_mean']}) p={l['p']}")
        if "fightin_words" in res:
            fw = res["fightin_words"]["top_terms"]
            for b in list(fw)[:4]:
                print(f"  FW {b}: {[w for w, z in fw[b][:5]]}")
        for k, v in res.get("pairwise", {}).items():
            if v:
                print(f"  {k:18s} within {v['within']:.4f} cross {v['cross']:.4f} "
                      f"DELTA {v['delta']:+.4f} p={v['p']}")
        print(f"  -> {o}")


if __name__ == "__main__":
    main()
