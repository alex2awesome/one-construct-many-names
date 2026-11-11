#!/usr/bin/env python3
"""W2: inventory richness per grain (L0 phrasing / R1 construct / R2 theme / R3 category)
for all 11 fields. Species = cluster id at the grain (chained through canonical partitions);
abundance = criterion instances. Reports observed K, bias-corrected Chao1, Good-Turing
missing mass f1/N, Heaps exponent, and doc-grain accumulation curves (stable-hash doc order,
never seeded shuffle). Answers "how many unique subfields per task" at the CLUSTERED grain —
the raw subtask_short annotation layer is 91-96% singletons (unbounded string space, audited
2026-07-20) and is NOT used here.

Output: outputs/lexicon/richness_20260720.json
"""
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict

LEX = "ROOT/outputs/lexicon"
TASKS = ["code-review", "creative-writing", "grant-funding", "humor",
         "legal-outcome-prediction", "math-stackexchange", "news-homepages",
         "notice-and-comment", "patents", "peer-review", "press-releases"]
GRAINS = ["L0", "R1", "R2", "R3"]
CURVE_POINTS = 50


def load_chain(task):
    l0 = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_L0v4.json")).items()}
    r1 = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_R1.json")).items()}
    r2 = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_R2.json")).items()}
    r3 = {str(k): str(v) for k, v in json.load(open(f"{LEX}/partition_{task}_R3.json")).items()}

    def up(v, table):
        if v in table:
            return table[v]
        base = re.sub(r"_c\d+$", "", v)
        return table.get(base)

    chain = {}
    miss = Counter()
    for key, c0 in l0.items():
        g1 = up(c0, r1)
        if g1 is None:
            miss["R1"] += 1
            continue
        g2 = up(g1, r2)
        if g2 is None:
            miss["R2"] += 1
            continue
        g3 = up(g2, r3)
        if g3 is None:
            miss["R3"] += 1
            continue
        chain[key] = {"L0": c0, "R1": g1, "R2": g2, "R3": g3}
    return chain, miss


def chao1(counts):
    K = len(counts)
    f1 = sum(1 for v in counts.values() if v == 1)
    f2 = sum(1 for v in counts.values() if v == 2)
    return K + f1 * (f1 - 1) / (2 * (f2 + 1))


def heaps_exponent(ns, ks):
    xs = [math.log(n) for n, k in zip(ns, ks) if n > 0 and k > 0]
    ys = [math.log(k) for n, k in zip(ns, ks) if n > 0 and k > 0]
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else None


def main():
    out = {}
    print(f"{'task':26}{'grain':>6}{'N':>7}{'K_obs':>7}{'chao1':>8}{'f1/N':>7}{'heaps_b':>8}")
    for task in TASKS:
        chain, miss = load_chain(task)
        if miss:
            print(f"{task}: unchained keys {dict(miss)} (excluded)")
        bydoc = defaultdict(list)
        for key, levels in chain.items():
            doc = key.split("::")[2] if key.count("::") >= 3 else key
            bydoc[doc].append(levels)
        docs = sorted(bydoc, key=lambda d: hashlib.md5(d.encode()).hexdigest())
        out[task] = {"n_instances": len(chain), "n_docs": len(docs),
                     "unchained": dict(miss), "grains": {}}
        for grain in GRAINS:
            counts = Counter(levels[grain] for levels in chain.values())
            N = sum(counts.values())
            f1 = sum(1 for v in counts.values() if v == 1)
            seen = set()
            ns, ks = [], []
            n_run = 0
            for doc in docs:
                for levels in bydoc[doc]:
                    n_run += 1
                    seen.add(levels[grain])
                ns.append(n_run)
                ks.append(len(seen))
            step = max(1, len(ns) // CURVE_POINTS)
            curve = [(ns[i], ks[i]) for i in range(0, len(ns), step)] + [(ns[-1], ks[-1])]
            b = heaps_exponent(ns, ks)
            g = {"K_obs": len(counts), "N": N, "f1": f1,
                 "f2": sum(1 for v in counts.values() if v == 2),
                 "chao1": round(chao1(counts), 1), "gt_missing_mass": round(f1 / N, 4),
                 "heaps_exponent": round(b, 3) if b is not None else None,
                 "accumulation_curve": curve}
            out[task]["grains"][grain] = g
            print(f"{task:26}{grain:>6}{N:>7}{g['K_obs']:>7}{g['chao1']:>8.0f}"
                  f"{g['gt_missing_mass']:>7.3f}{(g['heaps_exponent'] or 0):>8.3f}")
    path = f"{LEX}/richness_20260720.json"
    json.dump(out, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
