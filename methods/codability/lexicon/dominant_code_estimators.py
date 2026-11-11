#!/usr/bin/env python3
"""W6: dominant-code estimator bake-off. Raw modal share is size-biased (the confound that
killed the first codability headline), so we compare head-share estimators on a PREQUENTIAL
test: forecast P(next author's name == current modal name) per concept, scored on the
stable-hash 20% doc holdout in hash order, state updated after each scored event. The best
estimator becomes the paper-wide "dominant code strength" statistic.

Estimators (all argmax-identical; they differ in the PROBABILITY they assign the head):
  mle     c_max/N                        (plugin; overconfident at shallow N)
  gt      (c_max/N)*(1 - f1/N)           (Turing-discounted plugin)
  py      (c_max - d)/(theta + N)        (Pitman-Yor posterior predictive, shared task fit)
  dp      c_max/(theta_dp + N)           (Dirichlet-process predictive)
  eb      (c_max + a)/(N + a + b)        (empirical-Bayes Beta on train head shares, N>=3)

Also a model-based sanity arm: MSE against the long-run continuation head share simulated
from the fitted PY seeded with train counts (favors py by construction — reported as sanity,
the prequential arm is the fair test).

Output: outputs/lexicon/dominant_code_estimators_20260720.json
"""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from methods.codability.lexicon.codability_sampling_model import (
    LEX, TASKS, fit_dp, fit_py, hists, load_records, partitions)

SEED = 0
CLIP = 1e-6
SIM_M, SIM_T = 60, 300


def head(counter):
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counter else None


def eb_fit(state):
    shares = [max(c.values()) / sum(c.values()) for c in state.values() if sum(c.values()) >= 3]
    if len(shares) < 5:
        return 1.0, 1.0
    m, v = float(np.mean(shares)), float(np.var(shares))
    if v <= 0 or v >= m * (1 - m):
        return 1.0, 1.0
    s = m * (1 - m) / v - 1
    return max(m * s, .05), max((1 - m) * s, .05)


def forecasts(c, d, th, th_dp, a, b):
    N = sum(c.values())
    cmax = max(c.values())
    f1 = sum(1 for v in c.values() if v == 1)
    return {"mle": cmax / N,
            "gt": (cmax / N) * (1 - f1 / N),
            "py": (cmax - d) / (th + N),
            "dp": cmax / (th_dp + N),
            "eb": (cmax + a) / (N + a + b)}


def py_continue(counts, d, th, T, rng):
    counts = list(counts)
    head_i = int(np.argmax(counts))
    for _ in range(T):
        tot = sum(counts)
        if rng.random() < (th + len(counts) * d) / (th + tot):
            counts.append(1)
        else:
            w = np.array([x - d for x in counts], float)
            counts[int(rng.choice(len(counts), p=w / w.sum()))] += 1
    return counts[head_i] / sum(counts)


def run_task(task, rng):
    rows = load_records(task)
    named = [r for r in rows if r["name"]]
    d, th, _ = fit_py(hists(partitions(named)))
    th_dp, _ = fit_dp(hists(partitions(named)))
    tr = [r for r in named if int(hashlib.md5(r["doc"].encode()).hexdigest(), 16) % 5 != 0]
    te = sorted((r for r in named if int(hashlib.md5(r["doc"].encode()).hexdigest(), 16) % 5 == 0),
                key=lambda r: hashlib.md5(r["doc"].encode()).hexdigest())
    state = defaultdict(Counter)
    for r in tr:
        state[r["con"]][r["name"]] += 1
    a, b = eb_fit(state)
    names = ["mle", "gt", "py", "dp", "eb"]
    brier = {e: [] for e in names}
    ll = {e: [] for e in names}
    nbin = {e: defaultdict(list) for e in names}
    n_scored = 0
    for r in te:
        c = state[r["con"]]
        if not c:
            state[r["con"]][r["name"]] += 1
            continue
        N = sum(c.values())
        y = 1 if r["name"] == head(c) else 0
        fc = forecasts(c, d, th, th_dp, a, b)
        binlab = "1-2" if N <= 2 else ("3-5" if N <= 5 else "6+")
        for e, p in fc.items():
            p = float(np.clip(p, CLIP, 1 - CLIP))
            brier[e].append((p - y) ** 2)
            ll[e].append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
            nbin[e][binlab].append((p - y) ** 2)
        state[r["con"]][r["name"]] += 1
        n_scored += 1
    # model-based sanity arm: long-run head share under PY continuation
    sim_mse = {e: [] for e in names}
    cons = [c for c in state.values() if 3 <= sum(c.values()) <= 50]
    rng.shuffle(cons)
    for c in cons[:120]:
        truth = float(np.mean([py_continue(sorted(c.values(), reverse=True), d, th, SIM_T, rng)
                               for _ in range(SIM_M)]))
        fc = forecasts(c, d, th, th_dp, a, b)
        for e, p in fc.items():
            sim_mse[e].append((p - truth) ** 2)
    res = {"py_d": round(d, 3), "py_theta": round(th, 3), "dp_theta": round(th_dp, 3),
           "eb_a": round(a, 3), "eb_b": round(b, 3), "n_test_scored": n_scored,
           "prequential": {e: {"brier": round(float(np.mean(brier[e])), 4),
                               "logloss": round(float(np.mean(ll[e])), 4),
                               "brier_by_N": {k: round(float(np.mean(v)), 4)
                                              for k, v in sorted(nbin[e].items())}}
                           for e in names},
           "sim_headshare_mse": {e: round(float(np.mean(sim_mse[e])), 5) for e in names}}
    return res


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    print(f"{'task':22}{'est':>5}{'brier':>8}{'logloss':>9}{'N1-2':>8}{'N3-5':>8}{'N6+':>8}{'simMSE':>9}")
    for task in TASKS:
        res = run_task(task, rng)
        out[task] = res
        for e in ["mle", "gt", "py", "dp", "eb"]:
            p = res["prequential"][e]
            bb = p["brier_by_N"]
            print(f"{task:22}{e:>5}{p['brier']:>8.4f}{p['logloss']:>9.4f}"
                  f"{bb.get('1-2', float('nan')):>8.4f}{bb.get('3-5', float('nan')):>8.4f}"
                  f"{bb.get('6+', float('nan')):>8.4f}{res['sim_headshare_mse'][e]:>9.5f}")
    path = f"{LEX}/dominant_code_estimators_20260720.json"
    json.dump(out, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
