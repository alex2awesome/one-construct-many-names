#!/usr/bin/env python3
"""PREREG-20: sampling-theory validation on a known-denominator frame (AIRules).

The frame (99,967 subreddits; 467,973 rule tokens; subscriber counts) gives exact
truth for every estimand, so we can validate the estimators we use on the open web:

  Part A  Frame calibration: uniform vs popularity-biased samplers; the deception
          gap M_unif - M_samp; GT correctness AT its own estimand; IPW/HT recovery
          with estimated propensities, incl. the zero-support (truncated) regime.
  Part B  Multi-list capture-recapture: Chao1 + 5-list Chao-type lower bound vs
          true richness K under heterogeneous capture.
  Part C  Partial identification: reachability-floor bound fans vs frame truth.

Type system here = L0 surface grain (normalized rule short-name). The R1 construct
grain replication runs separately on the humor sub-frame (Codex-seated).
Outputs: outputs/lexicon/frame_calibration_20260724/results.json
"""
import gzip, json, re, sys, collections
import numpy as np

ROOT = 'ROOT'
OUT = f'{ROOT}/outputs/lexicon/frame_calibration_20260724'
RNG = np.random.default_rng(20260724)

NORM_RE = re.compile(r'[^a-z0-9 ]+')
def norm(s):
    return NORM_RE.sub(' ', s.lower()).strip()

# ---------------------------------------------------------------- frame loading
def load_frame():
    """Return (type_of_token, subs_of_token, sub_of_token, K, type_names)."""
    types, subs_w, sub_id = [], [], []
    tmap = {}
    with gzip.open(f'{ROOT}/datasets/prior_norms/airules_frame.jsonl.gz', 'rt') as f:
        for si, line in enumerate(f):
            r = json.loads(line)
            w = max(r['subs'], 1)
            for ru in r['rules']:
                t = norm(ru['sn'])
                if not t:
                    continue
                if t not in tmap:
                    tmap[t] = len(tmap)
                types.append(tmap[t]); subs_w.append(w); sub_id.append(si)
    names = [None] * len(tmap)
    for t, i in tmap.items():
        names[i] = t
    return (np.array(types), np.array(subs_w, dtype=float),
            np.array(sub_id), len(tmap), names)

# ---------------------------------------------------------------- truth helpers
def type_propensities(types, w, K):
    """pi_i = P(one draw hits type i) under token weights w (sum to 1)."""
    p = w / w.sum()
    pi = np.zeros(K)
    np.add.at(pi, types, p)
    return pi

def true_missing_mass(pi, seen_mask):
    """P(next draw from THIS sampler is an unseen type)."""
    return pi[~seen_mask].sum()

# ---------------------------------------------------------------- samplers
def draw(types, w, n):
    idx = RNG.choice(len(types), size=n, replace=True, p=w / w.sum())
    return types[idx]

# ---------------------------------------------------------------- estimators
def gt_missing_mass(sample):
    c = collections.Counter(sample)
    f1 = sum(1 for v in c.values() if v == 1)
    return f1 / len(sample), len(c)

def chao1(sample):
    c = collections.Counter(sample)
    f1 = sum(1 for v in c.values() if v == 1)
    f2 = sum(1 for v in c.values() if v == 2)
    k = len(c)
    return k + (f1 * f1 / (2 * f2) if f2 > 0 else f1 * (f1 - 1) / 2)

def multilist_chao(lists):
    """Chao (1987)-style lower bound from k incidence lists (species = types,
    'samples' = lists, incidence = type appears in list)."""
    inc = collections.Counter()
    for L in lists:
        for t in set(L):
            inc[t] += 1
    q1 = sum(1 for v in inc.values() if v == 1)
    q2 = sum(1 for v in inc.values() if v == 2)
    k_obs = len(inc)
    m = len(lists)
    corr = (m - 1) / m
    return k_obs + (corr * q1 * q1 / (2 * q2) if q2 > 0 else corr * q1 * (q1 - 1) / 2), k_obs

def ht_richness(sample, pi_hat_of_type, n):
    """Horvitz-Thompson species estimate: sum over SEEN types of
    1 / P(type detected in n draws), with estimated propensities."""
    seen = set(sample)
    tot = 0.0
    for t in seen:
        det = 1.0 - (1.0 - min(pi_hat_of_type[t], 1.0)) ** n
        tot += 1.0 / max(det, 1e-12)
    return tot

# ---------------------------------------------------------------- main
def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    types, subs_w, sub_id, K, names = load_frame()
    Ntok = len(types)
    uni_w = np.ones(Ntok)
    pop_w = subs_w.copy()
    # zero-support regime: bottom-quartile-subscriber tokens unreachable
    q25 = np.quantile(subs_w, 0.25)
    trunc_w = np.where(subs_w <= q25, 0.0, subs_w)

    pi_uni = type_propensities(types, uni_w, K)
    pi_pop = type_propensities(types, pop_w, K)
    pi_trunc = type_propensities(types, trunc_w, K)
    # truth: how much uniform-draw mass lives on types the truncated sampler can't reach
    unreachable_mass = pi_uni[pi_trunc == 0].sum()

    res = {'frame': {'subreddits': int(sub_id.max() + 1), 'tokens': int(Ntok),
                     'K_types': int(K), 'unreachable_uniform_mass_trunc': float(unreachable_mass),
                     'trunc_zero_support_types': int((pi_trunc == 0).sum())}}

    REPS = 200
    NS = [1000, 5000, 20000]

    # ---------- Part A: deception gap + GT correctness + IPW ----------
    # decile propensity model (observable: subscriber decile of the subreddit)
    dec = np.digitize(subs_w, np.quantile(subs_w, np.linspace(.1, .9, 9)))
    partA = {str(n): collections.defaultdict(list) for n in NS}
    for n in NS:
        A = partA[str(n)]
        for rep in range(REPS):
            for samp_name, w, pi in (('uniform', uni_w, pi_uni),
                                     ('popularity', pop_w, pi_pop),
                                     ('popularity_trunc', trunc_w, pi_trunc)):
                idx = RNG.choice(Ntok, size=n, replace=True, p=w / w.sum())
                s = types[idx]
                seen = np.zeros(K, bool); seen[np.unique(s)] = True
                m_samp = true_missing_mass(pi, seen)          # sampler-relative truth
                m_unif = true_missing_mass(pi_uni, seen)      # population (uniform) truth
                gt, k_obs = gt_missing_mass(s)
                A[f'{samp_name}_M_samp'].append(m_samp)
                A[f'{samp_name}_M_unif'].append(m_unif)
                A[f'{samp_name}_GT'].append(gt)
                A[f'{samp_name}_Kobs'].append(k_obs)
                # GT concentration band (McAllester-Schapire style, crude O(sqrt(log/ n)))
                band = 2.0 * np.sqrt(np.log(20) / n) + 1.0 / n
                A[f'{samp_name}_GT_in_band'].append(abs(gt - m_samp) <= band)
                if samp_name in ('popularity', 'popularity_trunc'):
                    # IPW/HT with decile-ESTIMATED propensities: empirical decile draw
                    # shares / frame decile token counts -> per-token rate -> per-type pi
                    s2 = s; d2 = dec[idx]
                    per_dec_rate = {}
                    for d in range(10):
                        cnt_frame = (dec == d).sum()
                        per_dec_rate[d] = ((d2 == d).sum() / n) / max(cnt_frame, 1)
                    pi_hat = np.zeros(K)
                    rate_tok = np.array([per_dec_rate[d] for d in dec])
                    np.add.at(pi_hat, types, rate_tok)
                    K_ht = ht_richness(s2, pi_hat, n)
                    A[f'{samp_name}_K_HT'].append(K_ht)
        # summarize
        out = {}
        for k, v in A.items():
            v = np.array(v, dtype=float)
            out[k] = {'mean': float(v.mean()), 'lo': float(np.quantile(v, .025)),
                      'hi': float(np.quantile(v, .975))}
        partA[str(n)] = out
    res['partA'] = partA

    # ---------- Part B: capture-recapture validity ----------
    partB = {}
    for samp_name, w in (('uniform', uni_w), ('popularity', pop_w)):
        rows = {'chao1': [], 'multilist': [], 'kobs_single': [], 'kobs_5list': []}
        for rep in range(REPS):
            s = draw(types, w, 5000)
            rows['chao1'].append(chao1(s)); rows['kobs_single'].append(len(set(s)))
            lists = [draw(types, w, 1000) for _ in range(5)]
            ml, kobs = multilist_chao(lists)
            rows['multilist'].append(ml); rows['kobs_5list'].append(kobs)
        partB[samp_name] = {
            'chao1_mean': float(np.mean(rows['chao1'])),
            'chao1_valid_rate': float(np.mean(np.array(rows['chao1']) <= K)),
            'multilist_mean': float(np.mean(rows['multilist'])),
            'multilist_valid_rate': float(np.mean(np.array(rows['multilist']) <= K)),
            'kobs_single_mean': float(np.mean(rows['kobs_single'])),
            'kobs_5list_mean': float(np.mean(rows['kobs_5list'])),
            'true_K': int(K)}
    res['partB'] = partB

    # ---------- Part C: partial identification (reachability floor) ----------
    # bound: with all normalized propensities >= pi_min, unseen TYPE COUNT after n draws
    # <= K_hat * (1 - pi_min)^n where K_hat is any valid upper... we use the honest
    # direction: given assumed floor, an upper bound on total K from the observed K_obs:
    # K <= K_obs / (1 - (1-pi_min)^n)  [each type detected w.p. >= 1-(1-pi_min)^n]
    # -> unseen fraction bound = 1 - K_obs/K_bound... equivalently:
    partC = {}
    true_min_pi = {'uniform': float(pi_uni[pi_uni > 0].min()),
                   'popularity': float(pi_pop[pi_pop > 0].min())}
    n = 20000
    grid = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    for samp_name, w in (('uniform', uni_w), ('popularity', pop_w)):
        s = draw(types, w, n)
        k_obs = len(set(s))
        fans = []
        for pmin in grid:
            det = 1.0 - (1.0 - pmin) ** n
            K_bound = k_obs / det
            fans.append({'pi_min': pmin, 'K_upper_bound': float(K_bound),
                         'unseen_frac_bound': float(max(0.0, 1 - k_obs / K_bound)),
                         'contains_truth': bool(K_bound >= K)})
        partC[samp_name] = {'n': n, 'K_obs': int(k_obs), 'true_K': int(K),
                            'true_unseen_frac': float(1 - k_obs / K),
                            'true_min_pi': true_min_pi[samp_name], 'fan': fans}
    res['partC'] = partC

    with open(f'{OUT}/results.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({'frame': res['frame']}, indent=1))
    print('written', f'{OUT}/results.json')

if __name__ == '__main__':
    main()
