"""Stratum assignment + frozen stratified splits + the probe-imbalance guard. EMBEDDING-FREE by
design (proposal §4.1): strata come from corpus METADATA (genre/venue/subtask tags) or a categorical
one-word judge classification — never topic models or embedding clusters, whose geometry shifts with
model training.

The guard implements failure mode (b) of the proposal §3: a stratum where the target signature is
near-constant (e.g. a horror-specific criterion scored on adventure probes) has UNDEFINED codability
— masking it is the fix; scoring it "low" is the bug."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def normalize_strata(tags: Sequence) -> np.ndarray:
    """Metadata tags → canonical stratum labels (lowercase, stripped; empty/NaN → 'unknown')."""
    out = []
    for t in tags:
        s = "" if t is None else str(t).strip().lower()
        out.append(s if s and s != "nan" else "unknown")
    return np.asarray(out, dtype=object)


def make_stratum_judge(vocab: Sequence[str], model: str = "glm-4.7", *,
                       backend: str = "zai_anthropic", max_tokens: int = 8) -> Callable:
    """Categorical stratum judge: one-word classification into ``vocab`` (embedding-free — a
    discrete judge decision, not a geometry). **GLM quota is binding — be sparing**: ONE batched
    call per corpus, cache the labels beside the probe pool. Lazy imports so offline tests never
    touch backends. Returns ``judge(texts) -> np.ndarray[str]`` (labels outside vocab → 'unknown')."""
    from methods.metric_implementer import backends as _backends, config as _cfgmod
    cfg = _cfgmod.ImplementerConfig(backend=backend)
    be = _backends.LLMBackend(model, "stratum_judge", cfg)
    vocab_l = [str(v).strip().lower() for v in vocab]
    sysp = ("You classify a text into exactly one category. Reply with ONE word from this list and "
            "nothing else: " + ", ".join(vocab_l) + ".")

    def judge(texts: Sequence[str]) -> np.ndarray:
        qs = [f"Text:\n{str(t)[:2000]}\n\nCategory (one word):" for t in texts]
        outs = be.generate_batch(qs, system=sysp, max_tokens=max_tokens, temperature=0.0, seed=0)
        labs = []
        for o in outs:
            w = (o or "").strip().lower().split()
            labs.append(w[0] if w and w[0] in vocab_l else "unknown")
        return np.asarray(labs, dtype=object)

    return judge


def stratified_split(strata: Sequence, *, held_frac: float = 0.5, min_train: int = 30,
                     min_held: int = 30, seed: int = 0) -> dict:
    """FROZEN per-stratum train/held split (proposal §4.1: rubrics are induced on stratum-g TRAIN
    pairs only and executed on held-out g′ — the same held indices for every metric of the task, so
    the split must be a function of (strata, seed) alone, never of any metric's verdicts).

    Returns {"strata": [g…], "train_idx": {g: np.ndarray}, "held_idx": {g: np.ndarray},
    "viable": {g: bool}, "held_mask": np.ndarray[bool]} — ``viable`` False when a stratum misses the
    quota (report it as UNDEFINED, don't silently pool)."""
    strata = normalize_strata(strata)
    rng = np.random.default_rng(seed)
    groups = sorted(set(strata.tolist()))
    train_idx: Dict[str, np.ndarray] = {}
    held_idx: Dict[str, np.ndarray] = {}
    viable: Dict[str, bool] = {}
    held_mask = np.zeros(len(strata), bool)
    for g in groups:
        idx = np.where(strata == g)[0]
        idx = idx[rng.permutation(len(idx))]
        n_held = int(round(held_frac * len(idx)))
        h, tr = idx[:n_held], idx[n_held:]
        train_idx[g], held_idx[g] = np.sort(tr), np.sort(h)
        viable[g] = bool(len(tr) >= min_train and len(h) >= min_held)
        held_mask[h] = True
    return {"strata": groups, "train_idx": train_idx, "held_idx": held_idx,
            "viable": viable, "held_mask": held_mask}


def probe_balance_guard(target: np.ndarray, strata: Sequence, *, min_n: int = 40,
                        min_rate: float = 0.05, thresh: float = 0.5) -> dict:
    """Failure mode (b): per stratum, codability of a metric is DEFINED only where the metric's own
    verdicts vary. Returns {"defined": {g: bool}, "n": {g}, "pos_rate": {g}} — a stratum with
    n < ``min_n`` or binarized positive-rate outside [min_rate, 1−min_rate] (near-constant
    signature) is UNDEFINED there, not low."""
    strata = normalize_strata(strata)
    y = (np.asarray(target, float) > thresh).astype(int)
    defined: Dict[str, bool] = {}
    n: Dict[str, int] = {}
    rate: Dict[str, float] = {}
    for g in sorted(set(strata.tolist())):
        m = strata == g
        yy = y[m & np.isfinite(np.asarray(target, float))]
        n[g] = int(m.sum())
        rate[g] = float(yy.mean()) if len(yy) else float("nan")
        defined[g] = bool(len(yy) >= min_n and min_rate <= rate[g] <= 1.0 - min_rate)
    return {"defined": defined, "n": n, "pos_rate": rate}
