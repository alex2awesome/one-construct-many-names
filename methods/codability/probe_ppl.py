#!/usr/bin/env python
"""T8c probe-perplexity covariate: mean token logprob of each probe text under each family's
model (instruct checkpoints as base proxies — noted deviation from the design, base weights
not cached on sk3). Output feeds the CPU partial: DiD partialled on the per-probe
family-familiarity difference. One model per process; direct vLLM (backend lacks
prompt_logprobs).
"""
import argparse
import json
import os

import numpy as np


def resolve(model):
    hub = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
                       "hub")
    d = os.path.join(hub, "models--" + model.replace("/", "--"))
    try:
        commit = open(os.path.join(d, "refs", "main")).read().strip()
        snap = os.path.join(d, "snapshots", commit)
        if os.path.isdir(snap):
            return snap
    except OSError:
        pass
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-probes", type=int, default=300)
    p.add_argument("--gepa-reserve", type=int, default=60)
    a = p.parse_args()

    from methods.metric_implementer import config as cfgmod
    from methods.metric_implementer.experiments.run_real_test import _load_texts
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), a.task)
    texts, _ = _load_texts(a.task, a.gepa_reserve + a.n_probes, cfg)
    probes = [t[: cfg.max_text_chars] for t in texts[a.gepa_reserve: a.gepa_reserve + a.n_probes]]

    from vllm import LLM, SamplingParams
    kwargs = dict(model=resolve(a.model), max_model_len=8192, trust_remote_code=True,
                  gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL") or 0.85),
                  enable_prefix_caching=False, tensor_parallel_size=1)
    bs = os.environ.get("VLLM_BLOCK_SIZE")
    if bs:
        kwargs["block_size"] = int(bs)
    eng = LLM(**kwargs)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    outs = eng.generate(probes, sp)
    means = []
    for o in outs:
        lps = [next(iter(d.values())).logprob for d in (o.prompt_logprobs or []) if d]
        means.append(float(np.mean(lps)) if lps else None)
    rep = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rep.setdefault(a.task, {})[a.model.split("/")[-1]] = means
    json.dump(rep, open(a.out, "w"))
    ok = [m for m in means if m is not None]
    print(f"[{a.model}] {a.task}: {len(ok)}/{len(means)} probes, mean logprob {np.mean(ok):.3f}")


if __name__ == "__main__":
    main()
