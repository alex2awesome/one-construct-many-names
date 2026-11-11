#!/usr/bin/env python3
"""W13 adaptive-sampling dashboard: Good-Turing missing mass of the growing LAY corpus.
Per field: N criterion draws, distinct heads (norm_name), f1, GT missing mass f1/N, Chao1,
plus author-type mix. High missing mass => more search waves warranted (plan-note rule)."""
import json, glob, sys
from collections import Counter
from methods.codability.lexicon.codability_sampling_model import norm_name
SP=("/private/tmp/claude-502/-Users-anon-Projects-stanford-research-norm-research/"
    "6eb8228d-7011-4800-87e2-61a172f6003c/scratchpad")
out={}
print(f"{'field':26} {'docs':>4} {'N':>5} {'heads':>5} {'f1':>4} {'GT-mm':>6} {'chao1':>6} lay%")
for path in sorted(glob.glob(f"{SP}/lay_extract_*.jsonl")):
    t=path.split("lay_extract_")[1][:-6]
    heads=Counter(); n_docs=0; atypes=Counter()
    for l in open(path):
        try: r=json.loads(l)
        except Exception: continue
        if r.get("doc_summary_row"): n_docs+=1; atypes[r.get("author_type","?")]+=1; continue
        nm=norm_name(r.get("head_term"))
        if nm: heads[nm]+=1
    N=sum(heads.values()); D=len(heads)
    f1=sum(1 for v in heads.values() if v==1); f2=sum(1 for v in heads.values() if v==2)
    mm=f1/N if N else None
    chao1=D+f1*f1/(2*f2) if f2 else (D+f1*(f1-1)/2 if f1 else D)
    lay=atypes.get("lay_individual",0); tot=sum(atypes.values()) or 1
    out[t]={"docs":n_docs,"N":N,"heads":D,"f1":f1,"gt_missing_mass":round(mm,3) if mm is not None else None,
            "chao1":round(chao1,1),"author_types":dict(atypes)}
    print(f"{t:26} {n_docs:4} {N:5} {D:5} {f1:4} {mm if mm is None else round(mm,3)!s:>6} {chao1:6.0f} {lay/tot:.0%}")
json.dump(out,open("ROOT/outputs/lexicon/lay_corpus_gt_20260722.json","w"),indent=1)
print("wrote outputs/lexicon/lay_corpus_gt_20260722.json")
