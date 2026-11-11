#!/usr/bin/env python
"""Post-Opus harvest (run after the Opus fleet completes):
  (1) CW two-family repair: Sonnet-screen(==2) AND Opus-confirm(==2) -> >=2-edge star apply -> score.
  (2) Truth-lock the 6 tasks: adjudicated_truth = Sonnet&GLM consensus + Opus verdict on disagreements.
Idempotent; safe to re-run as more Opus shards land.
    python -m methods.codability.lexicon.harvest_opus
"""
import json, os
from methods.codability.lexicon import repair
OUT = repair.OUT
SIX = ['press-releases', 'news-homepages', 'grant-funding',
       'legal-outcome-prediction', 'notice-and-comment', 'patents']


def cw_apply():
    cand = json.load(open(os.path.join(OUT, 'repair_candidates_creative-writing.json')))[:2500]
    base = repair.load_base_partition('creative-writing')
    # advanced = Sonnet screen==2 (screen_min=2); confirmed = Opus confirm==2
    edges = repair.ingest_verified('creative-writing', cand,
                                   'repair_votes/screen_creative-writing_*.jsonl',
                                   confirm_glob='repair_votes/confirm_creative-writing_*.jsonl',
                                   screen_min=2)
    before = repair.score_vs_truth('creative-writing', base)
    for me in (1, 2, 3):
        res = repair.apply_merges(base, edges, min_edges=me, task='creative-writing')
        after = repair.score_vs_truth('creative-writing', res['partition'])
        print(f'CW two-family min_edges={me}: edges={len(edges)} merges={res["n_merges"]} '
              f'recall {before["recall"]}->{after["recall"]} precision {before["precision"]}->{after["precision"]}')
        if me == 2:
            json.dump(res['partition'], open(os.path.join(OUT, 'partition_creative-writing_L0v2.json'), 'w'))


def truthlock():
    for t in SIX:
        cp = os.path.join(OUT, f'truthlock_consensus_{t}.json')
        if not os.path.exists(cp):
            print(f'{t}: no consensus file, skip'); continue
        cons = {k: bool(v) for k, v in json.load(open(cp)).items()}
        opus = repair._load_scored(f'repair_votes/truthlock_{t}_*.jsonl')
        truth = dict(cons)
        added = 0
        for pid, s in opus.items():
            if pid not in cons:                    # disagreement pids (+ any non-consensus anchors)
                truth[pid] = (s == 2); added += 1
        json.dump(truth, open(os.path.join(OUT, f'adjudicated_truth_{t}.json'), 'w'))
        print(f'{t}: truth locked n={len(truth)} (consensus {len(cons)} + Opus {added}); same={sum(truth.values())}')


if __name__ == '__main__':
    cw_apply()
    print('---')
    truthlock()
