"""Tests for the Face-2 decompression grid driver: checkpoint identity parsing (with the
mixed-level guard born from the 2026-07-02 level-dispatch bug), span decomposition (in-span vs
out-of-span rung judges), and report math (balanced accuracy, exemplar exclusion, dossier
normalization, within-reader self-reference)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from methods.codability import run_decompression_grid as g


def test_level_group_dispatch_is_explicit(monkeypatch):
    monkeypatch.setattr(g, "r1_groups", lambda task: [("R1", task)])
    monkeypatch.setattr(g, "r2_groups", lambda task, bucket: [("R2", task, bucket)])
    monkeypatch.setattr(g, "r3_groups", lambda task, bucket: [("R3", task, bucket)])
    assert g._groups("t", "b", "R1")[0][0] == "R1"
    assert g._groups("t", "b", "R2")[0][0] == "R2"
    assert g._groups("t", "b", "R3")[0][0] == "R3"


def test_ckpts_parsing_and_filter(tmp_path):
    for n in ("creative-writing_R3_metric7_sigs.npz", "creative-writing_R2_metric3_sigs.npz",
              "junk.npz"):
        np.savez(tmp_path / n, x=1)
    out = g._ckpts(str(tmp_path), None)
    assert set(out) == {7, 3} and out[7][0] == "R3" and out[3][0] == "R2"
    assert set(g._ckpts(str(tmp_path), [7])) == {7}


def test_ckpts_mixed_level_collision_refused(tmp_path):
    np.savez(tmp_path / "t_R3_metric5_sigs.npz", x=1)
    np.savez(tmp_path / "t_R2_metric5_sigs.npz", x=1)
    with pytest.raises(ValueError, match="mixed-level"):
        g._ckpts(str(tmp_path), None)


def test_span_r2_separates_in_vs_out_of_span():
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 1, (30, 240))                          # 30 criteria x 240 probes
    mask = np.ones(240, bool)
    mask[:5] = False
    y_in = 0.7 * S[3] + 0.3 * S[11] + rng.normal(0, 0.01, 240)
    y_out = rng.uniform(0, 1, 240)
    r_in, r_out = g._span_r2(S, y_in, mask), g._span_r2(S, y_out, mask)
    if r_in is None:
        pytest.skip("sklearn unavailable")
    assert r_in > 0.5                                         # assembly of known units
    assert r_out < 0.2                                        # content outside the census span


def test_report_math_exemplar_exclusion_and_self_reference(tmp_path):
    ref = np.r_[np.full(30, 0.9), np.full(30, 0.1)]           # 30 pos, 30 neg probes
    rng = np.random.default_rng(0)
    sigs = np.clip(np.tile(ref, (8, 1)) + rng.normal(0, 0.05, (8, 60)), 0, 1)
    ck = tmp_path / "creative-writing_R3_metric5_sigs.npz"
    np.savez(ck, name="m5", M_i=ref, sigs=sigs)
    ex = {"pos": [0, 1], "neg": [58, 59]}
    msgs = {"5": {"name": "m5", "level": "R3", "rubric": "r",
                  "rungs": {r: "msg" for r in g.RUNG_ORDER}, "exemplar_idx": ex, "word_len": {}}}
    (tmp_path / "messages.json").write_text(json.dumps(msgs))
    perfect = ref.copy()
    perfect[[0, 1, 58, 59]] = 1 - perfect[[0, 1, 58, 59]]     # adversarial ONLY at exemplar probes
    rows, meta = [], []
    for rung in g.RUNG_ORDER:
        pred = perfect if rung in ("definition", "dossier") else np.full(60, 0.9)  # name = all-YES
        rows.append(pred)
        meta.append({"gi": 5, "rung": rung, "form": "canonical"})
    np.savez(tmp_path / "grid_r1.npz", scores=np.vstack(rows),
             meta=np.array([json.dumps(m) for m in meta], dtype=object),
             reader="r1", ref_dir=str(tmp_path))
    out = g.report(SimpleNamespace(out_dir=str(tmp_path)), {5: ("R3", str(ck))})
    pr = out["r1"]["5"]
    assert pr["definition"]["bal_acc"] == 1.0                 # exemplar-probe sabotage excluded
    assert pr["name"]["bal_acc"] == 0.5                       # all-YES = chance, balanced
    assert pr["definition"]["rel_to_dossier"] == 1.0
    assert pr["definition"]["self_agree"] == 1.0
    assert pr["name"]["self_agree"] < 1.0                     # name disagrees with own dossier
    assert "span_r2" in pr["definition"]
