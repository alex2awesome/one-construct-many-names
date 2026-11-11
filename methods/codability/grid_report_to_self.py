#!/usr/bin/env python
"""Adapter: decompression-grid report.json -> grid_bits_self-format JSON (2026-07-04).

The grid driver's report phase already computes the executor-consistent self-readout
(self_bits = bits each rung transmits about the reader's OWN full_rubric verdict; H_self =
that verdict's entropy). This adapter reshapes report.json into the [size][gi][rung] format
consumed by isomorphism_census.py, so new domains join the census without touching the
CW/humor grid_bits_self.json produced by the original notebook.

Usage:
  python -m methods.codability.grid_report_to_self \
      --report notebooks/data/two_faces_20260702/r3_pr/grid_pr_v1/report.json \
      --domain press-releases \
      --out notebooks/data/two_faces_20260702/grid_bits_self_pr.json
"""
import argparse
import json

SIZE_MAP = {
    "Llama-3.2-1B-Instruct": "1B",
    "Llama-3.2-3B-Instruct": "3B",
    "Llama-3.1-8B-Instruct": "8B",
    "Llama-3.1-70B-Instruct": "70B",
    "Llama-3.3-70B-Instruct": "70B",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rep = json.load(open(a.report))
    out = {}
    for reader, per_gi in rep.items():
        size = SIZE_MAP.get(reader)
        if size is None:  # e.g. hash-named snapshot dirs — keep the raw name
            size = reader
        cells = {}
        for gi, rungs in per_gi.items():
            cell = {}
            hs = None
            for rung, v in rungs.items():
                if not isinstance(v, dict):
                    continue
                if v.get("H_self") is not None:
                    hs = v["H_self"]
                if v.get("self_bits") is not None:
                    cell[rung] = v["self_bits"]
            if hs is not None:
                cell["H_self"] = hs
            if cell:
                cells[gi] = cell
        out[size] = cells
    json.dump({a.domain: out}, open(a.out, "w"), indent=1)
    sizes = {s: len(c) for s, c in out.items()}
    print(f"{a.domain}: sizes={sizes} -> {a.out}")


if __name__ == "__main__":
    main()
