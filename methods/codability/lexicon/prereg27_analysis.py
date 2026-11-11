#!/usr/bin/env python
"""PREREG-27 analysis for domain argv[1]: the PREREG-26 statistics (prereg26_analysis.analyse) on p27_pairs.json.
PRIMARY = calibrated selection; SECONDARY (a) = restricted to selected pairs also strict-same by both families.
No strict-selection scoring (Amendment 1)."""
import json
import sys

sys.path.insert(0, "ROOT/methods/codability/lexicon")
import prereg26_analysis as A  # noqa: E402

DOM = sys.argv[1]
A.OUT = f"ROOT/outputs/lexicon/prereg27_{DOM}_20261012"
prim = json.load(open(f"{A.OUT}/p27_pairs.json"))
ss = lambda s: len(s.get("strict_verdicts", {})) == 2 and all(v == "same" for v in s["strict_verdicts"].values())
res = {"design": f"notes/2026-10-12__prereg27-{DOM}-name-swap.md (+ Amendment 1)", "n_selected": len(prim)}
for fam in A.FAMS:
    R = A.load_runs(fam)
    res[fam] = {"n_valid_runs": len(R),
                "PRIMARY_calibrated": A.analyse(fam, prim, R, lambda p: ""),
                "SECONDARY_a_restricted_to_strict_same": A.analyse(fam, [s for s in prim if ss(s)], R, lambda p: "")}
json.dump(res, open(f"{A.OUT}/prereg27_results.json", "w"), indent=1, ensure_ascii=False)
with open(f"{A.OUT}/prereg27_per_construct.tsv", "w") as f:
    cols = ("mid", "gap", "n_valid_runs", "within_A", "within_B", "between_name", "between_scaf", "delta_name", "delta_scaf", "name_A", "name_B")
    f.write("family\t" + "\t".join(cols) + "\n")
    for fam in A.FAMS:
        for r in res[fam]["PRIMARY_calibrated"]["per_construct"]:
            f.write(fam + "\t" + "\t".join("" if r[c] is None else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")
for fam in A.FAMS:
    for an in ("PRIMARY_calibrated", "SECONDARY_a_restricted_to_strict_same"):
        print(fam, an, {k: v for k, v in res[fam][an].items() if k != "per_construct"})
