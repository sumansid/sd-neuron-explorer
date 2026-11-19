"""Dose ladder summary: python sweep/calib_summary.py <run>  -> mean cos_base / |d_sat| per dose, neurons vs random."""
import json, sys, collections
import numpy as np
rows = [json.loads(l) for l in open(f"runs/{sys.argv[1]}.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    if r["kind"] != "baseline": by[(r["kind"], r["dose"])].append(r)
print(f"{'kind':8s} {'dose':>6s} {'cos_base':>8s} {'mse':>6s} {'|dsat|':>6s} {'|dlum|':>6s}  (n)")
for (k, d), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
    print(f"{k:8s} {d:6g} {np.mean([r['cos_base'] for r in rs]):8.3f} {np.mean([r['mse'] for r in rs]):6.3f} "
          f"{np.mean([abs(r['d_sat']) for r in rs]):6.3f} {np.mean([abs(r['d_lum']) for r in rs]):6.3f}  ({len(rs)})")
