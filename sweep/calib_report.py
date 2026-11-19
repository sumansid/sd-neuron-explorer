"""Dose-response summary of a calibration run: how much images change per dose, for neurons vs random directions.

    python sweep/calib_report.py calib-L68
"""
import collections
import json
import sys

import numpy as np

run = sys.argv[1]
rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
rows = [r for r in rows if r["kind"] != "baseline"]
by = collections.defaultdict(list)
for r in rows:
    by[(r["kind"], r["dose"])].append(r)
print("kind   dose   mse     cos_base cos_prompt  |dsat| |dlum|   dsat range     top-attr-shift")
for k in sorted(by):
    rs = by[k]
    m = lambda f: float(np.mean([f(r) for r in rs]))
    shifts = np.mean([r["d_attr"] for r in rs], 0)
    print(f"{k[0][:3]:4s} {k[1]:+6.2f}  {m(lambda r: r['mse']):.4f}  {m(lambda r: r['cos_base']):.3f}    {m(lambda r: r['cos_prompt']):.3f}     "
          f"{m(lambda r: abs(r['d_sat'])):.3f}  {m(lambda r: abs(r['d_lum'])):.3f}   {min(r['d_sat'] for r in rs):+.2f}..{max(r['d_sat'] for r in rs):+.2f}   "
          f"max|shift| {np.abs(shifts).max():.2f}")
# the most saturation-changing neurons at each dose
print("\nstrongest |d_sat| per unit (mean over prompts) at |dose|<=1:")
per = collections.defaultdict(list)
for r in rows:
    if abs(r["dose"]) <= 1:
        per[(r["kind"], r["neuron"], r["dose"])].append(r["d_sat"])
top = sorted(((np.mean(v), k) for k, v in per.items()), key=lambda x: -abs(x[0]))[:12]
for v, k in top:
    print(f"  {k[0][:3]} N{k[1]:5d} dose {k[2]:+.2f}  mean d_sat {v:+.3f}")
