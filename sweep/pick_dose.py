"""Largest |dose| in a calibration run whose mean neuron cos_base stays >= the floor: python sweep/pick_dose.py <run> 0.93"""
import json, sys, collections
import numpy as np
rows = [json.loads(l) for l in open(f"runs/{sys.argv[1]}.jsonl")]; floor = float(sys.argv[2])
by = collections.defaultdict(list)
for r in rows:
    if r["kind"] == "neuron": by[abs(r["dose"])].append(r["cos_base"])
ok = [d for d, v in by.items() if np.mean(v) >= floor]
print(f"{max(ok) if ok else min(by):g}")
