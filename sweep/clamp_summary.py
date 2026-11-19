"""Clamp run summary: python sweep/clamp_summary.py <run> [<run> ...] -> per kind/target mean concept-cosine gain,
similarity to baseline, mse and top lens; then per-prompt gains for the clamped feature."""
import json, sys, collections
import numpy as np
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]; meta = json.load(open(f"runs/{run}.meta.json"))
    print(f"\n== {run}  mask={meta.get('mask') or '-'} layers={meta.get('layers')}")
    by = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "baseline": by[(r["kind"], r["dose"])].append(r)
    print(f"{'kind':13s} {'target':>6s} {'n':>3s} {'dcos':>7s} {'cos_gg':>7s} {'cos_base':>8s}  top lens")
    for (k, d), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        attr = np.mean([r["d_attr"] for r in rs], 0); names = meta["attributes"]; top = np.argsort(-attr)[:3]
        print(f"{k:13s} {d:6g} {len(rs):3d} {np.mean([r['cos_concept']-r['base_cos_concept'] for r in rs]):+7.3f} "
              f"{np.mean([r['cos_concept'] for r in rs]):7.3f} {np.mean([r['cos_base'] for r in rs]):8.3f}  "
              + ", ".join(f"{names[i]}({attr[i]:+.1f})" for i in top))
    pp = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "clamp": pp[r["prompt"]][r["dose"]] = (round(r["cos_concept"] - r["base_cos_concept"], 3), round(r["cos_base"], 2))
    for p, v in pp.items(): print("   ", p[:30].ljust(30), {k: v[k] for k in sorted(v)})
