"""Which units of a run push one lens attribute (or one pixel metric) the most, against the random null.

    python sweep/attr_rank.py L18 --attr "the Golden Gate Bridge" --show 12
    python sweep/attr_rank.py L18 --metric d_sat --sign neg --show 12
"""
import argparse
import collections
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--attr", default="")
ap.add_argument("--metric", default="")
ap.add_argument("--sign", default="pos", choices=["pos", "neg"])
ap.add_argument("--show", type=int, default=12)
a = ap.parse_args()
meta = json.load(open(f"runs/{a.run}.meta.json"))
rows = [json.loads(l) for l in open(f"runs/{a.run}.jsonl") if '"baseline"' not in l]
ai = meta["attributes"].index(a.attr) if a.attr else None
per = collections.defaultdict(list)
for r in rows:
    per[(r["kind"], r["neuron"], r["dose"])].append(r["d_attr"][ai] if ai is not None else r[a.metric])
vals = {k: float(np.mean(v)) for k, v in per.items()}
sgn = 1 if a.sign == "pos" else -1
for d in sorted({k[2] for k in vals}):
    null = np.array([v for k, v in vals.items() if k[0] == "random" and k[2] == d])
    mu, sd = null.mean(), null.std() + 1e-6
    neu = sorted(((sgn * (v - mu) / sd, k[1], v) for k, v in vals.items() if k[0] == "neuron" and k[2] == d), reverse=True)
    print(f"dose {d:+g}: random {'z' if ai is not None else a.metric} mean {mu:+.3f} sd {sd:.3f} max {sgn * (sgn * null).max():+.3f}; "
          f"neurons above all controls: {sum(sgn * v > sgn * null.max() if sgn > 0 else v < null.min() for _, _, v in neu)}")
    for z, n, v in neu[: a.show]:
        print(f"   N{n:5d}  value {v:+.3f}  z {sgn * z:+.1f}")
