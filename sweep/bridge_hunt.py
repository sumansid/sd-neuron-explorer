"""Stage 1 of the bridge hunt: a shortlist of candidate Golden Gate *writers* in every layer of SDXL-Turbo (or FLUX with --model flux), cheap
enough to cover all 70 layers (332,800 neurons) in a minute, so that stage 2 (steering) only has to try ~170 per layer.

    python sweep/bridge_hunt.py --top-write 120 --top-read 40 --out runs/hunt_shortlist.json

Two ways a neuron can be a candidate:
* write alignment: the cosine between its down-projection column (what it adds to the residual stream when it
  fires) and the layer's Golden Gate concept vector v_L (stats/concept_goldengate.npz: mean residual difference
  between "... with the Golden Gate Bridge in the background" and plain prompts over 128 subjects). A neuron that
  writes along v_L is what a single "bridge neuron" would look like; the sign says which way to push it.
* reading: the per-neuron effect size d_L from the contrast search (stats/contrast_goldengate.npz). Readers are
  not writers, but a neuron that does both is the most plausible bridge neuron, so the strongest readers ride along.

Writes {layers: [{layer, name, inner, neurons, write_cos, read_d, max_abs_cos, n_write, n_read}], ...}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd_explorer import load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--top-write", type=int, default=120)
ap.add_argument("--top-read", type=int, default=40)
ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux", "flux3"])
ap.add_argument("--concept", default=None)
ap.add_argument("--contrast", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
sdir = "stats" if a.model == "sdxl" else f"stats-{a.model}"
a.concept = a.concept or f"{sdir}/concept_goldengate.npz"
a.contrast = a.contrast or f"{sdir}/contrast_goldengate.npz"
a.out = a.out or ("runs/hunt_shortlist.json" if a.model == "sdxl" else f"runs/hunt_{a.model}_shortlist.json")

sd = load_model(a.model)
concept = np.load(a.concept)
contrast = np.load(a.contrast)
out = {"top_write": a.top_write, "top_read": a.top_read, "layers": []}
print(f"{'layer':>5} {'name':10s} {'inner':>5} {'max|cos|':>8} {'#cos>0.1':>8} {'max|d|':>6}")
for li in range(len(sd.layers)):
    v = torch.tensor(concept[f"v_L{li}"], device=sd.device, dtype=torch.float32)
    W = sd.w_down(li).float()  # [dim, inner]
    cos = (W.T @ v) / (W.norm(dim=0) * v.norm() + 1e-8)  # [inner]
    cos = cos.cpu().numpy()
    d = contrast[f"d_L{li}"]
    writers = np.argsort(-np.abs(cos))[: a.top_write]
    readers = np.argsort(-np.abs(d))[: a.top_read]
    neurons = sorted(set(writers.tolist()) | set(readers.tolist()))
    out["layers"].append({"layer": li, "name": sd.names[li], "inner": int(sd.inner[li]), "neurons": neurons,
                          "write_cos": {int(n): round(float(cos[n]), 4) for n in neurons},
                          "read_d": {int(n): round(float(d[n]), 3) for n in neurons},
                          "max_abs_cos": round(float(np.abs(cos).max()), 4), "n_write": len(writers), "n_read": len(readers)})
    print(f"{li:5d} {sd.names[li]:10s} {sd.inner[li]:5d} {np.abs(cos).max():8.3f} {int((np.abs(cos) > 0.1).sum()):8d} {np.abs(d).max():6.2f}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"))
print("wrote", a.out, "total candidates", sum(len(L["neurons"]) for L in out["layers"]))
