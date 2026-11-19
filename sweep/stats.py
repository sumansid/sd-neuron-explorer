"""Per-neuron statistics for every FF layer of the UNet over a prompt corpus: abs_max and mean |activation|
per neuron, the per-token residual norm leaving each block (q10/50/90), and down-column norms.

    python sweep/stats.py --n 128 --out stats/sdxl-turbo

Writes stats/sdxl-turbo_L{li}.npz per layer and stats/sdxl-turbo.json with the layer table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd_explorer import load_model
from sweep.prompts import SUBJECTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="prompts (subjects) to run")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="stats prefix; default stats/sdxl-turbo or stats-flux/flux")
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux", "flux3"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    a.out = a.out or (f"stats-{a.model}/{a.model}" if a.model != "sdxl" else "stats/sdxl-turbo")
    prompts = [f"a photo of {s}" for s in SUBJECTS[: a.n]]
    sd = load_model(a.model, prewarm=prompts)
    L = len(sd.layers)
    abs_max = [torch.zeros(k, device=sd.device) for k in sd.inner]
    mean_abs = [torch.zeros(k, device=sd.device, dtype=torch.float64) for k in sd.inner]
    pos_mean = [torch.zeros(k, device=sd.device, dtype=torch.float64) for k in sd.inner]
    count = [0] * L
    resid = [[] for _ in range(L)]
    rng = np.random.default_rng(a.seed)

    def cap(li, out):
        x = out.float()  # [B, T, inner]
        abs_max[li] = torch.maximum(abs_max[li], x.abs().amax((0, 1)))
        mean_abs[li] += x.abs().sum((0, 1)).double()
        pos_mean[li] += x.sum((0, 1)).double()
        count[li] += x.shape[0] * x.shape[1]

    def res(li, out):
        n = out.float().norm(dim=-1).flatten()
        idx = torch.randint(0, n.numel(), (min(4096, n.numel()),), device=n.device)
        resid[li].append(n[idx].cpu())

    t0 = time.time()
    with sd.capturing(range(L), cap, res):
        for i in range(0, len(prompts), a.batch):
            chunk = prompts[i:i + a.batch]
            seeds = [a.seed * 100000 + i + j for j in range(len(chunk))]
            sd.generate(chunk, seeds)
            print(f"{i + len(chunk)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    table = []
    for li in range(L):
        w = sd.w_down(li).float()
        col_norm = w.norm(dim=0).cpu().numpy()
        r = torch.cat(resid[li]).numpy()
        q = np.percentile(r, [10, 50, 90])
        rec = dict(layer=li, name=sd.names[li], inner=sd.inner[li], dim=sd.dim[li],
                   resid_norm_q10_50_90=q.tolist(), abs_max_median=float(abs_max[li].median()),
                   col_norm_median=float(np.median(col_norm)), tokens=count[li])
        table.append(rec)
        np.savez(f"{a.out}_L{li}.npz", layer=li, name=sd.names[li], abs_max=abs_max[li].cpu().numpy(),
                 mean_abs=(mean_abs[li] / max(count[li], 1)).float().cpu().numpy(),
                 mean=(pos_mean[li] / max(count[li], 1)).float().cpu().numpy(),
                 resid_norm_q10_50_90=q, col_norm=col_norm)
        print(f"L{li:2d} {sd.names[li]:12s} inner {sd.inner[li]:5d} resid p50 {q[1]:7.2f} abs_max p50 "
              f"{rec['abs_max_median']:6.2f} col p50 {rec['col_norm_median']:.3f}")
    json.dump(table, open(f"{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
