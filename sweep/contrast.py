"""Targeted search: which neurons read a concept. For N subjects, generate the plain prompt and the concept prompt
("a photo of X" vs "a black and white photo of X") from the same noise, capture every FF neuron's token-mean
activation in both, and rank neurons in every layer by paired effect size. Also records the pixel metrics of both
images, so we know the prompt really moved saturation / luminance.

    python sweep/contrast.py --concept bw --n 96

Writes stats/contrast_{concept}.npz (per-layer d and t arrays) and stats/contrast_{concept}.json (top neurons
across all layers, both signs, plus per-layer summary).
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
from sd_explorer.metrics import pixel_metrics
from sweep.prompts import SUBJECTS, CONTRASTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="bw", choices=sorted(CONTRASTS))
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", default=None, help="stats dir; default stats or stats-flux")
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux", "flux3"])
    a = ap.parse_args()

    a.out = a.out or (f"stats-{a.model}" if a.model != "sdxl" else "stats")
    plain_t, concept_t = CONTRASTS[a.concept]
    subjects = SUBJECTS[: a.n]
    sd = load_model(a.model, prewarm=[t.format(x) for t in (plain_t, concept_t) for x in subjects])
    L = len(sd.layers)
    acts = {0: [[] for _ in range(L)], 1: [[] for _ in range(L)]}  # side -> layer -> list of [B, inner]
    side = {"v": 0}

    def cap(li, out):
        acts[side["v"]][li].append(out.float().mean(1).cpu())

    resid = {0: [[] for _ in range(L)], 1: [[] for _ in range(L)]}  # block outputs, token-mean, per side

    def cap_resid(li, out):
        resid[side["v"]][li].append(out.float().mean(1).cpu())

    pix = {0: [], 1: []}
    t0 = time.time()
    with sd.capturing(range(L), cap, cap_resid):
        for i in range(0, len(subjects), a.batch):
            chunk = subjects[i:i + a.batch]
            seeds = [7000 + i + j for j in range(len(chunk))]
            for s, tmpl in ((0, plain_t), (1, concept_t)):
                side["v"] = s
                img = sd.generate([tmpl.format(x) for x in chunk], seeds)
                m = pixel_metrics(img)
                pix[s].append(torch.stack([m["sat"], m["colorful"], m["lum"], m["contrast"]], 1).cpu())
            print(f"{i + len(chunk)}/{len(subjects)} {time.time() - t0:.0f}s", flush=True)

    P = {s: torch.cat(pix[s]).numpy() for s in (0, 1)}
    pix_summary = {k: {"plain": float(P[0][:, j].mean()), "concept": float(P[1][:, j].mean())}
                   for j, k in enumerate(["sat", "colorful", "lum", "contrast"])}
    print("pixel metrics plain -> concept:", json.dumps(pix_summary))

    rows, d_all, t_all, layer_summary = [], [], [], []
    for li in range(L):
        A = torch.cat(acts[0][li]).numpy()  # [N, inner]
        B = torch.cat(acts[1][li]).numpy()
        diff = B - A
        md, sd_ = diff.mean(0), diff.std(0) + 1e-6
        t = md / (sd_ / np.sqrt(len(diff)))
        pooled = np.sqrt((A.std(0) ** 2 + B.std(0) ** 2) / 2) + 1e-6
        d = md / pooled
        d_all.append(d.astype(np.float32)); t_all.append(t.astype(np.float32))
        layer_summary.append({"layer": li, "name": sd.names[li], "max_abs_d": float(np.abs(d).max()),
                              "n_abs_d_gt_1": int((np.abs(d) > 1).sum()), "n_abs_d_gt_2": int((np.abs(d) > 2).sum())})
        for n in np.argsort(-np.abs(d))[: a.top]:
            rows.append({"layer": li, "name": sd.names[li], "neuron": int(n), "d": float(d[n]), "t": float(t[n]),
                         "mean_plain": float(A[:, n].mean()), "mean_concept": float(B[:, n].mean()),
                         "frac_pos": float((diff[:, n] > 0).mean())})
    rows.sort(key=lambda r: -abs(r["d"]))
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/contrast_{a.concept}.npz", **{f"d_L{li}": d_all[li] for li in range(L)},
             **{f"t_L{li}": t_all[li] for li in range(L)})
    # concept vector per layer: mean(block output | concept) - mean(block output | plain), in residual space
    vec = {}
    for li in range(L):
        R0, R1 = torch.cat(resid[0][li]).numpy(), torch.cat(resid[1][li]).numpy()
        v = (R1 - R0).mean(0)
        proj = ((R1 - R0) @ (v / (np.linalg.norm(v) + 1e-8)))
        vec[f"v_L{li}"] = v.astype(np.float32)
        vec[f"sep_L{li}"] = np.float32(proj.mean() / (proj.std() + 1e-8))  # how consistently the pairs move along v
        layer_summary[li]["vec_norm"] = float(np.linalg.norm(v))
        layer_summary[li]["vec_sep"] = float(vec[f"sep_L{li}"])
    np.savez(f"{a.out}/concept_{a.concept}.npz", **vec)
    json.dump({"concept": a.concept, "templates": [plain_t, concept_t], "n": len(subjects), "pixels": pix_summary,
               "layers": layer_summary, "top": rows[: a.top * 4]}, open(f"{a.out}/contrast_{a.concept}.json", "w"), indent=1)
    print("\nlayers by max |d|:")
    for r in sorted(layer_summary, key=lambda r: -r["max_abs_d"])[:12]:
        print(f"  L{r['layer']:2d} {r['name']:12s} max|d| {r['max_abs_d']:.2f}  #|d|>1: {r['n_abs_d_gt_1']:4d}  #|d|>2: {r['n_abs_d_gt_2']}")
    print("\ntop neurons:")
    for r in rows[:30]:
        print(f"  L{r['layer']:2d} {r['name']:12s} N{r['neuron']:5d} d {r['d']:+.2f} t {r['t']:+.1f} "
              f"{r['mean_plain']:+.3f} -> {r['mean_concept']:+.3f} (pos {r['frac_pos']:.2f})")


if __name__ == "__main__":
    main()
