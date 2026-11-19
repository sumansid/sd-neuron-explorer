"""Push several neurons at once (across layers) on a set of prompts and report what the CLIP lens says, with a contact
sheet. For composing a bridge neuron with a colour or scene dial, e.g. "arch bridge + red tint":

    python sweep/combo.py --name arch_red --units 17:868:3,68:2092:1 --prompts demo
    python sweep/combo.py --name bridges --units 17:868:2,18:695:2,21:968:2 --prompts lake,boat,desert,street

--units  layer:neuron:dose[,layer:neuron:dose...]  (dose in multiples of the layer's median residual norm, 'rel')
Writes runs/combo-<name>.jsonl (one row per prompt with the attribute shifts vs the unsteered image) and
sheets/combo_<name>.jpg (baseline row, steered row).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd_explorer import load_model  # noqa: E402
from sd_explorer.metrics import Lens, pixel_metrics  # noqa: E402
from sweep.prompts import SEEDS, SWEEP_PROMPTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--units", required=True)
ap.add_argument("--prompts", default="demo")
ap.add_argument("--model", default="sdxl")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--full-vae", action="store_true")
ap.add_argument("--tile", type=int, default=192, help="tile size on the sheet")
ap.add_argument("--save", action="store_true", help="also save every image full size under gallery/<name>/")
a = ap.parse_args()
if a.prompts == "demo":
    from sweep.prompts import DEMO
    a.prompts = DEMO
prompts = a.prompts.split(",")
units = [tuple(float(x) if i == 2 else int(x) for i, x in enumerate(u.split(":"))) for u in a.units.split(",")]

sd = load_model(a.model, prewarm=[SWEEP_PROMPTS[p] for p in prompts])
sd.h = sd.w = a.size
lens = Lens()
stats = {li: float(np.load(f"stats/sdxl-turbo_L{li}.npz")["resid_norm_q10_50_90"][1]) for li, _, _ in units}
B = len(prompts)
pushes = {}
for li, n, dose in units:
    pushes.setdefault(li, torch.zeros(B, sd.dim[li], device=sd.device))
    pushes[li] += sd.push_for(li, "neuron", n, dose, stats[li])[None, :]
texts = [SWEEP_PROMPTS[p] for p in prompts]
seeds = [SEEDS[p] for p in prompts]
with torch.no_grad():
    base = sd.generate(texts, seeds, None, full_vae=a.full_vae)
    steer = sd.generate(texts, seeds, pushes, full_vae=a.full_vae)
    eb, es = lens.embed_images(base), lens.embed_images(steer)
    ab, as_ = lens.attribute_scores(eb), lens.attribute_scores(es)
gi = lens.attributes.index("the Golden Gate Bridge"); si = lens.attributes.index("a suspension bridge")
os.makedirs("runs", exist_ok=True); os.makedirs("sheets", exist_ok=True)
out = open(f"runs/combo-{a.name}.jsonl", "w")
print(f"{'prompt':10s} {'cos':>5s} {'gg Δ':>6s} {'susp Δ':>6s}  top shifts")
for i, p in enumerate(prompts):
    d = (as_[i] - ab[i]).cpu().numpy()
    cos = float(torch.nn.functional.cosine_similarity(eb[i], es[i], dim=0))
    top = [(lens.attributes[j], round(float(d[j]), 2)) for j in np.argsort(-d)[:4]]
    row = {"name": a.name, "units": units, "prompt": p, "cos_base": round(cos, 4), "gg": round(float(d[gi]), 3), "susp": round(float(d[si]), 3), "top": top}
    out.write(json.dumps(row) + "\n")
    print(f"{p:10s} {cos:5.2f} {d[gi]:+6.2f} {d[si]:+6.2f}  {top}")
S = a.tile
if a.save:
    os.makedirs(f"gallery/{a.name}", exist_ok=True)
    for i, p in enumerate(prompts):
        for tag, img in (("base", base), ("steered", steer)):
            t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(t).save(f"gallery/{a.name}/{p}_{tag}.jpg", quality=92)
sheet = Image.new("RGB", (S * B, S * 2 + 20), (17, 17, 17)); dr = ImageDraw.Draw(sheet)
dr.text((4, 4), f"{a.name}: units {units} · top: baseline, bottom: steered", fill=(230, 230, 230))
for i in range(B):
    for r, img in enumerate((base, steer)):
        t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sheet.paste(Image.fromarray(t).resize((S, S)), (i * S, 20 + r * S))
sheet.save(f"sheets/combo_{a.name}.jpg", quality=88)
print("sheet", f"sheets/combo_{a.name}.jpg")
