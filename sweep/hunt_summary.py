"""Stage 3 of the bridge hunt: rank every steered candidate across all layers by how much it moved the image toward
the Golden Gate (CLIP attribute shift as a z against that run's random-direction controls at the same dose), while
keeping the scene (cos to the unsteered image >= --min-cos). Prints the top, writes runs/hunt_ranking.json, and
builds sheets/hunt_top.jpg: the best candidates on the three prompts at their best dose against the baseline.

    python sweep/hunt_summary.py --min-cos 0.75 --top 16
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--min-cos", type=float, default=0.75)
ap.add_argument("--top", type=int, default=16)
ap.add_argument("--attr", default="the Golden Gate Bridge")
ap.add_argument("--by", default="score", choices=["score", "z"], help="score = gg z minus texture z (stripes/tiling/abstract/noise fool CLIP); z = raw gg z")
TEXTURE = ["stripes", "repeated tiling pattern", "an abstract pattern", "noise", "a glitchy distorted image", "polka dots", "a blurry out of focus image"]
ap.add_argument("--out", default="runs/hunt_ranking.json")
ap.add_argument("--sheet", default="sheets/hunt_top.jpg")
ap.add_argument("--prefix", default="hunt-L", help="run name prefix: hunt-L (sdxl) or hunt-flux-L")
ap.add_argument("--shortlist", default="runs/hunt_shortlist.json")
a = ap.parse_args()

short = json.load(open(a.shortlist)) if os.path.exists(a.shortlist) else {"layers": []}
wcos = {(L["layer"], int(n)): c for L in short["layers"] for n, c in L["write_cos"].items()}
rd = {(L["layer"], int(n)): c for L in short["layers"] for n, c in L["read_d"].items()}
cands = []
for metap in sorted(glob.glob(f"runs/{a.prefix}*.meta.json")):
    run = os.path.basename(metap)[: -len(".meta.json")]
    meta = json.load(open(metap))
    attrs = meta["attributes"]
    gi, si = attrs.index(a.attr), attrs.index("a suspension bridge")
    ti = [attrs.index(t) for t in TEXTURE if t in attrs]
    rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
    # per (kind, neuron, dose): mean over prompts of the gg shift, the suspension-bridge shift and cos_base
    agg = {}
    for r in rows:
        if r["kind"] == "baseline":
            continue
        e = agg.setdefault((r["kind"], r["neuron"], r["dose"]), {"gg": [], "sb": [], "cos": [], "tex": []})
        e["gg"].append(r["d_attr"][gi]); e["sb"].append(r["d_attr"][si]); e["cos"].append(r["cos_base"])
        e["tex"].append(max(r["d_attr"][j] for j in ti) if ti else 0.0)
    for k, e in agg.items():
        e["gg"], e["sb"], e["cos"], e["tex"] = float(np.mean(e["gg"])), float(np.mean(e["sb"])), float(np.mean(e["cos"])), float(np.mean(e["tex"]))
    null, nullt = {}, {}
    for (kind, n, dose), e in agg.items():
        if kind == "random":
            null.setdefault(dose, []).append(e["gg"]); nullt.setdefault(dose, []).append(e["tex"])
    null = {d: (float(np.mean(v)), float(np.std(v)) + 1e-6) for d, v in null.items()}
    nullt = {d: (float(np.mean(v)), float(np.std(v)) + 1e-6) for d, v in nullt.items()}
    for (kind, n, dose), e in agg.items():
        if kind != "neuron":
            continue
        mu, sdv = null.get(dose, (0.0, 1.0)); mut, sdt = nullt.get(dose, (0.0, 1.0))
        z, zt = (e["gg"] - mu) / sdv, (e["tex"] - mut) / sdt
        cands.append({"run": run, "layer": meta["layer"], "name": meta["name"], "neuron": n, "dose": dose, "gg": round(e["gg"], 3),
                      "z": round(z, 2), "ztex": round(zt, 2), "score": round(z - max(0.0, zt), 2), "sb": round(e["sb"], 3), "cos_base": round(e["cos"], 3),
                      "write_cos": wcos.get((meta["layer"], n)), "read_d": rd.get((meta["layer"], n)), "prompts": meta["prompts"]})
keep = [c for c in cands if c["cos_base"] >= a.min_cos]
key = a.by
keep.sort(key=lambda c: -c[key])
best = {}
for c in keep:  # one row per neuron: its best dose
    best.setdefault((c["layer"], c["neuron"]), c)
ranked = sorted(best.values(), key=lambda c: -c[key])
json.dump({"min_cos": a.min_cos, "n_candidates": len(cands), "n_kept": len(keep), "ranked": ranked[:500]}, open(a.out, "w"), indent=1)
print(f"{len(cands)} steered (neuron, dose) rows over {len(glob.glob('runs/hunt-L*.meta.json'))} layers; {len(keep)} keep the scene (cos >= {a.min_cos})")
print(f"ranked by {key}")
print(f"{'rank':>4} {'layer':>5} {'name':10s} {'neuron':>6} {'dose':>5} {'score':>6} {'gg z':>6} {'tex z':>6} {'gg Δ':>6} {'susp Δ':>6} {'cos':>5} {'w.cos':>6} {'read d':>6}")
for i, c in enumerate(ranked[: max(a.top, 40)]):
    print(f"{i + 1:4d} {c['layer']:5d} {c['name']:10s} {c['neuron']:6d} {c['dose']:+5g} {c['score']:6.1f} {c['z']:6.1f} {c['ztex']:6.1f} {c['gg']:+6.2f} {c['sb']:+6.2f} {c['cos_base']:5.2f} {c['write_cos'] if c['write_cos'] is not None else float('nan'):+6.2f} {c['read_d'] if c['read_d'] is not None else float('nan'):+6.2f}")

# contact sheet: baseline row, then the top candidates at their best dose
if ranked:
    prompts = ranked[0]["prompts"]
    S, LW = 160, 190
    top = ranked[: a.top]
    sheet = Image.new("RGB", (LW + S * len(prompts), S * (len(top) + 1)), (17, 17, 17))
    dr = ImageDraw.Draw(sheet)
    run0 = ranked[0]["run"]
    dr.text((6, 6), "baseline", fill=(230, 230, 230))
    for j, p in enumerate(prompts):
        f = f"thumbs/{run0}/b-1_{p}_+0.jpg"
        if os.path.exists(f):
            sheet.paste(Image.open(f).convert("RGB").resize((S, S)), (LW + j * S, 0))
    for i, c in enumerate(top):
        y = (i + 1) * S
        dr.text((6, y + 6), f"#{i + 1} L{c['layer']} N{c['neuron']} {c['dose']:+g}", fill=(230, 230, 230))
        dr.text((6, y + 22), f"score {c['score']:.1f} gg z {c['z']:.1f} tex z {c['ztex']:.1f} cos {c['cos_base']:.2f}", fill=(140, 140, 140))
        for j, p in enumerate(prompts):
            f = f"thumbs/{c['run']}/n{c['neuron']}_{p}_{c['dose']:+g}.jpg"
            if os.path.exists(f):
                sheet.paste(Image.open(f).convert("RGB").resize((S, S)), (LW + j * S, y))
    os.makedirs(os.path.dirname(a.sheet), exist_ok=True)
    sheet.save(a.sheet, quality=88)
    print("sheet", a.sheet)
