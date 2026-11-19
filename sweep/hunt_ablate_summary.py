"""Stage 4 summary: which candidates REMOVE the bridge from prompts that ask for it when pushed down?
For every (neuron, dose) in runs/ablate-L*.jsonl: the mean Golden Gate attribute shift over the three bridge prompts
(negative = bridge weaker), as a z against the run's random-direction controls at the same dose, and the cos to the
unsteered image. A bridge neuron shows a large negative shift at the negative doses while the random controls do not.
Writes runs/ablate_ranking.json and sheets/hunt_ablate.jpg (bridge prompts: baseline row, then the strongest removers).

    python sweep/hunt_ablate_summary.py --top 16
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=16)
ap.add_argument("--min-cos", type=float, default=0.55, help="removal changes the picture more than addition; a lower floor")
ap.add_argument("--out", default="runs/ablate_ranking.json")
ap.add_argument("--sheet", default="sheets/hunt_ablate.jpg")
a = ap.parse_args()

add = {}
if os.path.exists("runs/hunt_ranking.json"):
    for c in json.load(open("runs/hunt_ranking.json"))["ranked"]:
        add[(c["layer"], c["neuron"])] = c
rows_out = []
for metap in sorted(glob.glob("runs/ablate-L*.meta.json")):
    run = os.path.basename(metap)[: -len(".meta.json")]
    meta = json.load(open(metap))
    gi = meta["attributes"].index("the Golden Gate Bridge")
    rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
    base_gg = {}
    for r in rows:
        if r["kind"] == "baseline":
            base_gg[r["prompt"]] = r["attr"][gi] if "attr" in r else None
    agg = {}
    for r in rows:
        if r["kind"] == "baseline":
            continue
        e = agg.setdefault((r["kind"], r["neuron"], r["dose"]), {"gg": [], "cos": []})
        e["gg"].append(r["d_attr"][gi]); e["cos"].append(r["cos_base"])
    for e in agg.values():
        e["gg"], e["cos"] = float(np.mean(e["gg"])), float(np.mean(e["cos"]))
    null = {}
    for (kind, n, dose), e in agg.items():
        if kind == "random":
            null.setdefault(dose, []).append(e["gg"])
    null = {d: (float(np.mean(v)), float(np.std(v)) + 1e-6) for d, v in null.items()}
    for (kind, n, dose), e in agg.items():
        if kind != "neuron":
            continue
        mu, sdv = null.get(dose, (0.0, 1.0))
        rows_out.append({"run": run, "layer": meta["layer"], "name": meta["name"], "neuron": n, "dose": dose, "gg": round(e["gg"], 3),
                         "z": round((e["gg"] - mu) / sdv, 2), "cos_base": round(e["cos"], 3), "prompts": meta["prompts"],
                         "add_z": add.get((meta["layer"], n), {}).get("z"), "add_dose": add.get((meta["layer"], n), {}).get("dose")})
neg = [r for r in rows_out if r["dose"] < 0 and r["cos_base"] >= a.min_cos]
neg.sort(key=lambda r: r["z"])  # most negative first = strongest removal
best = {}
for r in neg:
    best.setdefault((r["layer"], r["neuron"]), r)
ranked = sorted(best.values(), key=lambda r: r["z"])
# the prize: units that both add (stage 2 z high) and remove (here z very negative)
both = sorted([r for r in ranked if r["add_z"] is not None], key=lambda r: -(r["add_z"] or 0) + r["z"])
json.dump({"ranked_removers": ranked[:300], "add_and_remove": both[:100]}, open(a.out, "w"), indent=1)
print(f"{len(rows_out)} steered rows; {len(neg)} negative pushes keep cos >= {a.min_cos}")
print(f"{'rank':>4} {'layer':>5} {'name':10s} {'neuron':>6} {'dose':>5} {'remove z':>8} {'gg Δ':>6} {'cos':>5} {'add z':>6}")
for i, r in enumerate(ranked[: max(a.top, 30)]):
    print(f"{i + 1:4d} {r['layer']:5d} {r['name']:10s} {r['neuron']:6d} {r['dose']:+5g} {r['z']:8.1f} {r['gg']:+6.2f} {r['cos_base']:5.2f} {r['add_z'] if r['add_z'] is not None else float('nan'):6.1f}")
print("\nadd AND remove (sum of stage-2 add z and stage-4 removal |z|):")
for i, r in enumerate(both[:15]):
    print(f"{i + 1:4d} L{r['layer']} N{r['neuron']}  add z {r['add_z']:5.1f} at {r['add_dose']:+g} · remove z {r['z']:6.1f} at {r['dose']:+g} · cos {r['cos_base']:.2f}")
if ranked:
    prompts = ranked[0]["prompts"]; S, LW = 160, 200
    top = ranked[: a.top]
    sheet = Image.new("RGB", (LW + S * len(prompts), S * (len(top) + 1)), (17, 17, 17)); dr = ImageDraw.Draw(sheet)
    dr.text((6, 6), "baseline (bridge prompts)", fill=(230, 230, 230))
    for j, p in enumerate(prompts):
        f = f"thumbs/{ranked[0]['run']}/b-1_{p}_+0.jpg"
        if os.path.exists(f):
            sheet.paste(Image.open(f).convert("RGB").resize((S, S)), (LW + j * S, 0))
    for i, r in enumerate(top):
        y = (i + 1) * S
        dr.text((6, y + 6), f"#{i + 1} L{r['layer']} N{r['neuron']} {r['dose']:+g}", fill=(230, 230, 230))
        dr.text((6, y + 22), f"remove z {r['z']:.1f} · add z {r['add_z'] if r['add_z'] is not None else float('nan'):.1f}", fill=(140, 140, 140))
        for j, p in enumerate(prompts):
            f = f"thumbs/{r['run']}/n{r['neuron']}_{p}_{r['dose']:+g}.jpg"
            if os.path.exists(f):
                sheet.paste(Image.open(f).convert("RGB").resize((S, S)), (LW + j * S, y))
    os.makedirs(os.path.dirname(a.sheet), exist_ok=True)
    sheet.save(a.sheet, quality=88); print("sheet", a.sheet)
