"""Steer the top candidate neurons of a contrast search, across layers: does a neuron that *reads* the concept also
*write* it? Takes the top-K neurons by |d| from stats/contrast_<concept>.json, pushes each at the given doses on the
sweep prompts, and records the same per-image measurements as steer_sweep plus the CLIP cosine to a concept text.
Rows carry their layer, so one run spans layers; random controls are drawn in each layer touched.

    python sweep/probe.py --concept goldengate --top 24 --doses=-2,-1,1,2 --text "the Golden Gate Bridge"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd_explorer import load_model
from sd_explorer.metrics import Lens, pixel_metrics
from sweep.prompts import SWEEP_PROMPTS, SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--neurons", default="", help='explicit "L68:2122,L10:3238" instead of the contrast top')
    ap.add_argument("--vectors", default="", help='steer along the concept vector (stats/concept_<concept>.npz) at '
                    'these layers, e.g. "23,41,46,37+41+46" (a+b+c pushes several layers at once)')
    ap.add_argument("--doses", default="-2,-1,1,2")
    ap.add_argument("--text", default="", help="concept text for a CLIP cosine column")
    ap.add_argument("--vec-drop-top", type=int, default=0, help="zero the k largest-|v| channels of each concept "
                    "vector before pushing (massive-activation channels in DiTs otherwise dominate the push)")
    ap.add_argument("--prompts", default="car,woman,fruit,lake")
    ap.add_argument("--random-per-layer", type=int, default=8)
    ap.add_argument("--run", default=None)
    ap.add_argument("--stats", default=None, help="stats prefix; default stats/sdxl-turbo or stats-flux/flux")
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux", "flux3"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--thumb", type=int, default=160)
    a = ap.parse_args()
    run = a.run or f"probe-{a.concept}"
    a.stats = a.stats or (f"stats-{a.model}/{a.model}" if a.model != "sdxl" else "stats/sdxl-turbo")
    sdir = os.path.dirname(a.stats)

    vec_groups = [[int(x) for x in g.split("+")] for g in a.vectors.split(",") if g] if a.vectors else []
    cands = []
    if a.neurons:
        cands = [(int(x.split(":")[0][1:]), int(x.split(":")[1])) for x in a.neurons.split(",")]
    elif vec_groups:
        pass
    else:
        top = json.load(open(f"{sdir}/contrast_{a.concept}.json"))["top"][: a.top]
        cands = [(r["layer"], r["neuron"]) for r in top]
    layers = sorted({li for li, _ in cands} | {li for g in vec_groups for li in g})
    cvec = np.load(f"{sdir}/concept_{a.concept}.npz") if vec_groups else None
    doses = [float(x) for x in a.doses.split(",")]
    prompts = a.prompts.split(",")
    sd, lens = load_model(a.model, prewarm=[SWEEP_PROMPTS[p] for p in prompts]), Lens()
    unit = {li: float(np.load(f"{a.stats}_L{li}.npz")["resid_norm_q10_50_90"][1]) for li in layers}
    text_emb = {p: lens.embed_text([SWEEP_PROMPTS[p]])[0] for p in prompts}
    concept_emb = lens.embed_text([a.text])[0] if a.text else None

    jobs = [{"layer": layers[0], "kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in prompts]
    units = [(li, "neuron", n) for li, n in cands] + [(li, "random", r) for li in layers for r in range(a.random_per_layer)]
    units += [(g[0], "vector", gi) for gi, g in enumerate(vec_groups)]  # neuron id = group index
    for li, kind, n in units:
        for p in prompts:
            for d in doses:
                jobs.append({"layer": li, "kind": kind, "neuron": n, "prompt": p, "dose": d})
    for j in jobs:
        j["seed"] = SEEDS[j["prompt"]]
        j["unit"] = unit[j["layer"]] if j["kind"] != "baseline" else 0.0
        j["rel_push"] = abs(j["dose"])

    os.makedirs("runs", exist_ok=True)
    tdir = f"thumbs/{run}"
    os.makedirs(tdir, exist_ok=True)

    def run_batch(batch):
        pushes = {}
        for i, j in enumerate(batch):
            if j["kind"] == "baseline":
                continue
            if j["kind"] == "vector":
                for li in vec_groups[j["neuron"]]:
                    v = torch.tensor(cvec[f"v_L{li}"], device=sd.device)
                    if a.vec_drop_top:
                        v = v.clone(); v[v.abs().topk(a.vec_drop_top).indices] = 0
                    pushes.setdefault(li, torch.zeros(len(batch), sd.dim[li], device=sd.device))
                    pushes[li][i] = v / (v.norm() + 1e-8) * (j["dose"] * unit[li])
                continue
            pushes.setdefault(j["layer"], torch.zeros(len(batch), sd.dim[j["layer"]], device=sd.device))
            pushes[j["layer"]][i] = sd.push_for(j["layer"], j["kind"], j["neuron"], j["dose"], j["unit"])
        img = sd.generate([SWEEP_PROMPTS[j["prompt"]] for j in batch], [j["seed"] for j in batch], pushes)
        emb = lens.embed_images(img)
        return img, pixel_metrics(img), emb, lens.attribute_scores(emb)

    bjobs = [j for j in jobs if j["kind"] == "baseline"]
    img, pm, emb, attr = run_batch(bjobs)
    base = {j["prompt"]: (img[i], emb[i], attr[i], {k: float(v[i]) for k, v in pm.items()}) for i, j in enumerate(bjobs)}
    out = open(f"runs/{run}.jsonl", "w")
    t0 = time.time()
    for bi in range(0, len(jobs), a.batch):
        batch = jobs[bi:bi + a.batch]
        img, pm, emb, attr = run_batch(batch)
        for i, j in enumerate(batch):
            p = j["prompt"]
            bimg, bemb, battr, bpix = base[p]
            row = dict(j)
            row.update({k: round(float(v[i]), 4) for k, v in pm.items()})
            if j["kind"] != "baseline":
                row.update({f"d_{k}": round(float(v[i]) - bpix[k], 4) for k, v in pm.items()})
                row["mse"] = round(float(((img[i] - bimg) ** 2).mean()), 5)
                row["cos_base"] = round(float(emb[i] @ bemb), 4)
                row["d_attr"] = [round(float(x), 2) for x in (attr[i] - battr).tolist()]
            row["cos_prompt"] = round(float(emb[i] @ text_emb[p]), 4)
            if concept_emb is not None:
                row["cos_concept"] = round(float(emb[i] @ concept_emb), 4)
            row["attr"] = [round(float(x), 2) for x in attr[i].tolist()]
            row["thumb"] = f"L{j['layer']}_{j['kind'][0]}{j['neuron']}_{p}_{j['dose']:+g}.jpg"
            if j["kind"] == "vector":
                row["group"] = "+".join(f"L{li}" for li in vec_groups[j["neuron"]])
            out.write(json.dumps(row) + "\n")
            t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(t).resize((a.thumb, a.thumb), Image.LANCZOS).save(f"{tdir}/{row['thumb']}", quality=85)
        print(f"{min(bi + a.batch, len(jobs))}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    out.close()
    json.dump({"layer": layers[0], "layers": layers, "name": "+".join(sd.names[li] for li in layers), "inner": 0,
               "dim": 0, "prompts": prompts, "doses": doses, "dose_unit": "rel", "attributes": lens.attributes,
               "random_controls": a.random_per_layer * len(layers), "n_neurons": len(cands), "concept": a.concept,
               "text": a.text, "probe": True, "model": a.model}, open(f"runs/{run}.meta.json", "w"), indent=1)

    # quick report: per candidate, the concept cosine shift and the lens
    rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
    base_cos = {p: next(r for r in rows if r["kind"] == "baseline" and r["prompt"] == p).get("cos_concept", 0) for p in prompts}
    print(f"\nbaseline cos to '{a.text}':", {p: round(c, 3) for p, c in base_cos.items()})
    print("unit               dose   Δcos_concept  Δsat    Δlum    cos_base  top attribute shifts")
    per = {}
    for r in rows:
        if r["kind"] == "baseline":
            continue
        per.setdefault((r["layer"], r["kind"], r["neuron"], r["dose"]), []).append(r)
    recs = []
    for k, rs in per.items():
        dc = float(np.mean([r.get("cos_concept", 0) - base_cos[r["prompt"]] for r in rs]))
        recs.append((dc, k, rs))
    for dc, k, rs in sorted(recs, key=lambda x: -x[0])[:24]:
        sh = np.mean([r["d_attr"] for r in rs], 0)
        top = ", ".join(f"{lens.attributes[i]}({sh[i]:+.1f})" for i in np.argsort(-sh)[:3])
        tag = rs[0].get("group", f"L{k[0]:2d}")
        print(f"{tag:>12s} {k[1][:3]} N{k[2]:5d} {k[3]:+5.1f}   {dc:+.4f}     {np.mean([r['d_sat'] for r in rs]):+.3f}  "
              f"{np.mean([r['d_lum'] for r in rs]):+.3f}  {np.mean([r['cos_base'] for r in rs]):.3f}    {top}")


if __name__ == "__main__":
    main()
