"""Steer every neuron of one FF layer up and down on a few fixed prompts, keep every image and what we can measure
of it. Random-direction controls (a random unit vector in the layer's residual space, same push size) get the same
treatment and give the null for "how often does an effect this size appear by chance".

Dose units ("rel", default): fractions of the median per-token residual norm leaving the block, so every neuron
gets the same push size. "absmax": multiples of abs_max[n] * ||W_down[:, n]||, the neuron's own range.

    python sweep/steer_sweep.py --layer 40 --neurons all --doses=-1,1 --random-controls 128 --run L40
    python sweep/steer_sweep.py --layer 40 --neurons random:48 --doses=-4,-2,-1,-0.5,0.5,1,2,4 --run calib-L40

Each row of runs/<run>.jsonl: layer, kind, neuron, prompt, dose, rel_push, seed, pixel metrics (sat, colorful, lum,
contrast) and their deltas vs the unsteered baseline, mse and CLIP cosine to the baseline image, CLIP cosine to the
prompt, and the full CLIP attribute-score vector. Thumbnails go to thumbs/<run>/. CLIP image embeddings go to
runs/<run>.emb.npy (row order = jsonl order). Rerunning resumes.
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


def parse_neurons(spec, n, seed):
    if spec == "all":
        return list(range(n))
    if spec.startswith("random:"):
        return sorted(np.random.default_rng(seed + 1).choice(n, int(spec.split(":")[1]), replace=False).tolist())
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def thumb_name(job):
    return f"{job['kind'][0]}{job['neuron']}_{job['prompt']}_{job['dose']:+g}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux", "flux3"])
    ap.add_argument("--neurons", default="all")
    ap.add_argument("--prompts", default="car,woman,fruit,lake")
    ap.add_argument("--doses", default="-1,1")
    ap.add_argument("--dose-unit", default="rel", choices=["rel", "absmax"])
    ap.add_argument("--random-controls", type=int, default=128)
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--full-vae", action="store_true", help="decode with the full VAE (demo quality)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.prompts == "demo":
        from sweep.prompts import DEMO
        a.prompts = DEMO

    a.stats = a.stats or (f"stats-{a.model}/{a.model}" if a.model != "sdxl" else "stats/sdxl-turbo")
    st = np.load(f"{a.stats}_L{a.layer}.npz")
    resid_p50 = float(st["resid_norm_q10_50_90"][1])
    abs_max, col_norm = st["abs_max"], st["col_norm"]
    sd = load_model(a.model, prewarm=[SWEEP_PROMPTS[p] for p in a.prompts.split(",")])
    lens = Lens()
    li = a.layer
    neurons = parse_neurons(a.neurons, sd.inner[li], a.seed)
    prompts = a.prompts.split(",")
    doses = [float(x) for x in a.doses.split(",")]

    def unit_for(kind, n):
        if a.dose_unit == "rel":
            return resid_p50
        return float(abs_max[n] * col_norm[n]) if kind == "neuron" else float(np.median(abs_max * col_norm))

    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in prompts]
    units = [("neuron", n) for n in neurons] + [("random", r) for r in range(a.random_controls)]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(units)
    for kind, n in units:
        for p in prompts:
            for d in doses:
                jobs.append({"kind": kind, "neuron": int(n), "prompt": p, "dose": d})
    for j in jobs:
        j["layer"] = li
        j["seed"] = SEEDS[j["prompt"]]
        j["unit"] = unit_for(j["kind"], j["neuron"]) if j["kind"] != "baseline" else 0.0
        j["rel_push"] = round(abs(j["dose"]) * j["unit"] / resid_p50, 4)

    os.makedirs("runs", exist_ok=True)
    tdir = f"thumbs/{a.run}"
    os.makedirs(tdir, exist_ok=True)
    out_path, emb_path = f"runs/{a.run}.jsonl", f"runs/{a.run}.emb.npy"
    done = set()
    embs = []
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["kind"], r["neuron"], r["prompt"], r["dose"]))
        if os.path.exists(emb_path):
            embs = list(np.load(emb_path))
    todo = [j for j in jobs if (j["kind"], j["neuron"], j["prompt"], j["dose"]) not in done]
    print(f"layer L{li} {sd.names[li]} inner {sd.inner[li]} resid p50 {resid_p50:.2f}; {len(jobs)} jobs, {len(todo)} to do")

    # baselines first (they anchor every delta); keep their images + embeddings in memory
    base_img, base_emb, base_attr, base_pix = {}, {}, {}, {}
    def run_batch(batch):
        pushes = torch.zeros(len(batch), sd.dim[li], device=sd.device)
        for i, j in enumerate(batch):
            if j["kind"] != "baseline":
                pushes[i] = sd.push_for(li, j["kind"], j["neuron"], j["dose"], j["unit"])
        img = sd.generate([SWEEP_PROMPTS[j["prompt"]] for j in batch], [j["seed"] for j in batch], {li: pushes}, full_vae=a.full_vae)
        pm = pixel_metrics(img)
        emb = lens.embed_images(img)
        attr = lens.attribute_scores(emb)
        return img, pm, emb, attr

    text_emb = {p: lens.embed_text([SWEEP_PROMPTS[p]])[0] for p in prompts}
    bjobs = [j for j in jobs if j["kind"] == "baseline"]
    img, pm, emb, attr = run_batch(bjobs)
    for i, j in enumerate(bjobs):
        p = j["prompt"]
        base_img[p], base_emb[p], base_attr[p] = img[i], emb[i], attr[i]
        base_pix[p] = {k: float(v[i]) for k, v in pm.items()}

    f = open(out_path, "a")
    t0, n_done = time.time(), 0
    for bi in range(0, len(todo), a.batch):
        batch = todo[bi:bi + a.batch]
        img, pm, emb, attr = run_batch(batch)
        for i, j in enumerate(batch):
            p = j["prompt"]
            row = dict(j)
            row.update({k: round(float(v[i]), 4) for k, v in pm.items()})
            if j["kind"] != "baseline":
                row.update({f"d_{k}": round(float(v[i]) - base_pix[p][k], 4) for k, v in pm.items()})
                row["mse"] = round(float(((img[i] - base_img[p]) ** 2).mean()), 5)
                row["cos_base"] = round(float(emb[i] @ base_emb[p]), 4)
                row["d_attr"] = [round(float(x), 2) for x in (attr[i] - base_attr[p]).tolist()]
            row["cos_prompt"] = round(float(emb[i] @ text_emb[p]), 4)
            row["attr"] = [round(float(x), 2) for x in attr[i].tolist()]
            row["thumb"] = thumb_name(j)
            f.write(json.dumps(row) + "\n")
            embs.append(emb[i].cpu().numpy().astype(np.float16))
            t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(t).resize((a.thumb, a.thumb), Image.LANCZOS).save(f"{tdir}/{row['thumb']}", quality=85)
        n_done += len(batch)
        if (bi // a.batch) % 20 == 0:
            f.flush()
            np.save(emb_path, np.stack(embs))
            rate = n_done / (time.time() - t0)
            print(f"{n_done}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo) - n_done) / max(rate, 1e-6) / 60:.0f} min", flush=True)
    f.close()
    np.save(emb_path, np.stack(embs))
    json.dump({"layer": li, "name": sd.names[li], "inner": sd.inner[li], "dim": sd.dim[li], "resid_p50": resid_p50,
               "prompts": prompts, "doses": doses, "dose_unit": a.dose_unit, "attributes": lens.attributes,
               "random_controls": a.random_controls, "n_neurons": len(neurons)},
              open(f"runs/{a.run}.meta.json", "w"), indent=1)
    print("done", out_path)


if __name__ == "__main__":
    main()
