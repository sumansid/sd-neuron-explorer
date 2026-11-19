"""Anthropic-style feature clamping on a diffusion transformer: at one layer, read each image token's activation of
an SAE feature with the encoder and set it to a target value by adding (target - a_f) * d_f along the feature's
decoder direction (in raw residual space). Targets are multiples of the feature's largest activation seen on the
corpus, so 1.0 = "as strong as it ever naturally fires", 0 = ablation. Random features of the same SAE at the same
targets are the controls.

    python sweep/sae_steer.py --data sae/flux --layer 24 --features 1234 --targets 0,1,2,4,8 --text "the Golden Gate Bridge"

Rows go to runs/<run>.jsonl (kind "clamp"/"clamp_random", neuron = feature id, dose = target multiple), thumbnails
to thumbs/<run>/, and a prompt x target contact sheet per feature to sheets/<model>_sae_L<li>_f<F>_clamp.jpg.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd_explorer import load_model
from sd_explorer.metrics import Lens, pixel_metrics
from sweep.prompts import SWEEP_PROMPTS, SEEDS
from sweep.sae_find import SAE

EXTRA_PROMPTS = {
    "ggcar": ("a photo of a red sports car on a city street with the Golden Gate Bridge in the background", 1001),
    "ggbridge": ("the Golden Gate Bridge at sunset", 1005),
    # the "any prompt" demo set: twelve unrelated scenes, none mentions a bridge
    "cat": ("a photo of a cat sleeping on a windowsill", 2001),
    "kitchen": ("a photo of a modern kitchen with marble counters", 2002),
    "astronaut": ("an astronaut floating in space above Earth", 2003),
    "breakfast": ("a plate of pancakes with maple syrup and berries", 2004),
    "castle": ("a medieval castle on a hill in the fog", 2005),
    "dog": ("a golden retriever running on a beach", 2006),
    "street": ("a busy Tokyo street at night with neon signs", 2007),
    "library": ("an old library with tall wooden bookshelves", 2008),
    "desert": ("sand dunes in the desert at noon", 2009),
    "portrait": ("a studio portrait of an old man with a grey beard", 2010),
    "boat": ("a wooden rowing boat on a calm river", 2011),
    "forest": ("a path through a pine forest in winter snow", 2012),
}
DEMO = "cat,kitchen,astronaut,breakfast,castle,dog,street,library,desert,portrait,boat,forest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sae/flux")
    ap.add_argument("--layer", required=True, help="one layer, or several: 21,24,26")
    ap.add_argument("--features", required=True, help="feature ids; for several layers give one per layer joined "
                    "with ':' (f21:f24:f26), several such groups comma separated")
    ap.add_argument("--targets", default="0,1,2,4,8")
    ap.add_argument("--random-features", type=int, default=6)
    ap.add_argument("--prompts", default="car,woman,fruit,lake,ggcar,ggbridge")
    ap.add_argument("--text", default="the Golden Gate Bridge")
    ap.add_argument("--run", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", default="", help="clamp only part of the image: 'top:0.5' = the top half of token rows, "
                    "'bottom:0.4', 'left:0.5', 'right:0.5', 'band:0.35:0.55' (rows between two fractions)")
    ap.add_argument("--size", type=int, default=512, help="image side in px (the token grid scales with it)")
    ap.add_argument("--steps", type=int, default=0, help="denoising steps (0 = the model's default)")
    ap.add_argument("--full-vae", action="store_true", help="decode with the full VAE instead of the tiny one")
    ap.add_argument("--from-step", type=int, default=0, help="apply the clamp only from this denoising step on (0 = all)")
    ap.add_argument("--to-step", type=int, default=0, help="stop clamping before this step (0 = never)")
    ap.add_argument("--save-full", default="", help="also save full-size JPEGs of every image into this folder")
    a = ap.parse_args()
    if a.prompts == "demo":
        a.prompts = DEMO
    meta = json.load(open(f"{a.data}/meta.json"))
    model = meta["model"]
    layers = [int(x) for x in a.layer.split(",")]
    li = layers[0]
    run = a.run or f"{model}-sae-L{'+'.join(map(str, layers))}-clamp"
    saes = {l: SAE(f"{a.data}/sae_L{l}.pt") for l in layers}
    fmaxs = {l: np.load(f"{a.data}/feat_L{l}_max.npy").astype(np.float32) for l in layers}
    sae, fmax = saes[li], fmaxs[li]
    # one entry per layer joined with ':', each entry a feature id or a family 'f1+f2+f3' clamped together
    groups = [tuple(tuple(int(x) for x in part.split("+")) for part in g.split(":")) for g in a.features.split(",")]
    assert all(len(g) == len(layers) for g in groups), "give one feature (or family) per layer, joined with ':'"
    groups = [tuple(fam if len(fam) > 1 else fam[0] for fam in g) for g in groups]
    first = lambda x: x[0] if isinstance(x, tuple) else x
    feats = [first(g[0]) for g in groups]  # the first layer's (first) feature id names the group
    group_of = {first(g[0]): g for g in groups}
    rng = np.random.default_rng(a.seed)
    rand = []
    for _ in range(a.random_features):
        g = tuple(int(rng.choice(np.where((fmaxs[l] > 0).mean(0) > 0.01)[0])) for l in layers)
        rand.append(g[0]); group_of[g[0]] = g
    targets = [float(x) for x in a.targets.split(",")]
    prompts = {}
    for p in a.prompts.split(","):
        prompts[p] = (SWEEP_PROMPTS[p], SEEDS[p]) if p in SWEEP_PROMPTS else EXTRA_PROMPTS[p]
    sd = load_model(model, prewarm=[v[0] for v in prompts.values()])
    if a.size != 512:
        sd.h = sd.w = a.size
        sd.t_img = (a.size // 16) * (a.size // 16)
    if a.steps:
        sd.steps = a.steps
    lens = Lens()
    text_emb = lens.embed_text([a.text])[0]
    dev = sd.device

    units = [("clamp", f) for f in feats] + [("clamp_random", f) for f in rand]
    G = a.size // 16  # token grid side at this resolution
    tok_mask = torch.ones(G * G, device=dev)
    if a.mask:
        parts = a.mask.split(":"); side, frac = parts[0], float(parts[1])
        rows = torch.arange(G * G, device=dev) // G; cols = torch.arange(G * G, device=dev) % G
        tok_mask = {"top": rows < frac * G, "bottom": rows >= (1 - frac) * G, "left": cols < frac * G,
                    "right": cols >= (1 - frac) * G,
                    "band": (rows >= frac * G) & (rows < float(parts[2]) * G) if side == "band" else None}[side].float()
    fam = lambda x: list(x) if isinstance(x, tuple) else [x]
    scale = {f: float(fmax[:, f].max()) for _, f in units}  # reported scale: the first layer's (first) feature
    scales = {f: {l: {ff: float(fmaxs[l][:, ff].max()) for ff in fam(g)} for l, g in zip(layers, group_of[f])} for _, f in units}
    jobs = [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in prompts]
    for kind, f in units:
        for p in prompts:
            for t in targets:
                jobs.append({"kind": kind, "neuron": f, "prompt": p, "dose": t})
    for j in jobs:
        j["layer"], j["seed"] = li, prompts[j["prompt"]][1]
        j["target_abs"] = round(j["dose"] * scale.get(j["neuron"], 0.0), 3)
    os.makedirs("runs", exist_ok=True)
    tdir = f"thumbs/{run}"
    os.makedirs(tdir, exist_ok=True)

    def run_batch(batch):
        # all rows of a batch share one feature (or are baselines); per-row target in absolute units, NaN = untouched
        f = next((j["neuron"] for j in batch if j["kind"] != "baseline"), None)
        tgt = torch.tensor([j["target_abs"] if j["kind"] != "baseline" else float("nan") for j in batch], device=dev)
        if f is not None:
            mult = torch.tensor([j["dose"] if j["kind"] != "baseline" else float("nan") for j in batch], device=dev)
            edits = {}
            counters = {l: 0 for l in layers}  # denoising step seen by each layer's hook in this batch
            for l, g in zip(layers, group_of[f]):
                def make(l=l, ffs=fam(g)):
                    s_ = saes[l]
                    enc_w = s_.W_enc[:, ffs]; enc_b = s_.b_enc[ffs]  # [D, n], [n]
                    dec = s_.W_dec[ffs] * s_.sigma  # [n, D] raw-space contribution of one unit of each feature
                    t_abs = mult[:, None] * torch.tensor([scales[f][l][ff] for ff in ffs], device=dev)[None, :]  # [B, n]

                    def edit(hid):
                        step = counters[l]; counters[l] += 1
                        if step < a.from_step or (a.to_step and step >= a.to_step):
                            return hid
                        xs = (hid.float() - s_.mu) / s_.sigma
                        pre = (xs - s_.b_pre) @ enc_w + enc_b  # [B, T, n]
                        act = torch.relu(pre)
                        delta = t_abs[:, None, :] - act  # [B, T, n]
                        delta = torch.where(torch.isnan(delta), torch.zeros_like(delta), delta) * tok_mask[None, :, None]
                        return (hid.float() + delta @ dec).to(hid.dtype)
                    return edit
                edits[l] = make()
            ctx = sd.editing(edits)
        else:
            import contextlib
            ctx = contextlib.nullcontext()
        with ctx:
            img = sd.generate([prompts[j["prompt"]][0] for j in batch], [j["seed"] for j in batch], full_vae=a.full_vae)
        emb = lens.embed_images(img)
        return img, pixel_metrics(img), emb, lens.attribute_scores(emb)

    bjobs = [j for j in jobs if j["kind"] == "baseline"]
    img, pm, emb, attr = run_batch(bjobs)
    base = {j["prompt"]: (img[i], emb[i], attr[i], {k: float(v[i]) for k, v in pm.items()}) for i, j in enumerate(bjobs)}
    out = open(f"runs/{run}.jsonl", "w")
    rows = []
    t0 = time.time()
    # group jobs by unit so each batch shares a feature
    for kind, f in units:
        ujobs = [j for j in jobs if j["kind"] == kind and j["neuron"] == f]
        for bi in range(0, len(ujobs), a.batch):
            batch = ujobs[bi:bi + a.batch]
            img, pm, emb, attr = run_batch(batch)
            for i, j in enumerate(batch):
                p = j["prompt"]
                bimg, bemb, battr, bpix = base[p]
                row = dict(j)
                row.update({k: round(float(v[i]), 4) for k, v in pm.items()})
                row.update({f"d_{k}": round(float(v[i]) - bpix[k], 4) for k, v in pm.items()})
                row["mse"] = round(float(((img[i] - bimg) ** 2).mean()), 5)
                row["cos_base"] = round(float(emb[i] @ bemb), 4)
                row["d_attr"] = [round(float(x), 2) for x in (attr[i] - battr).tolist()]
                row["attr"] = [round(float(x), 2) for x in attr[i].tolist()]
                row["cos_concept"] = round(float(emb[i] @ text_emb), 4)
                row["base_cos_concept"] = round(float(bemb @ text_emb), 4)
                row["thumb"] = f"L{li}_{kind[0]}{f}_{p}_{j['dose']:+g}.jpg"
                out.write(json.dumps(row) + "\n")
                rows.append(row)
                t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(t).resize((a.thumb, a.thumb), Image.LANCZOS).save(f"{tdir}/{row['thumb']}", quality=85)
                if a.save_full:
                    os.makedirs(a.save_full, exist_ok=True)
                    Image.fromarray(t).save(f"{a.save_full}/{row['thumb']}", quality=92)
        print(f"{kind} f{f} done {time.time() - t0:.0f}s", flush=True)
    for p, (bimg, bemb, battr, bpix) in base.items():
        row = {"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0, "layer": li, "seed": prompts[p][1],
               "cos_concept": round(float(bemb @ text_emb), 4), "thumb": f"L{li}_base_{p}.jpg",
               "attr": [round(float(x), 2) for x in battr.tolist()]}
        row.update({k: round(v, 4) for k, v in bpix.items()})
        out.write(json.dumps(row) + "\n")
        t = (bimg.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(t).resize((a.thumb, a.thumb), Image.LANCZOS).save(f"{tdir}/{row['thumb']}", quality=85)
        if a.save_full:
            Image.fromarray(t).save(f"{a.save_full}/{row['thumb']}", quality=92)
    out.close()
    json.dump({"layer": li, "name": sd.names[li], "inner": 0, "dim": sd.dim[li], "prompts": list(prompts), "doses": targets,
               "dose_unit": "feature_max_multiple", "attributes": lens.attributes, "random_controls": len(rand),
               "n_neurons": len(feats), "concept": "sae", "text": a.text, "probe": True, "model": model,
               "feature_scale": scale, "layers": layers, "mask": a.mask, "groups": {str(k): [fam(x) for x in v] for k, v in group_of.items()}, "size": a.size, "steps": sd.steps,
               "from_step": a.from_step, "to_step": a.to_step}, open(f"runs/{run}.meta.json", "w"), indent=1)

    # report + sheets
    print(f"\nΔcos to '{a.text}' (mean over prompts), cos to baseline image; controls are random alive features")
    print("unit                target  Δcos_concept  cos_base  Δsat    top attribute shifts")
    for kind, f in units:
        for t in targets:
            rs = [r for r in rows if r["kind"] == kind and r["neuron"] == f and r["dose"] == t]
            dc = np.mean([r["cos_concept"] - r["base_cos_concept"] for r in rs])
            sh = np.mean([r["d_attr"] for r in rs], 0)
            top = ", ".join(f"{lens.attributes[i]}({sh[i]:+.1f})" for i in np.argsort(-sh)[:3])
            print(f"{kind:13s} f{f:6d}  {t:5.1f}   {dc:+.4f}     {np.mean([r['cos_base'] for r in rs]):.3f}   "
                  f"{np.mean([r['d_sat'] for r in rs]):+.3f}  {top}")
    os.makedirs("sheets", exist_ok=True)
    W = a.thumb
    for f in feats:
        sheet = Image.new("RGB", (W * (len(targets) + 1), W * len(prompts) + 20), "white")
        dr = ImageDraw.Draw(sheet)
        gname = ':'.join('+'.join(map(str, fam(x))) for x in group_of[f])
        dr.text((4, 3), f"{model} L{'+'.join(map(str, layers))} SAE feature(s) {gname} clamped "
                f"to k x max activation; columns: baseline, " + ", ".join(f"{t:g}x" for t in targets), fill="black")
        for r, p in enumerate(prompts):
            sheet.paste(Image.open(f"{tdir}/L{li}_base_{p}.jpg").resize((W, W)), (0, 20 + r * W))
            for c, t in enumerate(targets):
                sheet.paste(Image.open(f"{tdir}/L{li}_c{f}_{p}_{t:+g}.jpg").resize((W, W)), ((c + 1) * W, 20 + r * W))
        tag = (f"_s{a.from_step}" if a.from_step else "") + (f"_{a.size}" if a.size != 512 else "")
        path = f"sheets/{model}_sae_L{'+'.join(map(str, layers))}_f{gname.replace(':', '_')}_clamp_{'-'.join(f'{t:g}' for t in targets)}{('_' + a.mask.replace(':', '')) if a.mask else ''}{tag}.jpg"
        sheet.save(path, quality=88)
        print("wrote", path)


if __name__ == "__main__":
    main()
