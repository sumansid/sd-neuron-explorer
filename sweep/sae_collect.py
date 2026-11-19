"""Collect residual-stream activations for sparse-autoencoder training: the image tokens leaving chosen blocks, at
every denoising step, over the SAE prompt corpus (sweep/prompts.py: broad subjects plus Golden Gate / other-bridge /
other-landmark slices). A random subset of token positions is kept per image and step; the positions are shared
across layers so the same tokens can be compared layer to layer.

    python sweep/sae_collect.py --model flux --layers 6,12,21,24,26 --tokens 192 --seeds 2 --out sae/flux

Writes sae/<model>/L<li>/shard_<k>.npz (x: [n, dim] fp16, img: int32, step: int8, pos: int16), sae/<model>/meta.json,
sae/<model>/imgs/<id>.jpg (thumbnails), sae/<model>/emb.npy + attr.npy (CLIP embedding and attribute scores per image).
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
from sd_explorer.metrics import Lens
from sweep.prompts import sae_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux", choices=["sdxl", "flux", "flux3"])
    ap.add_argument("--layers", default="6,12,21,24,26")
    ap.add_argument("--tokens", type=int, default=192, help="token positions kept per image and step")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N prompts")
    ap.add_argument("--flush", type=int, default=24, help="batches per shard")
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"sae/{a.model}"
    layers = [int(x) for x in a.layers.split(",")]

    corpus = sae_corpus()
    if a.limit:
        corpus = corpus[: a.limit]
    jobs = []
    for pi, (p, tags) in enumerate(corpus):
        for s in range(a.seeds):
            jobs.append({"id": len(jobs), "prompt": p, "tags": tags, "seed": 50_000 + pi * 10 + s})
    print(f"{len(corpus)} prompts x {a.seeds} seeds = {len(jobs)} images; layers {layers}; {a.tokens} tokens/step")

    sd = load_model(a.model, prewarm=[p for p, _ in corpus])
    lens = Lens()
    T, D = sd.t_img, sd.dim[layers[0]]
    steps = sd.steps
    for li in layers:
        os.makedirs(f"{a.out}/L{li}", exist_ok=True)
    os.makedirs(f"{a.out}/imgs", exist_ok=True)

    state = {"counter": {}, "pos": None, "ids": None}
    bufs = {li: {"x": [], "img": [], "step": [], "pos": []} for li in layers}

    def res(li, out):
        st = state["counter"].get(li, 0)
        state["counter"][li] = st + 1
        if st >= steps:
            return
        pos = state["pos"][:, st]  # [B, K]
        B = out.shape[0]
        g = out[torch.arange(B, device=out.device)[:, None], pos]  # [B, K, D]
        bufs[li]["x"].append(g.to(torch.float16).cpu())
        bufs[li]["img"].append(np.repeat(state["ids"], pos.shape[1]).astype(np.int32))
        bufs[li]["step"].append(np.full(B * pos.shape[1], st, np.int8))
        bufs[li]["pos"].append(pos.cpu().numpy().astype(np.int16).reshape(-1))

    def flush(k):
        for li in layers:
            b = bufs[li]
            if not b["x"]:
                continue
            np.savez(f"{a.out}/L{li}/shard_{k:03d}.npz", x=torch.cat(b["x"]).numpy().reshape(-1, D),
                     img=np.concatenate(b["img"]), step=np.concatenate(b["step"]), pos=np.concatenate(b["pos"]))
            for key in b:
                b[key].clear()

    embs, attrs = [], []
    t0 = time.time()
    shard = 0
    with sd.capturing(layers, lambda li, out: None, res):
        for bi in range(0, len(jobs), a.batch):
            batch = jobs[bi:bi + a.batch]
            B = len(batch)
            rng = np.random.default_rng(batch[0]["id"])
            pos = np.stack([np.stack([rng.choice(T, a.tokens, replace=False) for _ in range(steps)]) for _ in range(B)])
            state["pos"] = torch.tensor(pos, device=sd.device)
            state["ids"] = np.array([j["id"] for j in batch])
            state["counter"] = {}
            img = sd.generate([j["prompt"] for j in batch], [j["seed"] for j in batch])
            emb = lens.embed_images(img)
            embs.append(emb.cpu().numpy().astype(np.float16))
            attrs.append(lens.attribute_scores(emb).cpu().numpy().astype(np.float16))
            for i, j in enumerate(batch):
                t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(t).resize((a.thumb, a.thumb), Image.LANCZOS).save(f"{a.out}/imgs/{j['id']}.jpg", quality=88)
            if (bi // a.batch + 1) % a.flush == 0:
                flush(shard)
                shard += 1
            if (bi // a.batch) % 10 == 0:
                done = bi + B
                rate = done / (time.time() - t0)
                print(f"{done}/{len(jobs)} {rate:.2f} img/s eta {(len(jobs) - done) / max(rate, 1e-6) / 60:.0f} min", flush=True)
    flush(shard)
    np.save(f"{a.out}/emb.npy", np.concatenate(embs))
    np.save(f"{a.out}/attr.npy", np.concatenate(attrs))
    json.dump({"model": a.model, "layers": layers, "names": [sd.names[li] for li in layers], "tokens": a.tokens,
               "steps": steps, "t_img": T, "dim": D, "grid": int(round(T ** 0.5)), "seeds": a.seeds,
               "attributes": lens.attributes, "items": jobs}, open(f"{a.out}/meta.json", "w"))
    print(f"done: {len(jobs)} images, {len(jobs) * steps * a.tokens} tokens per layer, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
