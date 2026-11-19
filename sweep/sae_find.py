"""Find what the SAE features of one layer respond to, Anthropic-dashboard style: score every feature on every corpus
image, rank features against a prompt tag (which images were prompted with the Golden Gate Bridge), name the top
features with the CLIP lens, draw where they fire, and export their decoder directions for steering with probe.py.

    python sweep/sae_find.py --data sae/flux --layer 24 --phase score                # regenerate corpus, feature maxima
    python sweep/sae_find.py --data sae/flux --layer 24 --phase rank --tag gg --top 24
    python sweep/sae_find.py --data sae/flux --layer 24 --phase maps --features 123,456 --export saegg

score writes sae/<model>/feat_L<li>_max.npy and _mean.npy ([n_img, F] fp16); rank writes sae/<model>/find_L<li>_<tag>.json;
maps writes sheets/<model>_sae_L<li>_f<F>.jpg; --export writes stats-flux/concept_<name>.npz with v_L<li> for probe.py.
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


class SAE:
    def __init__(self, path, device="cuda"):
        d = torch.load(path, map_location=device)
        self.W_enc, self.b_enc, self.W_dec, self.b_pre = (d[k].float() for k in ("W_enc", "b_enc", "W_dec", "b_pre"))
        self.mu, self.sigma, self.k, self.layer = d["mu"], d["sigma"], d["k"], d["layer"]
        self.F = self.W_dec.shape[0]
        self.info = {k: d[k] for k in ("fvu", "dead", "n_tokens", "features")}

    @torch.no_grad()
    def acts(self, x):  # x [..., D] raw residual -> [..., F] sparse TopK activations
        xs = (x.float() - self.mu) / self.sigma
        pre = (xs - self.b_pre) @ self.W_enc + self.b_enc
        v, i = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))

    def raw_dir(self, f):  # decoder direction in raw residual space, unit norm
        v = self.W_dec[f] * self.sigma
        return v / v.norm()


def auroc(scores, pos):
    order = scores.argsort()
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sae/flux")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--layers", default="", help="score: several layers at once (each needs its sae_L<li>.pt)")
    ap.add_argument("--sae", default=None)
    ap.add_argument("--phase", default="rank", choices=["score", "rank", "maps"])
    ap.add_argument("--tag", default="gg")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--min-density", type=float, default=0.003, help="min fraction of images where the feature fires")
    ap.add_argument("--features", default="", help="maps: feature ids, comma separated")
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--export", default="", help="maps: write stats-flux/concept_<name>.npz with the first feature's direction")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seeds", default="all", help="score: 'all' or 'first' seed per prompt")
    a = ap.parse_args()
    meta = json.load(open(f"{a.data}/meta.json"))
    li, model = a.layer, meta["model"]
    sae_path = a.sae or f"{a.data}/sae_L{li}.pt"
    items = meta["items"]
    T, G, steps = meta["t_img"], meta["grid"], meta["steps"]

    if a.phase == "score":
        from sd_explorer import load_model
        score_layers = [int(x) for x in a.layers.split(",")] if a.layers else [li]
        saes = {l: SAE(a.sae if (a.sae and len(score_layers) == 1) else f"{a.data}/sae_L{l}.pt") for l in score_layers}
        sel = items if a.seeds == "all" else [it for it in items if it["seed"] % 10 == 0]
        sd = load_model(model, prewarm=sorted({it["prompt"] for it in sel}))
        fmax = {l: np.zeros((len(items), saes[l].F), np.float16) for l in score_layers}
        fmean = {l: np.zeros((len(items), saes[l].F), np.float16) for l in score_layers}
        acc = {}

        def res(l, out):
            A = saes[l].acts(out)  # [B, T, F]
            d = acc.setdefault(l, {})
            d["max"] = torch.maximum(d["max"], A.amax(1)) if "max" in d else A.amax(1)
            d["sum"] = d.get("sum", 0) + A.mean(1)
            d["n"] = d.get("n", 0) + 1

        t0 = time.time()
        with sd.capturing(score_layers, lambda l, o: None, res):
            for bi in range(0, len(sel), a.batch):
                batch = sel[bi:bi + a.batch]
                acc.clear()
                sd.generate([j["prompt"] for j in batch], [j["seed"] for j in batch])
                ids = [j["id"] for j in batch]
                for l in score_layers:
                    fmax[l][ids] = acc[l]["max"].cpu().numpy().astype(np.float16)
                    fmean[l][ids] = (acc[l]["sum"] / acc[l]["n"]).cpu().numpy().astype(np.float16)
                if (bi // a.batch) % 20 == 0:
                    print(f"{bi + len(batch)}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
        for l in score_layers:
            np.save(f"{a.data}/feat_L{l}_max.npy", fmax[l])
            np.save(f"{a.data}/feat_L{l}_mean.npy", fmean[l])
        print("saved feature scores for layers", score_layers)
        return

    fmax = np.load(f"{a.data}/feat_L{li}_max.npy").astype(np.float32)
    attr = np.load(f"{a.data}/attr.npy").astype(np.float32)
    attributes = meta["attributes"]
    tags = lambda key: np.array([key in it["tags"] for it in items])
    scored = fmax.any(1)  # images the score phase covered

    if a.phase == "rank":
        d = torch.load(sae_path, map_location="cpu")
        W_dec, sigma = d["W_dec"].numpy(), d["sigma"].numpy()
        pos = tags(a.tag) & scored
        neg = ~tags(a.tag) & scored
        X = fmax[scored]
        dens = (X > 0).mean(0)
        m_pos, m_neg = fmax[pos].mean(0), fmax[neg].mean(0)
        z = (m_pos - m_neg) / (fmax[neg].std(0) + 1e-6)
        ctrl = {k: fmax[tags(k) & scored].mean(0) for k in ("bridge", "landmark", "bw", "painting")}
        m_main = fmax[tags("gg_main") & scored].mean(0) if tags("gg_main").any() else None
        cvec = None
        cpath = f"stats-{model}/concept_goldengate.npz" if model != "sdxl" else "stats/concept_goldengate.npz"
        if os.path.exists(cpath) and f"v_L{li}" in np.load(cpath):
            cvec = np.load(cpath)[f"v_L{li}"]; cvec = cvec / np.linalg.norm(cvec)
        cand = np.where(dens >= a.min_density)[0]
        pos_s = pos[scored]
        au = np.array([auroc(X[:, f], pos_s) for f in cand])
        order = cand[np.argsort(-au)][: a.top]
        recs = []
        base_attr = attr[scored].mean(0)
        print(f"L{li} ({meta['names'][meta['layers'].index(li)] if li in meta['layers'] else li}): {len(cand)} features with "
              f"density >= {a.min_density}; tag '{a.tag}' {pos.sum()} vs {neg.sum()} images")
        print(" rank  feat   auroc    z   mean_tag mean_rest  bridge landmark  dens  cos_cv  prec@50  lens")
        for r, f in enumerate(order):
            top_imgs = np.argsort(-fmax[:, f])[:50]
            prec = float(pos[top_imgs].mean())
            sh = attr[top_imgs[:16]].mean(0) - base_attr
            lens = [(attributes[i], round(float(sh[i]), 2)) for i in np.argsort(-sh)[:4]]
            raw = W_dec[f] * sigma; raw /= np.linalg.norm(raw)
            cos_cv = float(raw @ cvec) if cvec is not None else None
            rec = {"feature": int(f), "auroc": float(au[cand == f][0]), "z": float(z[f]), "mean_tag": float(m_pos[f]),
                   "mean_rest": float(m_neg[f]), "mean_bridge": float(ctrl["bridge"][f]), "mean_landmark": float(ctrl["landmark"][f]),
                   "mean_gg_main": None if m_main is None else float(m_main[f]), "density": float(dens[f]), "prec50": prec,
                   "cos_concept": cos_cv, "lens": lens, "top_imgs": [int(i) for i in top_imgs[:24]],
                   "top_prompts": [items[int(i)]["prompt"] for i in top_imgs[:6]]}
            recs.append(rec)
            print(f"{r + 1:5d} {f:6d}  {rec['auroc']:.3f} {rec['z']:5.1f}  {rec['mean_tag']:7.2f} {rec['mean_rest']:8.2f}  "
                  f"{rec['mean_bridge']:6.2f} {rec['mean_landmark']:7.2f}  {rec['density']:.3f}  "
                  f"{(cos_cv if cos_cv is not None else 0):+.2f}   {prec:.2f}   "
                  + ", ".join(f"{n}({v:+.1f})" for n, v in lens))
        json.dump({"layer": li, "tag": a.tag, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "sae": sae_path,
                   "sae_info": {k: (float(v) if hasattr(v, "__float__") else v) for k, v in
                                {"fvu": d["fvu"], "dead": d["dead"], "features": d["features"], "k": d["k"]}.items()},
                   "top": recs}, open(f"{a.data}/find_L{li}_{a.tag}.json", "w"), indent=1)
        return

    # maps: regenerate the top images of chosen features, draw per-token activation heatmaps
    from sd_explorer import load_model
    sae = SAE(sae_path)
    feats = [int(x) for x in a.features.split(",") if x]
    need = {}
    for f in feats:
        for i in np.argsort(-fmax[:, f])[: a.n_imgs]:
            need.setdefault(int(i), set()).add(f)
    ids = sorted(need)
    sd = load_model(model, prewarm=sorted({items[i]["prompt"] for i in ids}))
    fidx = torch.tensor(feats, device=sd.device)
    maps = {}  # (img, f) -> [steps, T]
    cur = {}

    def res(_li, out):
        A = sae.acts(out)[:, :, fidx]  # [B, T, nf]
        cur.setdefault("steps", []).append(A.cpu())

    imgs = {}
    with sd.capturing([li], lambda l, o: None, res):
        for bi in range(0, len(ids), a.batch):
            batch = ids[bi:bi + a.batch]
            cur.clear()
            img = sd.generate([items[i]["prompt"] for i in batch], [items[i]["seed"] for i in batch])
            S = torch.stack(cur["steps"], 1)  # [B, steps, T, nf]
            for b, i in enumerate(batch):
                imgs[i] = (img[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for fi, f in enumerate(feats):
                    maps[(i, f)] = S[b, :, :, fi].numpy()
    os.makedirs("sheets", exist_ok=True)
    W = 192
    for f in feats:
        top = [int(i) for i in np.argsort(-fmax[:, f])[: a.n_imgs]]
        vmax = max(maps[(i, f)].max() for i in top) + 1e-6
        sheet = Image.new("RGB", (W * len(top), W * 2 + 36), "white")
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 2), f"L{li} feature {f}: top {len(top)} images (row 1) and where it fires, max over {steps} steps (row 2); "
                f"density {float((fmax[:, f] > 0).mean()):.3f}", fill="black")
        for c, i in enumerate(top):
            im = Image.fromarray(imgs[i]).resize((W, W), Image.LANCZOS)
            sheet.paste(im, (c * W, 36))
            m = maps[(i, f)].max(0).reshape(G, G) / vmax
            heat = Image.fromarray((m * 255).astype(np.uint8)).resize((W, W), Image.BILINEAR)
            red = Image.new("RGB", (W, W), (255, 40, 0))
            over = Image.composite(red, im, heat.point(lambda v: int(v * 0.85)))
            sheet.paste(over, (c * W, 36 + W))
            dr.text((c * W + 3, 20), f"{fmax[i, f]:.1f} {items[i]['prompt'][:30]}", fill="black")
        path = f"sheets/{model}_sae_L{li}_f{f}.jpg"
        sheet.save(path, quality=88)
        print("wrote", path, "| top prompts:", [items[i]["prompt"] for i in top[:4]])
    if a.export:
        sdir = f"stats-{model}" if model != "sdxl" else "stats"
        path = f"{sdir}/concept_{a.export}.npz"
        cur_npz = dict(np.load(path)) if os.path.exists(path) else {}
        cur_npz[f"v_L{li}"] = sae.raw_dir(feats[0]).cpu().numpy()
        np.savez(path, **cur_npz)
        print(f"exported feature {feats[0]} direction as {path}:v_L{li} (keys now {sorted(cur_npz)})")


if __name__ == "__main__":
    main()
