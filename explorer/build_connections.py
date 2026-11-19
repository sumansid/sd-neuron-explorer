"""Cross-layer connections between FLUX SAE features, computed on the CPU from the saved autoencoders and the
per-image corpus activations. Writes explorer/cache/conn_flux_sae.npz for the explorer's feature panel and model map.

    python explorer/build_connections.py            # ~1-2 min on a laptop

Two kinds of link, both Lyra-style "strongest links up and down" (top-K per feature, both directions):

* direct   feature f at layer A writes W_dec_A[f]*sigma_A into the shared 3072-wide residual stream; feature g at
           layer B reads it with W_enc_B[:, g]/sigma_B. The cosine between the write and read directions is the
           direct-path link, i.e. what reaches g if the blocks in between passed the stream through unchanged.
           Needs the weights, so only layers with a local checkpoint (sae/flux2: L21, L24, L26).
* coact    correlation across the 1,704-image corpus of the two features' per-image max activations (sae/flux/
           feat_L*_max.npy). Says which features fire on the same images, whatever the path. Available for all
           five SAE layers (L6, L12, L21, L24, L26).

Also stores, per layer and feature: a Golden Gate z-score (mean on gg-tagged images vs the rest, in units of the
rest's std), the corpus density, the six images it fires hardest on, and its top CLIP attributes (correlation of
activation with the 93 attribute scores over the corpus), so the explorer can colour the SAE sheets and describe
a feature without a GPU.
"""
import ast
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "explorer/cache/conn_flux_sae.pt"
DT = {"<f2": torch.float16, "<f4": torch.float32, "<f8": torch.float64, "<i2": torch.int16, "<i4": torch.int32, "<i8": torch.int64}


def load_npy(path):
    """Minimal .npy reader straight into torch (this machine's torch cannot bridge to its numpy)."""
    with open(path, "rb") as fh:
        assert fh.read(6) == b"\x93NUMPY"
        major = fh.read(2)[0]
        hlen = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
        hdr = ast.literal_eval(fh.read(hlen).decode("latin1"))
        data = fh.read()
    assert not hdr["fortran_order"], path
    t = torch.frombuffer(bytearray(data), dtype=DT[hdr["descr"]])
    return t.reshape(hdr["shape"]).clone()
K_DIRECT, K_COACT, CHUNK = 24, 16, 2048
WEIGHTS = {21: "sae/flux2/sae_L21_fp16.pt", 24: "sae/flux2/sae_L24_fp16.pt", 26: "sae/flux2/sae_L26_fp16.pt"}
ACT_LAYERS = [6, 12, 21, 24, 26]
DIRECT_PAIRS = [(21, 24), (24, 26), (21, 26)]
COACT_PAIRS = [(6, 12), (12, 21), (21, 24), (24, 26), (21, 26), (12, 24), (6, 21)]


def topk_both(mat_fn, n_a, n_b, k):
    """mat_fn(rows) -> [len(rows), n_b] block of the A x B link matrix. Returns top-k per row (A -> B, 'down') and
    per column (B <- A, 'up') without materialising the whole matrix."""
    down_v = torch.empty(n_a, k); down_i = torch.empty(n_a, k, dtype=torch.int32)
    up_v = torch.full((k, n_b), -1e9); up_i = torch.zeros(k, n_b, dtype=torch.int32)
    for s in range(0, n_a, CHUNK):
        rows = torch.arange(s, min(s + CHUNK, n_a))
        m = mat_fn(rows)  # [c, n_b]
        v, i = m.topk(k, dim=1)
        down_v[rows], down_i[rows] = v, i.int()
        cat_v = torch.cat([up_v, m], 0)
        cat_i = torch.cat([up_i, rows[:, None].expand(-1, n_b).int()], 0)
        v2, j = cat_v.topk(k, dim=0)
        up_v, up_i = v2, torch.gather(cat_i, 0, j)
    return down_i, down_v.half(), up_i.T.contiguous(), up_v.T.contiguous().half()


def main():
    t0 = time.time()
    out = {}
    meta = json.load(open("sae/flux/meta.json"))
    items = meta["items"]
    gg = torch.tensor([1.0 if "gg" in it["tags"] else 0.0 for it in items])
    attr = load_npy("sae/flux/attr.npy").float()  # [N, 93]
    attr_z = (attr - attr.mean(0)) / (attr.std(0) + 1e-6)

    # ---- per-layer feature descriptors from the corpus activations ---------------------------------------------
    acts = {}
    for l in ACT_LAYERS:
        p = f"sae/flux/feat_L{l}_max.npy"
        if not os.path.exists(p):
            print("missing", p); continue
        a = load_npy(p).float()  # [N, n]
        acts[l] = a
        alive = a.max(0).values > 0
        # standardised mean difference; the pooled std keeps features that are silent off the bridge from blowing up
        mu_gg = a[gg > 0].mean(0); mu_rest = a[gg == 0].mean(0); sd_all = a.std(0) + 1e-6
        out[f"gg_z_L{l}"] = torch.where(alive, (mu_gg - mu_rest) / sd_all, torch.zeros_like(mu_gg)).clamp(-50, 50).half()
        out[f"density_L{l}"] = (a > 0).float().mean(0).half()
        out[f"top_imgs_L{l}"] = a.topk(6, dim=0).indices.T.contiguous().short()  # [n, 6]
        az = (a - a.mean(0)) / (a.std(0) + 1e-6)
        corr = (az.T @ attr_z) / a.shape[0]  # [n, 93]
        v, i = corr.topk(4, dim=1)
        out[f"lens_idx_L{l}"], out[f"lens_val_L{l}"] = i.short(), v.half()
        print(f"L{l}: {int(alive.sum())} alive features, gg z max {float(out[f'gg_z_L{l}'].max()):.1f}  ({time.time() - t0:.0f}s)")

    # ---- co-activation links ---------------------------------------------------------------------------------------
    for la, lb in COACT_PAIRS:
        if la not in acts or lb not in acts:
            continue
        za = (acts[la] - acts[la].mean(0)) / (acts[la].std(0) + 1e-6)
        zb = (acts[lb] - acts[lb].mean(0)) / (acts[lb].std(0) + 1e-6)
        dead_a, dead_b = acts[la].max(0).values <= 0, acts[lb].max(0).values <= 0
        za[:, dead_a] = 0; zb[:, dead_b] = 0
        n = za.shape[0]
        di, dv, ui, uv = topk_both(lambda rows: (za[:, rows].T @ zb) / n, za.shape[1], zb.shape[1], K_COACT)
        out[f"coact_{la}_{lb}_down_idx"], out[f"coact_{la}_{lb}_down_val"] = di, dv
        out[f"coact_{la}_{lb}_up_idx"], out[f"coact_{la}_{lb}_up_val"] = ui, uv
        print(f"coact L{la}->L{lb}: median top link r={float(dv[:, 0].float().median()):.2f}  ({time.time() - t0:.0f}s)")

    # ---- direct-path links through the residual stream -------------------------------------------------------------
    saes = {}
    for l, p in WEIGHTS.items():
        if os.path.exists(p):
            saes[l] = torch.load(p, map_location="cpu")
    for la, lb in DIRECT_PAIRS:
        if la not in saes or lb not in saes:
            continue
        A, B = saes[la], saes[lb]
        write = A["W_dec"].float() * A["sigma"].float()[None, :]  # [n, D] raw-space write direction per feature
        read = B["W_enc"].float() / B["sigma"].float()[:, None]  # [D, n] raw-space read direction per feature
        write = write / (write.norm(dim=1, keepdim=True) + 1e-6)
        read = read / (read.norm(dim=0, keepdim=True) + 1e-6)
        if "density" in A:
            write[A["density"] <= 1e-5] = 0
        if "density" in B:
            read[:, B["density"] <= 1e-5] = 0
        di, dv, ui, uv = topk_both(lambda rows: write[rows] @ read, write.shape[0], read.shape[1], K_DIRECT)
        out[f"direct_{la}_{lb}_down_idx"], out[f"direct_{la}_{lb}_down_val"] = di, dv
        out[f"direct_{la}_{lb}_up_idx"], out[f"direct_{la}_{lb}_up_val"] = ui, uv
        print(f"direct L{la}->L{lb}: median top cos={float(dv[:, 0].float().median()):.3f}, max {float(dv[:, 0].float().max()):.3f}  ({time.time() - t0:.0f}s)")

    out["layers_act"] = sorted(acts)
    out["layers_direct"] = sorted(saes)
    out["attributes"] = meta["attributes"]
    out["k"] = {"direct": K_DIRECT, "coact": K_COACT}
    os.makedirs("explorer/cache", exist_ok=True)
    torch.save(out, OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
