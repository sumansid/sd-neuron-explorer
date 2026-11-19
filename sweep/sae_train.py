"""Train a TopK sparse autoencoder on one layer's collected residual activations.

Each channel is standardised first (mean 0, std 1 over tokens) so FLUX's massive-activation channels do not swamp
the reconstruction loss; the decoder directions are mapped back to raw residual space when exported. TopK keeps
exactly k latents per token; an AuxK term (reconstructing the residual with the top dead latents) revives dead ones.

    python sweep/sae_train.py --data sae/flux --layer 24 --features 24576 --k 32 --steps 3000

Writes sae/<model>/sae_L<li>.pt with W_enc [dim, F], b_enc [F], W_dec [F, dim] (unit rows), b_pre [dim], mu, sigma.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def load_layer(data, li, device):
    xs = []
    for f in sorted(glob.glob(f"{data}/L{li}/shard_*.npz")):
        xs.append(torch.from_numpy(np.load(f)["x"]))
    X = torch.cat(xs).to(device)  # fp16 [N, D]
    return X


class TopKSAE(torch.nn.Module):
    def __init__(self, d, f, k):
        super().__init__()
        self.k = k
        W = torch.randn(f, d)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W)
        self.W_enc = torch.nn.Parameter(W.t().clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(f))
        self.b_pre = torch.nn.Parameter(torch.zeros(d))

    def encode_pre(self, xs):
        return (xs - self.b_pre) @ self.W_enc + self.b_enc

    def topk(self, pre):
        v, i = pre.topk(self.k, dim=-1)
        v = F.relu(v)
        acts = torch.zeros_like(pre).scatter_(-1, i, v)
        return acts, i

    def forward(self, xs):
        pre = self.encode_pre(xs)
        acts, idx = self.topk(pre)
        return acts @ self.W_dec + self.b_pre, acts, idx, pre

    @torch.no_grad()
    def norm_dec(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sae/flux")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--features", type=int, default=24576)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-k", type=int, default=256)
    ap.add_argument("--aux-coef", type=float, default=1 / 32)
    ap.add_argument("--dead-steps", type=int, default=400)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda"
    out = a.out or f"{a.data}/sae_L{a.layer}.pt"

    X = load_layer(a.data, a.layer, dev)
    N, D = X.shape
    perm = torch.randperm(N, device=dev)
    n_hold = int(N * a.holdout)
    hold, train = perm[:n_hold], perm[n_hold:]
    # channel standardisation (chunked in fp32)
    s1 = torch.zeros(D, device=dev, dtype=torch.float64)
    s2 = torch.zeros(D, device=dev, dtype=torch.float64)
    for i in range(0, N, 262144):
        c = X[i:i + 262144].double()
        s1 += c.sum(0); s2 += (c * c).sum(0)
    mu = (s1 / N).float()
    sigma = ((s2 / N) - (s1 / N) ** 2).clamp_min(1e-8).sqrt().float()
    print(f"L{a.layer}: {N} tokens, dim {D}; sigma min/median/max {sigma.min():.3g}/{sigma.median():.3g}/{sigma.max():.3g}; "
          f"top-5 sigma channels {sigma.topk(5).indices.tolist()}")

    sae = TopKSAE(D, a.features, a.k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=6.25e-10)
    last_fire = torch.zeros(a.features, device=dev, dtype=torch.long)
    t0 = time.time()

    def std(x):
        return (x.float() - mu) / sigma

    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 200) * (0.5 * (1 + math.cos(math.pi * step / a.steps)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = train[torch.randint(0, train.numel(), (a.batch,), device=dev)]
        xs = std(X[idx])
        recon, acts, top_idx, pre = sae(xs)
        var = xs.var(0).sum()
        mse = (recon - xs).pow(2).sum(1).mean()
        loss = mse / var * D  # normalised so 1.0 == predicting the mean
        fired = torch.zeros(a.features, device=dev, dtype=torch.bool)
        fired[top_idx.unique()] = True
        last_fire[fired] = step
        dead = (step - last_fire) > a.dead_steps
        n_dead = int(dead.sum())
        if n_dead > 0 and step > a.dead_steps:
            resid = (xs - recon).detach()
            pre_dead = pre.masked_fill(~dead, float("-inf"))
            kk = min(a.aux_k, n_dead)
            v, i = pre_dead.topk(kk, dim=-1)
            aux_acts = torch.zeros_like(pre).scatter_(-1, i, F.relu(v))
            aux_recon = aux_acts @ sae.W_dec
            aux = (aux_recon - resid).pow(2).sum(1).mean() / resid.pow(2).sum(1).mean().clamp_min(1e-8)
            loss = loss + a.aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # remove the gradient component parallel to each decoder direction (keeps unit norm meaningful)
        with torch.no_grad():
            g = sae.W_dec.grad
            g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.norm_dec()
        if step % 100 == 0 or step == a.steps:
            with torch.no_grad():
                hx = std(X[hold[:16384]])
                hr, hacts, _, _ = sae(hx)
                fvu = float(((hr - hx).pow(2).sum(1).mean()) / hx.pow(2).sum(1).mean())
            print(f"step {step} loss {float(loss):.4f} fvu(holdout) {fvu:.4f} dead {n_dead}/{a.features} "
                  f"lr {lr:.2e} {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        # feature usage over a large sample, for density
        cnt = torch.zeros(a.features, device=dev)
        for i in range(0, min(N, 524288), 16384):
            xs = std(X[perm[i:i + 16384]])
            _, acts, _, _ = sae(xs)
            cnt += (acts > 0).float().sum(0)
        density = (cnt / min(N, 524288)).cpu()
    torch.save({"W_enc": sae.W_enc.data.cpu(), "b_enc": sae.b_enc.data.cpu(), "W_dec": sae.W_dec.data.cpu(),
                "b_pre": sae.b_pre.data.cpu(), "mu": mu.cpu(), "sigma": sigma.cpu(), "k": a.k, "layer": a.layer,
                "features": a.features, "fvu": fvu, "dead": n_dead, "density": density, "n_tokens": N}, out)
    print(f"saved {out}: fvu {fvu:.4f}, dead {n_dead}, alive-with-density>1e-5 {(density > 1e-5).sum().item()}")


if __name__ == "__main__":
    main()
