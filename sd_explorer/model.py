"""SDXL-Turbo with hooks on every feed-forward (FF) layer of the UNet.

A "neuron" is one channel of the FF's inner activation: in diffusers' FeedForward the first module is a GEGLU
(Linear dim -> 2*inner, then hidden * gelu(gate)), the last is Linear inner -> dim. Neuron n's activation is
GEGLU output channel n, its "down column" is net[2].weight[:, n] (what it writes into the residual stream).

Steering adds a vector to the FF output (the mlp_out hook of the reference explorer): pushing neuron n by a dose
means adding dose * unit * W_down[:, n] / ||W_down[:, n]|| at every spatial token, where unit is the median
residual norm leaving that block ("rel" doses, the same push size for every neuron) or the neuron's own abs_max
times its column norm ("absmax" doses). Random-direction controls add a random unit vector of the same size.

SDXL-Turbo is a one-step model, so each image is one UNet forward pass and a neuron fires once per image.
"""
import contextlib
import math
import re

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, AutoencoderTiny

MODEL = "stabilityai/sdxl-turbo"
TINY_VAE = "madebyollin/taesdxl"


def ff_layers(unet):
    """All FeedForward modules of the UNet in forward order (down blocks, mid block, up blocks):
    (name, block_module, ff_module). named_modules() lists mid_block after up_blocks, so it is reordered here."""
    groups = {"down": [], "mid": [], "up": []}
    for name, m in unet.named_modules():
        if m.__class__.__name__ == "BasicTransformerBlock":
            groups[name.split("_")[0]].append((name, m, m.ff))
    return groups["down"] + groups["mid"] + groups["up"]


def short_name(name):
    """down_blocks.2.attentions.1.transformer_blocks.7 -> d2.a1.t7 ; mid_block.attentions.0.transformer_blocks.3 -> m.a0.t3"""
    s = name.replace("down_blocks.", "d").replace("up_blocks.", "u").replace("mid_block", "m")
    s = s.replace(".attentions.", ".a").replace(".transformer_blocks.", ".t")
    return s


class SDX:
    def __init__(self, device="cuda", dtype=torch.float16, tiny_vae=True, height=512, width=512):
        self.device, self.dtype, self.h, self.w = device, dtype, height, width
        self.pipe = AutoPipelineForText2Image.from_pretrained(MODEL, torch_dtype=dtype, variant="fp16").to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.full_vae = self.pipe.vae
        self.tiny = AutoencoderTiny.from_pretrained(TINY_VAE, torch_dtype=dtype).to(device) if tiny_vae else None
        self.unet = self.pipe.unet
        self.layers = ff_layers(self.unet)
        self.names = [short_name(n) for n, _, _ in self.layers]
        self.inner = [ff.net[2].in_features for _, _, ff in self.layers]
        self.dim = [ff.net[2].out_features for _, _, ff in self.layers]
        self._embeds = {}
        self._steer = {}     # layer index -> [B, dim] push tensor (added at FF output)
        self._capture = {}   # layer index -> list of callables fed the GEGLU output [B, T, inner]
        self._resid = {}     # layer index -> list of callables fed the block output [B, T, dim]
        self._handles = []
        self._install()

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _install(self):
        for li, (name, block, ff) in enumerate(self.layers):
            def ff_hook(mod, inp, out, li=li):
                p = self._steer.get(li)
                if p is not None:
                    out = out + p[:, None, :].to(out.dtype)
                return out
            def act_hook(mod, inp, out, li=li):
                for fn in self._capture.get(li, ()):
                    fn(out)
            def blk_hook(mod, inp, out, li=li):
                for fn in self._resid.get(li, ()):
                    fn(out)
            self._handles.append(ff.register_forward_hook(ff_hook))
            self._handles.append(ff.net[0].register_forward_hook(act_hook))
            self._handles.append(block.register_forward_hook(blk_hook))

    def w_down(self, li):
        return self.layers[li][2].net[2].weight.detach()  # [dim, inner]

    def col_unit(self, li, n):
        c = self.w_down(li)[:, n].float()
        return c / (c.norm() + 1e-8)

    @contextlib.contextmanager
    def steering(self, pushes):
        """pushes: {layer_index: tensor [B, dim]} added to that FF's output for the whole batch."""
        self._steer = {k: v.to(self.device) for k, v in pushes.items()}
        try:
            yield
        finally:
            self._steer = {}

    @contextlib.contextmanager
    def capturing(self, layers, fn, resid_fn=None):
        for li in layers:
            self._capture.setdefault(li, []).append(lambda out, li=li: fn(li, out))
            if resid_fn is not None:
                self._resid.setdefault(li, []).append(lambda out, li=li: resid_fn(li, out))
        try:
            yield
        finally:
            self._capture, self._resid = {}, {}

    # ---- generation --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, prompts):
        """Cached SDXL prompt embeddings: returns (prompt_embeds [B,77,2048], pooled [B,1280])."""
        todo = [p for p in prompts if p not in self._embeds]
        for i in range(0, len(todo), 16):
            chunk = todo[i:i + 16]
            pe, _, pooled, _ = self.pipe.encode_prompt(prompt=chunk, device=self.device, num_images_per_prompt=1,
                                                       do_classifier_free_guidance=False)
            for p, a, b in zip(chunk, pe, pooled):
                self._embeds[p] = (a, b)
        pe = torch.stack([self._embeds[p][0] for p in prompts])
        pooled = torch.stack([self._embeds[p][1] for p in prompts])
        return pe, pooled

    def latents(self, seeds):
        shape = (1, self.unet.config.in_channels, self.h // 8, self.w // 8)
        outs = []
        for s in seeds:
            g = torch.Generator(device="cpu").manual_seed(int(s))
            outs.append(torch.randn(shape, generator=g, dtype=self.dtype))
        return torch.cat(outs).to(self.device)

    @torch.no_grad()
    def generate(self, prompts, seeds, pushes=None, full_vae=False, steps=1):
        """Images in [0,1] as a float tensor [B,3,H,W]. One UNet pass per image at steps=1."""
        pe, pooled = self.embed(prompts)
        lat = self.latents(seeds)
        self.pipe.vae = self.full_vae if (full_vae or self.tiny is None) else self.tiny
        with self.steering(pushes or {}):
            out = self.pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled, latents=lat, num_inference_steps=steps,
                            guidance_scale=0.0, height=self.h, width=self.w, output_type="pt")
        return out.images.float().clamp(0, 1)

    def push_for(self, li, kind, n, dose, unit, rng=None):
        """One push vector [dim] for a unit: a neuron's down column direction or a random direction, scaled."""
        if kind == "neuron":
            d = self.col_unit(li, n)
        else:
            g = torch.Generator(device="cpu").manual_seed(10_000_003 * (li + 1) + int(n))
            d = torch.randn(self.dim[li], generator=g)
            d = (d / d.norm()).to(self.device)
        return d * (dose * unit)
