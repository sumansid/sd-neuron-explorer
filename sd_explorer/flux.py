"""A FLUX-architecture DiT (Shuttle 3 / FLUX.1 schnell) with the same hook surface as SDX in model.py.

Layers, in forward order: 19 double-stream blocks ("d0".."d18"), whose image-stream feed-forward is
GELU(tanh)(Linear 3072 -> 12288) then Linear 12288 -> 3072 (`ff.net[0]`, `ff.net[2]`); then 38 single-stream blocks
("s0".."s37"), whose MLP branch is `proj_mlp` (3072 -> 12288), `act_mlp`, and the MLP half of `proj_out`
(columns 3072: of its weight). A neuron is one of the 12,288 inner channels; its down column is what it writes into
the 3072-wide residual stream. Pushes are added to the block's image-stream output at every image token.

Memory on a 32 GB card: the transformer (bf16, 24 GB) and the T5 encoder (9.5 GB) do not fit together, so prompts
are encoded first (`prewarm`), T5 moves to the CPU, and unseen prompts later are encoded on the CPU (slow).
Images decode through the tiny FLUX decoder (taef1) for sweeps and the full VAE on request.
"""
import contextlib

import torch
from diffusers import AutoencoderTiny, FluxPipeline

import os
# FLUX.1 schnell itself is a gated repo; Shuttle 3 is an ungated Apache-2.0 fine-tune of it with the same
# architecture (19 double + 38 single blocks) and the same 4-step sampling. Override with SD_FLUX_MODEL.
MODEL = os.environ.get("SD_FLUX_MODEL", "shuttleai/shuttle-3-diffusion")
TINY_VAE = "madebyollin/taef1"


class FLX:
    def __init__(self, device="cuda", dtype=torch.bfloat16, tiny_vae=True, height=512, width=512, steps=4,
                 prewarm=(), max_seq=256):
        self.device, self.dtype, self.h, self.w, self.steps, self.max_seq = device, dtype, height, width, steps, max_seq
        self.pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)
        self.t_img = (height // 16) * (width // 16)
        self._embeds = {}
        # text encoders first, on the GPU, for every prompt we know we will need
        self.pipe.text_encoder.to(device)
        self.pipe.text_encoder_2.to(device)
        self.t5_device = device
        if prewarm:
            self.embed(list(prewarm))
        if os.environ.get("SD_FLUX_T5_GPU", "") not in ("1", "true", "yes"):
            # sweeps know their prompts up front: park T5 (9 GB) on the CPU after prewarming. Live use sets
            # SD_FLUX_T5_GPU=1 to keep it on the GPU, since encoding a new prompt on the CPU takes ~6 s.
            self.pipe.text_encoder_2.to("cpu")
            self.t5_device = "cpu"
        torch.cuda.empty_cache()
        self.tr = self.pipe.transformer.to(device)
        self.pipe.vae.to(device)
        self.tiny = AutoencoderTiny.from_pretrained(TINY_VAE, torch_dtype=dtype).to(device) if tiny_vae else None
        self.layers = [(f"transformer_blocks.{i}", b, "double") for i, b in enumerate(self.tr.transformer_blocks)] + \
                      [(f"single_transformer_blocks.{i}", b, "single") for i, b in enumerate(self.tr.single_transformer_blocks)]
        self.names = [f"d{i}" for i in range(len(self.tr.transformer_blocks))] + [f"s{i}" for i in range(len(self.tr.single_transformer_blocks))]
        self.inner = [self.tr.config.num_attention_heads * self.tr.config.attention_head_dim * 4] * len(self.layers)
        self.dim = [self.tr.config.num_attention_heads * self.tr.config.attention_head_dim] * len(self.layers)
        self._steer, self._capture, self._resid, self._edit = {}, {}, {}, {}
        self._install()

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _install(self):
        for li, (name, block, kind) in enumerate(self.layers):
            act_mod = block.ff.net[0] if kind == "double" else block.act_mlp

            def act_hook(mod, inp, out, li=li):
                for fn in self._capture.get(li, ()):
                    fn(out[:, -self.t_img:, :])  # image tokens only (single blocks see text + image)

            def blk_hook(mod, inp, out, li=li):
                enc, hid = out
                p = self._steer.get(li)
                if p is not None:
                    hid = hid + p[:, None, :].to(hid.dtype)
                for fn in self._edit.get(li, ()):
                    hid = fn(hid)  # token-wise edits (e.g. clamping an SAE feature)
                for fn in self._resid.get(li, ()):
                    fn(hid)
                return enc, hid
            act_mod.register_forward_hook(act_hook)
            block.register_forward_hook(blk_hook)

    def w_down(self, li):
        name, block, kind = self.layers[li]
        w = block.ff.net[2].weight if kind == "double" else block.proj_out.weight[:, self.dim[li]:]
        return w.detach()  # [dim, inner]

    def col_unit(self, li, n):
        c = self.w_down(li)[:, n].float()
        return c / (c.norm() + 1e-8)

    @contextlib.contextmanager
    def steering(self, pushes):
        self._steer = {k: v.to(self.device) for k, v in pushes.items()}
        try:
            yield
        finally:
            self._steer = {}

    @contextlib.contextmanager
    def editing(self, edits):
        """edits: {layer: fn(hid) -> hid}, applied to the block's image-stream output at every step."""
        self._edit = {k: [v] for k, v in edits.items()}
        try:
            yield
        finally:
            self._edit = {}

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
        todo = [p for p in prompts if p not in self._embeds]
        for i in range(0, len(todo), 8):
            chunk = todo[i:i + 8]
            t5 = self.pipe._get_t5_prompt_embeds(prompt=chunk, device=self.t5_device, max_sequence_length=self.max_seq)
            clip = self.pipe._get_clip_prompt_embeds(prompt=chunk, device=self.device)
            for p, a, b in zip(chunk, t5, clip):
                self._embeds[p] = (a.to(self.device, self.dtype), b.to(self.device, self.dtype))
        pe = torch.stack([self._embeds[p][0] for p in prompts])
        pooled = torch.stack([self._embeds[p][1] for p in prompts])
        return pe, pooled

    @torch.no_grad()
    def generate(self, prompts, seeds, pushes=None, full_vae=False, steps=None):
        pe, pooled = self.embed(prompts)
        gens = [torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds]
        with self.steering(pushes or {}):
            lat = self.pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled, num_inference_steps=steps or self.steps,
                            guidance_scale=0.0, height=self.h, width=self.w, generator=gens, output_type="latent",
                            max_sequence_length=self.max_seq).images
        lat = self.pipe._unpack_latents(lat, self.h, self.w, self.pipe.vae_scale_factor)
        if full_vae or self.tiny is None:
            vae = self.pipe.vae
            x = vae.decode(lat / vae.config.scaling_factor + vae.config.shift_factor).sample
        else:
            x = self.tiny.decode(lat).sample
        return (x.float() / 2 + 0.5).clamp(0, 1)

    def push_for(self, li, kind, n, dose, unit, rng=None):
        if kind == "neuron":
            d = self.col_unit(li, n)
        else:
            g = torch.Generator(device="cpu").manual_seed(10_000_003 * (li + 1) + int(n))
            d = torch.randn(self.dim[li], generator=g)
            d = (d / d.norm()).to(self.device)
        return d * (dose * unit)
