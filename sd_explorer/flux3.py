"""FLUX 3 Image Klein (black-forest-labs/flux-3-image-klein, 5.6B, distilled: 16 steps, no guidance) with the same
hook surface as FLX in flux.py and SDX in model.py.

Layers, in forward order (image path only): 5 image "mode" blocks ("m0".."m4": attention and feed-forward in
parallel on image tokens alone; the text has its own 5 blocks we leave untouched), then 28 single-stream blocks
("s0".."s27", text tokens first, then image tokens). Every feed-forward is SwiGLU: `ff.linear_in` (3072 -> 2x9216),
`ff.act_fn` (-> 9216), `ff.linear_out` (9216 -> 3072). A neuron is one of the 9,216 inner channels; its column of
`linear_out` is what it writes into the 3072-wide residual stream. Pushes and edits are applied to the block's
image-stream output at every image token.

The model ships as a diffusers ModularPipeline whose first block is a 9B prompt upsampler; we drop that block so
our short prompts go to the Qwen3-VL text encoder as written. The repo is gated: accept the licence on Hugging
Face and put HF_TOKEN in the pod's environment before loading. Override the repo with SD_FLUX3_MODEL.
"""
import contextlib
import os

import numpy as np
import torch

MODEL = os.environ.get("SD_FLUX3_MODEL", "black-forest-labs/flux-3-image-klein")


class FL3:
    def __init__(self, device="cuda", dtype=torch.bfloat16, tiny_vae=False, height=512, width=512, steps=16,
                 prewarm=(), max_seq=None):
        from diffusers.modular_pipelines.flux3 import Flux3DistilledAutoBlocks
        self.device, self.dtype, self.h, self.w, self.steps = device, dtype, height, width, steps
        # build the block graph first and drop the prompt upsampler (a separate 9B model) before the pipeline
        # exists: popping it from a loaded pipeline's block list does not take effect
        blocks = Flux3DistilledAutoBlocks()
        for name in list(blocks.sub_blocks.keys()):
            if "upsampl" in name.lower():
                blocks.sub_blocks.pop(name)
                print(f"dropped pipeline block {name}")
        self.pipe = blocks.init_pipeline(MODEL)
        self.pipe.load_components(dtype={"vae": torch.float32, "default": dtype})
        self.pipe.to(device)
        self.tr = self.pipe.transformer
        self.t_img = (height // 16) * (width // 16)
        cfg = self.tr.config
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        inner = int(dim * float(getattr(cfg, "mlp_ratio", 3.0)))
        mode = list(self.tr.mode_blocks["image"])
        single = list(self.tr.single_transformer_blocks)
        self.layers = [(f"mode_blocks.image.{i}", b, "mode") for i, b in enumerate(mode)] + \
                      [(f"single_transformer_blocks.{i}", b, "single") for i, b in enumerate(single)]
        self.names = [f"m{i}" for i in range(len(mode))] + [f"s{i}" for i in range(len(single))]
        self.dim = [dim] * len(self.layers)
        self.inner = [inner] * len(self.layers)
        self._steer, self._capture, self._resid, self._edit = {}, {}, {}, {}
        self._install()

    def _drop_upsampler(self):
        """Remove the prompt-upsampler sub-block (a separate 9B model) so prompts are used as written."""
        blocks = getattr(self.pipe, "blocks", None)
        subs = getattr(blocks, "sub_blocks", None)
        if subs is None:
            return
        for name in list(subs.keys()):
            if "upsampl" in name.lower():
                subs.pop(name)
                print(f"dropped pipeline block {name}")

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _install(self):
        for li, (name, block, kind) in enumerate(self.layers):
            def act_hook(mod, inp, out, li=li):
                for fn in self._capture.get(li, ()):
                    fn(out[:, -self.t_img:, :])  # image tokens are last (single blocks see text + image)

            def blk_hook(mod, inp, out, li=li, kind=kind):
                if kind == "single":
                    enc, hid = out
                else:
                    enc, hid = None, out
                p = self._steer.get(li)
                if p is not None:
                    hid = hid + p[:, None, :].to(hid.dtype)
                for fn in self._edit.get(li, ()):
                    hid = fn(hid)
                for fn in self._resid.get(li, ()):
                    fn(hid)
                return (enc, hid) if kind == "single" else hid
            block.ff.act_fn.register_forward_hook(act_hook)
            block.register_forward_hook(blk_hook)

    def w_down(self, li):
        return self.layers[li][1].ff.linear_out.weight.detach()  # [dim, inner]

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
    def embed(self, prompts):
        return None  # the modular pipeline encodes prompts itself; kept for interface parity

    @torch.no_grad()
    def generate(self, prompts, seeds, pushes=None, full_vae=True, steps=None):
        """The modular pipeline encodes one prompt per call, so a mixed batch is run as one call per distinct
        prompt (its seeds as a generator list, num_images_per_prompt = group size) and reassembled in order.
        Pushes are indexed by the original batch position, so they are sliced per group."""
        prompts = list(prompts)
        groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(p, []).append(i)
        out = [None] * len(prompts)
        for p, idx in groups.items():
            gens = [torch.Generator(device="cpu").manual_seed(int(seeds[i])) for i in idx]
            sub = {li: v[idx] for li, v in (pushes or {}).items()}
            with self.steering(sub):
                res = self.pipe(prompt=p, generator=gens, num_images_per_prompt=len(idx), height=self.h,
                                width=self.w, num_inference_steps=steps or self.steps, output="images")
            imgs = res if isinstance(res, (list, tuple)) else [res]
            for i, im in zip(idx, imgs):
                out[i] = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(self.device)

    def push_for(self, li, kind, n, dose, unit, rng=None):
        if kind == "neuron":
            d = self.col_unit(li, n)
        else:
            g = torch.Generator(device="cpu").manual_seed(10_000_003 * (li + 1) + int(n))
            d = torch.randn(self.dim[li], generator=g)
            d = (d / d.norm()).to(self.device)
        return d * (dose * unit)
