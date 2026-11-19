"""Generation traces: what the picture looks like after every denoising step, and where one unit fires at each step.

    with Trace(sd, unit) as tr:
        img = sd.generate(prompts, seeds, pushes, ...)
    tr.frames()  -> list over steps of {"i", "sigma", "xt": [B,3,h,w], "x0": [B,3,h,w]}  (tiny-VAE decodes, [0,1])
    tr.acts      -> list over steps of [B, G, G] activation maps of the unit (None without a unit)

`xt` is the noisy latent the step started from (pure noise at step 0), `x0` the model's clean-image estimate at that
step: for the flow-matching FLUX scheduler x0 = x_t - sigma * v, for SDXL-Turbo's Euler-ancestral scheduler the
scheduler's own pred_original_sample. Both come from wrapping scheduler.step, so no pipeline code changes.

unit = {"kind": "neuron", "layer": li, "neuron": n}             activation = the GEGLU/GELU output of channel n
     | {"kind": "clamp", "layer": li, "neuron": f, "sae": {...}} activation = relu(((h-mu)/sigma - b_pre) @ W_enc[:, f] + b_enc[f])
"""
import math

import torch


class Trace:
    def __init__(self, sd, unit=None):
        self.sd, self.unit = sd, unit
        self.steps, self.acts = [], ([] if unit else None)
        self._orig = None
        self._ctx = None

    # ---- scheduler wrap ----------------------------------------------------------------------------------------
    def __enter__(self):
        sch = self.sd.pipe.scheduler
        self._orig = sch.step
        trace = self

        def step(model_output, timestep, sample, *a, **kw):
            out = trace._orig(model_output, timestep, sample, *a, **kw)
            x0 = getattr(out, "pred_original_sample", None)
            sigma = None
            if hasattr(sch, "sigmas") and getattr(sch, "step_index", None) is not None:
                idx = max(0, sch.step_index - 1)  # step() has already advanced the index
                sigma = float(sch.sigmas[idx])
            if x0 is None and sigma is not None:  # flow matching: x_t = (1 - s) x0 + s eps, v = eps - x0
                x0 = sample - sigma * model_output
            prev = out[0] if isinstance(out, tuple) else out.prev_sample
            trace.steps.append({"i": len(trace.steps), "sigma": sigma, "t": float(timestep) if torch.is_tensor(timestep) else timestep,
                                "xt": sample.detach(), "x0": (x0 if x0 is not None else prev).detach(), "prev": prev.detach()})
            return out
        sch.step = step
        if self.unit:
            self._ctx = self._capture()
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self.sd.pipe.scheduler.step = self._orig
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
        return False

    # ---- unit activation per step ------------------------------------------------------------------------------
    def _capture(self):
        u, li = self.unit, int(self.unit["layer"])
        if u["kind"] == "clamp":
            s = u["sae"]
            enc_w, enc_b = s["W_enc"][:, int(u["neuron"])], s["b_enc"][int(u["neuron"])]

            def resid_fn(l, hid):
                xs = (hid.float() - s["mu"]) / s["sigma"]
                act = torch.relu((xs - s["b_pre"]) @ enc_w + enc_b)  # [B, T]
                self.acts.append(self._grid(act))
            return self.sd.capturing([li], lambda l, out: None, resid_fn=resid_fn)

        def fn(l, out):
            self.acts.append(self._grid(out[:, :, int(u["neuron"])].float()))
        return self.sd.capturing([li], fn)

    @staticmethod
    def _grid(act):
        B, T = act.shape
        G = int(math.sqrt(T))
        if G * G != T:  # non-square token grid (rare): fall back to the largest square prefix
            act = act[:, : G * G]
        return act.reshape(B, G, G).cpu()

    # ---- decoding ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, lat):
        sd = self.sd
        if hasattr(sd.pipe, "_unpack_latents"):  # FLUX: packed [B, T, 64] -> [B, 16, h/8, w/8]
            lat = sd.pipe._unpack_latents(lat, sd.h, sd.w, sd.pipe.vae_scale_factor)
            x = sd.tiny.decode(lat.to(sd.tiny.dtype)).sample
        else:
            sf = getattr(sd.tiny.config, "scaling_factor", 1.0) or 1.0
            x = sd.tiny.decode((lat / sf).to(sd.tiny.dtype)).sample
        return (x.float() / 2 + 0.5).clamp(0, 1)

    def frames(self):
        out = []
        for s in self.steps:
            out.append({"i": s["i"], "sigma": s["sigma"], "t": s["t"], "xt": self.decode(s["xt"]), "x0": self.decode(s["x0"])})
        return out
