"""Push one SDXL-Turbo neuron while redrawing a photo (image-to-image).
    python sweep/img2img_steer.py --image inputs/holmes.jpg --layer 18 --neuron 297 --doses=0,-1,-2,-3 \
        --strength 0.6 --steps 4 --prompt "a studio portrait photo of a woman" --out thumbs/i2i-holmes
The photo is noised to `strength` of the way and denoised in `steps` (strength*steps must be >= 1).
The neuron push is added at the layer's output on every step, in the same "rel" units as the sweeps.
"""
import argparse, os
import numpy as np, torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from sd_explorer.model import SDX

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True); ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--neuron", type=int, required=True); ap.add_argument("--doses", default="0,-1,-2")
ap.add_argument("--strength", default="0.6"); ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--prompt", default="a photo"); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--stats", default="stats/sdxl-turbo"); ap.add_argument("--out", required=True)
a = ap.parse_args()

sd = SDX(tiny_vae=False)
pipe = AutoPipelineForImage2Image.from_pipe(sd.pipe)   # shares the hooked UNet
pipe.set_progress_bar_config(disable=True)
st = np.load(f"{a.stats}_L{a.layer}.npz")
unit = float(st["resid_norm_q10_50_90"][1])            # "rel" dose unit: the layer's median residual norm
img = Image.open(a.image).convert("RGB").resize((512, 512), Image.LANCZOS)
os.makedirs(a.out, exist_ok=True)
img.save(f"{a.out}/input.jpg", quality=95)
for strength in [float(s) for s in a.strength.split(",")]:
    for dose in [float(d) for d in a.doses.split(",")]:
        push = sd.push_for(a.layer, "neuron", a.neuron, dose, unit)[None]   # [1, dim]
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        with torch.no_grad(), sd.steering({a.layer: push} if dose else {}):
            out = pipe(prompt=a.prompt, image=img, strength=strength, num_inference_steps=a.steps,
                       guidance_scale=0.0, generator=g).images[0]
        name = f"{a.out}/n{a.neuron}_s{strength:g}_{dose:+g}.jpg"
        out.save(name, quality=95); print(name)
print("I2I_DONE")
