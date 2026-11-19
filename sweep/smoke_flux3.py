"""Smoke test for the FLUX 3 Klein adapter: load, generate two prompts, push one neuron, check the hooks."""
import sys, time
sys.path.insert(0, ".")
import torch
from PIL import Image
from sd_explorer.flux3 import FL3

t0 = time.time()
sd = FL3()
print(f"loaded in {time.time() - t0:.0f}s; layers {len(sd.layers)} dim {sd.dim[0]} inner {sd.inner[0]} t_img {sd.t_img}")
print("pipeline blocks:", list(getattr(sd.pipe.blocks, "sub_blocks", {}).keys()))
prompts = ["a photo of a red sports car on a city street", "a mountain lake at sunrise"]
seen = {}
def cap(li, out): seen[li] = tuple(out.shape)
t0 = time.time()
with sd.capturing([0, 5, 20], cap):
    img = sd.generate(prompts, [1001, 1005])
dt = time.time() - t0
print(f"generated {img.shape} in {dt:.1f}s ({len(prompts) / dt:.2f} img/s); captured {seen}")
Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save("/workspace/smoke_base.jpg")
# a push: does a big neuron push change the image, and does the same seed reproduce the baseline?
li = 10
push = torch.zeros(2, sd.dim[li], device=sd.device); push[0] = sd.push_for(li, "neuron", 123, 64.0, 1.0)
img2 = sd.generate(prompts, [1001, 1005], {li: push})
same = torch.nn.functional.cosine_similarity(img[1].flatten(), img2[1].flatten(), dim=0).item()
diff = torch.nn.functional.cosine_similarity(img[0].flatten(), img2[0].flatten(), dim=0).item()
print(f"unpushed prompt reproduces: cos {same:.4f}; pushed prompt changed: cos {diff:.4f}")
Image.fromarray((img2[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save("/workspace/smoke_push.jpg")
print("SMOKE_OK")
