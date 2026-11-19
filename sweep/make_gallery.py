"""Assemble before/after pairs into a gallery folder.
    python sweep/make_gallery.py picks --picks runs/best_of_085.json --out gallery/gg
    python sweep/make_gallery.py unit --run sdxl-demo-L18 --neuron 2683 --dose -2 --out gallery/dog
Copies <prompt>_base.jpg and <prompt>_steered.jpg, and writes a 2-column contact sheet gallery.jpg in the folder."""
import argparse, json, os, shutil
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["picks", "unit"])
ap.add_argument("--picks"); ap.add_argument("--run"); ap.add_argument("--neuron", type=int); ap.add_argument("--dose", type=float)
ap.add_argument("--out", required=True); ap.add_argument("--cell", type=int, default=512)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pairs = []
if a.mode == "picks":
    seen = set()
    for b in json.load(open(a.picks)):
        if b["prompt"] in seen: continue
        seen.add(b["prompt"])
        li = b["thumb"].split("_")[0][1:]
        src = f"full/{b['run']}/{b['thumb']}"; base = f"full/{b['run']}/L{li}_base_{b['prompt']}.jpg"
        pairs.append((b["prompt"], base, src, f"{b['run']} x{b['dose']:g}  dcos {b['dcos']:+.3f}  cos {b['cos_base']:.2f}"))
else:
    meta = json.load(open(f"runs/{a.run}.meta.json"))
    for p in meta["prompts"]:
        base = f"thumbs/{a.run}/b-1_{p}_+0.jpg"
        src = f"thumbs/{a.run}/n{a.neuron}_{p}_{a.dose:+g}.jpg"
        pairs.append((p, base, src, f"N{a.neuron} dose {a.dose:+g}"))
for p, base, src, label in pairs:
    shutil.copy(base, f"{a.out}/{p}_base.jpg"); shutil.copy(src, f"{a.out}/{p}_steered.jpg")
W = a.cell; rows = len(pairs)
sheet = Image.new("RGB", (W * 2 + 24, rows * (W + 28) + 8), "white"); dr = ImageDraw.Draw(sheet)
for i, (p, base, src, label) in enumerate(pairs):
    y = 8 + i * (W + 28)
    sheet.paste(Image.open(base).convert("RGB").resize((W, W), Image.LANCZOS), (8, y))
    sheet.paste(Image.open(src).convert("RGB").resize((W, W), Image.LANCZOS), (W + 16, y))
    dr.text((8, y + W + 6), f"{p}: prompt only", fill="black"); dr.text((W + 16, y + W + 6), f"{p}: {label}", fill="black")
sheet.save(f"{a.out}/gallery.jpg", quality=90); print("wrote", f"{a.out}/gallery.jpg", len(pairs), "pairs")
