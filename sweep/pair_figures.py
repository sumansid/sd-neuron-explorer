"""Figures from the saved galleries (gallery/<name>/<prompt>_{base,steered}.jpg):

  * ladder: the two-neuron Golden Gate at several doses, one row per prompt, columns = off, x0.5, x0.75, x1, x1.25
    and the uneven splits; plus the suspension neuron alone.
  * pairs: clean off | on pairs (no UI) for the write-up and the video stills.

    python sweep/pair_figures.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("gallery/figures", exist_ok=True)
BG, FG, DIM, PINK = (16, 16, 18), (236, 236, 236), (150, 150, 158), (255, 105, 180)


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_T, F_S = font(26), font(17)


def img(path, s):
    try:
        return Image.open(path).convert("RGB").resize((s, s), Image.LANCZOS)
    except Exception:
        return Image.new("RGB", (s, s), (40, 40, 44))


def grid(name, title, subtitle, cols, prompts, tile=200, gap=6, label_w=120):
    """cols: list of (column label, folder, tag) where tag is 'base' or 'steered'."""
    W = label_w + len(cols) * (tile + gap) + gap
    H = 92 + len(prompts) * (tile + gap) + 10
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((gap + 8, 12), title, fill=FG, font=F_T)
    d.text((gap + 8, 48), subtitle, fill=DIM, font=F_S)
    for j, (lab, _, _) in enumerate(cols):
        d.text((label_w + gap + j * (tile + gap) + 4, 70), lab, fill=PINK if j else DIM, font=F_S)
    for i, p in enumerate(prompts):
        y = 92 + i * (tile + gap)
        d.text((gap + 8, y + tile // 2 - 9), p, fill=DIM, font=F_S)
        for j, (_, folder, tag) in enumerate(cols):
            im.paste(img(f"gallery/{folder}/{p}_{tag}.jpg", tile), (label_w + gap + j * (tile + gap), y))
    out = f"gallery/figures/{name}.jpg"
    im.save(out, quality=90)
    print(out, im.size)


def pair(name, folder, prompt, left="off", right="on", size=640):
    im = Image.new("RGB", (size * 2 + 18, size + 40), BG)
    d = ImageDraw.Draw(im)
    im.paste(img(f"gallery/{folder}/{prompt}_base.jpg", size), (6, 6))
    im.paste(img(f"gallery/{folder}/{prompt}_steered.jpg", size), (size + 12, 6))
    d.text((8, size + 12), left, fill=DIM, font=F_S)
    d.text((size + 14, size + 12), right, fill=PINK, font=F_S)
    out = f"gallery/figures/pair_{name}.jpg"
    im.save(out, quality=92)
    print(out)


PROMPTS = ["astronaut", "boat", "castle", "street", "dog", "portrait", "forest", "desert"]

grid("pair_dose_ladder", "The two-neuron Golden Gate at increasing dose (layer 17 neuron 868 + layer 18 neuron 695)",
     "same prompt and seed in each row; the slider in the generate view scales both pushes together",
     [("off", "gg2_full", "base"), ("×0.5 (+1, +1)", "pair_x050", "steered"), ("×0.75 (+1.5, +1.5)", "pair_x075", "steered"),
      ("×1 (+2, +2)", "gg2_full", "steered"), ("×1.25 (+2.5, +2.5)", "pair_x125", "steered")], PROMPTS)
grid("pair_split_ladder", "Uneven splits between the two neurons",
     "the bridge neuron (layer 17) and the suspension neuron (layer 18) at different strengths",
     [("off", "gg2_full", "base"), ("868 +0.5, 695 +2", "pair_a052", "steered"), ("868 +1, 695 +2", "pair_a12", "steered"),
      ("868 +2, 695 +1", "pair_a21", "steered"), ("868 +2, 695 +2", "gg2_full", "steered")], PROMPTS)
grid("susp_dose_ladder", "The suspension-bridge neuron alone (layer 18 neuron 695)", "pushed by 1, 1.5, 2 and 3 units",
     [("off", "susp_x1", "base"), ("+1", "susp_x1", "steered"), ("+1.5", "susp_x15", "steered"), ("+2", "susp_x2", "steered"), ("+3", "susp3_full", "steered")], PROMPTS)

# clean off | on pairs for the video stills and the write-up
for p in ["astronaut", "boat", "castle", "street", "portrait", "dog"]:
    pair(f"gg2_{p}", "gg2_full", p, "neurons off", "layer 17 neuron 868 + layer 18 neuron 695 on (+2, +2)")
for p in ["boat", "dog", "desert", "castle", "astronaut", "street"]:
    pair(f"flux_{p}", "gg", p, "feature off", "FLUX feature 13287 clamped on (horizon band)")
for p in ["street", "astronaut", "library", "portrait"]:
    pair(f"dog_{p}", "dog", p, "neuron off", "layer 18 neuron 2683 on (−2): the dog neuron")
for p in ["breakfast", "street"]:
    pair(f"buzz_{p}", "buzz_full", p, "neuron off", "layer 18 neuron 1505 on (−2): Buzz Lightyear")
print("done")
