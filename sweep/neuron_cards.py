"""Neuron cards: before/after figures in the style of Anthropic's feature cards (Scaling Monosemanticity).

Each card: a title bar (layer, neuron, name), a lens line, then one row of images per condition ("off", "on" at the
chosen dose), the same prompt in every column. Sources are the saved sweep thumbnails (thumbs/<run>/n<N>_<prompt>_<dose>.jpg,
baseline b-1_<prompt>_+0.jpg) or a gallery folder (gallery/<name>/<prompt>_{base,steered}.jpg).

    python sweep/neuron_cards.py            # writes gallery/cards/*.jpg
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "gallery/cards"
os.makedirs(OUT, exist_ok=True)

BG = (16, 16, 18)
FG = (236, 236, 236)
DIM = (150, 150, 158)
PINK = (255, 105, 180)


def font(size, bold=False):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=1 if (bold and p.endswith(".ttc")) else 0)
            except Exception:
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE, F_SUB, F_LABEL, F_SMALL = font(30, True), font(20), font(19), font(16)


def tile(path, size):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (size, size), (40, 40, 44))
        ImageDraw.Draw(im).text((8, 8), "missing", fill=DIM, font=F_SMALL)
    return im.resize((size, size), Image.LANCZOS)


def card(name, title, subtitle, lens, rows, prompts, tile_px=224, gap=6, label_w=150):
    """rows: list of (row_label, path_fn) where path_fn(prompt) -> image path."""
    n = len(prompts)
    W = label_w + n * tile_px + (n + 1) * gap
    header = 118
    footer = 34
    H = header + len(rows) * (tile_px + gap) + footer
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((gap + 10, 14), title, fill=FG, font=F_TITLE)
    d.text((gap + 10, 54), subtitle, fill=DIM, font=F_SUB)
    d.text((gap + 10, 84), lens, fill=PINK, font=F_SUB)
    y = header
    for r, (label, fn) in enumerate(rows):
        d.text((gap + 10, y + tile_px // 2 - 12), label, fill=FG if r else DIM, font=F_LABEL)
        for i, p in enumerate(prompts):
            x = label_w + gap + i * (tile_px + gap)
            img.paste(tile(fn(p), tile_px), (x, y))
        y += tile_px + gap
    for i, p in enumerate(prompts):
        x = label_w + gap + i * (tile_px + gap)
        d.text((x + 4, y + 6), p, fill=DIM, font=F_SMALL)
    path = f"{OUT}/{name}.jpg"
    img.save(path, quality=90)
    print(path, img.size)
    return path


def thumbs(run):
    def mk(n, dose):
        def fn(prompt):
            if n is None:
                return f"thumbs/{run}/b-1_{prompt}_+0.jpg"
            return f"thumbs/{run}/n{n}_{prompt}_{dose}.jpg"
        return fn
    return mk


def gallery(folder):
    return (lambda p: f"gallery/{folder}/{p}_base.jpg"), (lambda p: f"gallery/{folder}/{p}_steered.jpg")


SWEEP = ["car", "woman", "fruit", "lake"]
DEMO6 = ["astronaut", "castle", "street", "portrait", "boat", "dog"]
DEMO6B = ["astronaut", "kitchen", "street", "library", "portrait", "forest"]

L18 = thumbs("L18")
L68 = thumbs("L68")
F17 = thumbs("focus-L17")
F18 = thumbs("focus-L18")

# --- subject neurons, SDXL-Turbo layer 18 (full-size galleries where we have them) ---
b, s = gallery("dog")
card("dog", "SDXL-Turbo · layer 18 · neuron 2683 · the dog neuron", "pushed down by 2 units; twelve unrelated prompts, same seeds",
     "lens: a dog +20.9 · an animal +15.5 · beats every random control", [("off", b), ("on (−2)", s)], DEMO6B)
b, s = gallery("n976p")
card("aerial", "SDXL-Turbo · layer 18 · neuron 976 · the aerial-view neuron", "pushed up by 2 units",
     "lens: an aerial view · (CLIP also says \"Golden Gate +20\": it reads maps as bridges)", [("off", b), ("on (+2)", s)], DEMO6B)
card("anime_child", "SDXL-Turbo · layer 18 · neuron 300 · the anime-child neuron", "sweep prompts, pushed down by 2 units",
     "lens: a child +20.3 · a cartoon +16.5 · a baby +15.5", [("off", L18(None, 0)), ("on (−2)", L18(300, "-2"))], SWEEP)
card("buzz", "SDXL-Turbo · layer 18 · neuron 1505 · Buzz Lightyear", "sweep prompts, pushed down by 2 units",
     "every subject becomes the toy astronaut, the fruit bowl included", [("off", L18(None, 0)), ("on (−2)", L18(1505, "-2"))], SWEEP)
card("bird", "SDXL-Turbo · layer 18 · neuron 1532 · bird of prey", "sweep prompts, pushed down by 2 units",
     "a raptor enters the frame; the portrait grows wings", [("off", L18(None, 0)), ("on (−2)", L18(1532, "-2"))], SWEEP)
card("fire", "SDXL-Turbo · layer 18 · neuron 5097 · fire and horror", "sweep prompts, pushed down by 2 units",
     "lens: horror +10 · war +10 · the car burns, faces double and melt", [("off", L18(None, 0)), ("on (−2)", L18(5097, "-2"))], SWEEP)
card("eye", "SDXL-Turbo · layer 18 · neuron 297 · the eye", "sweep prompts, pushed down by 2 units",
     "the frame becomes a close-up eye", [("off", L18(None, 0)), ("on (−2)", L18(297, "-2"))], SWEEP)
card("old_face", "SDXL-Turbo · layer 18 · neuron 3462 · the old face", "sweep prompts, pushed down by 2 units",
     "a weathered old face replaces the subject", [("off", L18(None, 0)), ("on (−2)", L18(3462, "-2"))], SWEEP)
card("dog_ladder", "SDXL-Turbo · layer 18 · neuron 2683 · the dog neuron, dose ladder", "sweep prompts; the subject takes over between −1 and −2",
     "−1: scene intact · −2: a dog's face in the scene's colours", [("off", L18(None, 0)), ("−1", L18(2683, "-1")), ("−2", L18(2683, "-2"))], SWEEP)

# --- dials, layer 68 ---
card("brightness", "SDXL-Turbo · layer 68 · neuron 151 · the brightness dial", "sweep prompts; pushed both ways",
     "−2: mean luminance 0.88 (high key) · +2: 0.09 (night) · plateaus beyond, the schedule's ceiling",
     [("off", L68(None, 0)), ("−2", L68(151, "-2")), ("+2", L68(151, "+2"))], SWEEP)
card("saturation", "SDXL-Turbo · layer 68 · neuron 485 · the saturation dial", "sweep prompts; pushed both ways",
     "−2: Δsat −0.11, muted to near-grey · +2: +0.21, warm and saturated",
     [("off", L68(None, 0)), ("−2", L68(485, "-2")), ("+2", L68(485, "+2"))], SWEEP)
card("red_tint", "SDXL-Turbo · layer 68 · neuron 2092 · the red-tint dial", "sweep prompts; the dial that made the bridges orange",
     "a colour cast in latent-channel space, one block before the pixels",
     [("off", L68(None, 0)), ("+1", L68(2092, "+1")), ("+2", L68(2092, "+2"))], SWEEP)

# --- the bridge circuit ---
card("bridge_L17N868", "SDXL-Turbo · layer 17 · neuron 868 · the bridge neuron", "twelve demo prompts (six shown), pushed up by 2 units",
     "lens: a suspension bridge +4.9 · similarity to baseline 0.81 · every prompt gets a bridge",
     [("off", F17(None, 0)), ("on (+2)", F17(868, "+2"))], DEMO6)
card("susp_L18N695", "SDXL-Turbo · layer 18 · neuron 695 · the suspension-bridge neuron", "twelve demo prompts (six shown), pushed up by 3 units",
     "lens: a suspension bridge +4.5 · similarity 0.77 · towers and cables; pushed down it deletes the Golden Gate",
     [("off", F18(None, 0)), ("on (+3)", F18(695, "+3"))], DEMO6)
b, s = gallery("gg2_full")
card("pair", "SDXL-Turbo · layers 17 + 18 · neurons 868 + 695 · the two-neuron Golden Gate", "both pushed up by 2 units; full-resolution decode",
     "lens: \"the Golden Gate Bridge\" is the top shift on 5 of 12 prompts, second on 4 more (portrait +13.1, astronaut +11.7, castle +8.5)",
     [("off", b), ("on (+2, +2)", s)], DEMO6)
b, s = gallery("gg")
card("flux_feature", "FLUX (Shuttle 3) · block s5 · SAE feature 13287 · the Golden Gate feature", "best clamp per prompt, horizon band or family, scene kept (similarity ≥ 0.85)",
     "fires only on Golden Gate images (37.7 vs 1.6 elsewhere, 3.9 on the Brooklyn Bridge)",
     [("off", b), ("clamped on", s)], ["boat", "dog", "desert", "castle", "street", "astronaut"])
print("done")

# --- twelve-prompt cards from the full-resolution galleries made on the live pod ---
DEMO12 = ["cat", "kitchen", "astronaut", "breakfast", "castle", "dog", "street", "library", "desert", "portrait", "boat", "forest"]
for name, folder, title, sub, lens in [
    ("buzz12", "buzz_full", "SDXL-Turbo · layer 18 · neuron 1505 · Buzz Lightyear", "twelve prompts, pushed down by 2 units, full-resolution decode", "every subject becomes the toy astronaut"),
    ("bird12", "bird_full", "SDXL-Turbo · layer 18 · neuron 1532 · bird of prey", "twelve prompts, pushed down by 2 units", "a raptor enters every frame"),
    ("fire12", "horror_full", "SDXL-Turbo · layer 18 · neuron 5097 · fire and horror", "twelve prompts, pushed down by 2 units", "lens: horror, war, fire"),
    ("eye12", "eye_full", "SDXL-Turbo · layer 18 · neuron 297 · the eye", "twelve prompts, pushed down by 2 units", "the frame becomes a close-up eye"),
    ("oldface12", "oldface_full", "SDXL-Turbo · layer 18 · neuron 3462 · the old face", "twelve prompts, pushed down by 2 units", "a weathered old face replaces the subject"),
    ("anime12", "anime_full", "SDXL-Turbo · layer 18 · neuron 300 · the anime child", "twelve prompts, pushed down by 2 units", "lens: a child, a cartoon, a baby"),
]:
    b, s = gallery(folder)
    card(name, title, sub, lens, [("off", b), ("on (−2)", s)], DEMO12, tile_px=150, label_w=120)
bb, sb = gallery("bright_full"); _, sd = gallery("dark_full")
card("brightness12", "SDXL-Turbo · layer 68 · neuron 151 · the brightness dial", "twelve prompts, pushed both ways", "−2: high key · +2: night; the model will not go past 0.9 or below 0.1 mean luminance",
     [("off", bb), ("−2", sb), ("+2", sd)], DEMO12, tile_px=150, label_w=120)
bs, ss = gallery("sat_full"); _, sdsat = gallery("desat_full")
card("saturation12", "SDXL-Turbo · layer 68 · neuron 485 · the saturation dial", "twelve prompts, pushed both ways", "−2: muted to near-grey · +2: warm and saturated",
     [("off", bs), ("−2", sdsat), ("+2", ss)], DEMO12, tile_px=150, label_w=120)
print("cards12 done")
