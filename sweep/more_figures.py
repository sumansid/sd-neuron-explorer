"""More figures for the write-up, all from saved data (no GPU):

  * tier_L*.jpg      the next tier of neurons per swept layer, off | on rows from the sweep thumbnails
  * cards/*12.jpg    twelve-prompt off/on cards for the galleries not yet carded
  * necessity_panel  the removal test: three Golden Gate prompts, pushed down
  * pair_*.jpg       clean off | on pairs, and 2-column pair grids

    python sweep/more_figures.py
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, "sweep")
from neuron_cards import card, gallery, font, tile, BG, FG, DIM, PINK  # noqa: E402

os.makedirs("gallery/figures", exist_ok=True)
F_T, F_S, F_XS = font(26, True), font(17), font(14)
DEMO12 = ["cat", "kitchen", "astronaut", "breakfast", "castle", "dog", "street", "library", "desert", "portrait", "boat", "forest"]
SWEEP = ["car", "woman", "fruit", "lake"]


def units(run):
    out = {}
    for l in open(f"runs/{run}.units.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "neuron":
            out[r["neuron"]] = r
    return out


def tier(run, layer, neurons, title, subtitle, tile_px=140, gap=5, label_w=300):
    U = units(run)
    n = len(SWEEP)
    W = label_w + (2 * n + 1) * (tile_px + gap) + gap
    H = 100 + len(neurons) * (tile_px + gap) + 30
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((12, 12), title, fill=FG, font=F_T)
    d.text((12, 50), subtitle, fill=DIM, font=F_S)
    x0 = label_w + gap
    d.text((x0 + 4, 78), "off (the unsteered image)", fill=DIM, font=F_S)
    d.text((x0 + (n + 1) * (tile_px + gap) + 4, 78), "on (the neuron pushed, dose in the label)", fill=PINK, font=F_S)
    y = 100
    for nn, dose in neurons:
        r = U[nn]
        v = r["dose"][dose]
        lens = ", ".join(f"{a} {s:+.0f}" for a, s, _ in v["lens"][:3])
        d.text((12, y + 18), f"N{nn}  dose {dose}", fill=FG, font=F_S)
        d.text((12, y + 44), f"mass {v['mass']:.0f} · similarity {v['cos_base']:.2f}", fill=DIM, font=F_XS)
        # wrap lens
        words, line, lines = lens.split(" "), "", []
        for w in words:
            if d.textlength(line + " " + w, font=F_XS) > label_w - 24:
                lines.append(line); line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        for k, ln in enumerate(lines[:3]):
            d.text((12, y + 66 + 17 * k), ln, fill=PINK, font=F_XS)
        for i, p in enumerate(SWEEP):
            im.paste(tile(f"thumbs/{run}/b-1_{p}_+0.jpg", tile_px), (x0 + i * (tile_px + gap), y))
            im.paste(tile(f"thumbs/{run}/n{nn}_{p}_{dose}.jpg", tile_px), (x0 + (n + 1 + i) * (tile_px + gap), y))
        y += tile_px + gap
    for i, p in enumerate(SWEEP):
        d.text((x0 + i * (tile_px + gap) + 4, y + 6), p, fill=DIM, font=F_XS)
        d.text((x0 + (n + 1 + i) * (tile_px + gap) + 4, y + 6), p, fill=DIM, font=F_XS)
    out = f"gallery/figures/tier_{run}.jpg"
    im.save(out, quality=90)
    print(out, im.size)


tier("L18", 18, [(3704, "+2"), (2356, "-2"), (3641, "-2"), (4168, "+2"), (938, "-2"), (568, "-2"), (1622, "+2"), (2886, "-2"),
                 (216, "+2"), (3254, "-2"), (2298, "-2"), (1172, "+2"), (1480, "-2"), (508, "+2")],
     "SDXL-Turbo · layer 18 · the next tier of subject neurons",
     "the fourteen highest-ranked neurons after the ones on the cards above; sweep prompts, one push each")
tier("L68", 68, [(889, "-2"), (2081, "-2"), (953, "+2"), (1864, "-2"), (777, "+2"), (1496, "+2"), (294, "+2"), (526, "+2"),
                 (382, "-2"), (2295, "+2"), (1223, "+2"), (1707, "+2")],
     "SDXL-Turbo · layer 68 · more dials: night, neon, tints, grain",
     "the last block writes tone; its top neurons are casts and textures, not subjects")
tier("L10", 10, [(3985, "-2"), (206, "-2"), (2763, "-2"), (4745, "+2"), (2693, "-2"), (4952, "+2"), (1783, "+2"), (544, "+2"),
                 (1288, "-2"), (4685, "+2"), (4548, "+2"), (3739, "-2")],
     "SDXL-Turbo · layer 10 · layout and tiling neurons",
     "the first deep layer repeats, tiles and cartoons the image; the two at the bottom are its best desaturators")
tier("L46", 46, [(1519, "+2"), (4163, "+2"), (4533, "+2"), (1498, "+2"), (1921, "-2"), (1573, "-2"), (2258, "+2"), (3126, "-2"),
                 (4610, "-2"), (639, "+2"), (1279, "-2"), (2907, "+2")],
     "SDXL-Turbo · layer 46 · style and structure neurons",
     "the up block reads the Golden Gate most strongly but writes maps, diagrams, old faces and glitch")

# --- twelve-prompt cards for the galleries not yet carded ---
for name, folder, title, sub, lens, on in [
    ("dog12", "dog", "SDXL-Turbo · layer 18 · neuron 2683 · the dog neuron", "all twelve prompts, pushed down by 2 units, full-resolution decode", "every prompt becomes a dog that keeps the prompt's world", "on (−2)"),
    ("aerial12", "n976p", "SDXL-Turbo · layer 18 · neuron 976 · the aerial-view neuron, pushed up", "all twelve prompts, +2", "the scene is re-rendered from above", "on (+2)"),
    ("aerialdown12", "n976m", "SDXL-Turbo · layer 18 · neuron 976 · the same neuron, pushed down", "all twelve prompts, −2", "the opposite direction: close-ups and interiors", "on (−2)"),
    ("bridge12", "arch2_full", "SDXL-Turbo · layer 17 · neuron 868 · the bridge neuron", "all twelve prompts, +2", "a bridge in every prompt: steel over the pancakes, a tower in the fog, a bridge in the forest", "on (+2)"),
    ("pair12", "gg2_full", "SDXL-Turbo · layers 17 + 18 · neurons 868 + 695 · the two-neuron Golden Gate", "all twelve prompts, +2 and +2", "\"the Golden Gate Bridge\" is the top lens shift on 5 of 12, second on 4 more", "on (+2, +2)"),
    ("pairgentle12", "pair_x050", "SDXL-Turbo · the two-neuron Golden Gate, gentle", "all twelve prompts, +1 and +1", "similarity to baseline 0.77: the subject survives more often, the bridge still arrives", "on (+1, +1)"),
    ("gg3_12", "gg3_full", "SDXL-Turbo · the pair plus the red-tint dial", "+2, +2 and layer 68 neuron 2092 at +0.5", "the dial was meant to paint the towers orange; it mostly adds glitch", "on (3 units)"),
    ("susp15_12", "susp_x15", "SDXL-Turbo · layer 18 · neuron 695 · the suspension-bridge neuron at +1.5", "all twelve prompts", "the scene-keeping setting: the subject stays, a suspension bridge appears behind it", "on (+1.5)"),
    ("susp3_12", "susp3_full", "SDXL-Turbo · layer 18 · neuron 695 · the suspension-bridge neuron at +3", "all twelve prompts", "at +3 the subject gives way to towers and cables", "on (+3)"),
    ("flux12", "gg", "FLUX (Shuttle 3) · block s5 · SAE feature 13287 · the Golden Gate feature", "all twelve prompts, best clamp per prompt with similarity ≥ 0.85", "it works on open scenes with a horizon; close-ups, interiors and faces only get its material", "clamped on"),
]:
    b, s = gallery(folder)
    card(name, title, sub, lens, [("off", b), (on, s)], DEMO12, tile_px=150, label_w=120)

# --- the necessity test, as a panel ---
PR = [("ggshirt", "a man in a white t-shirt, the Golden Gate behind him"), ("ggflower", "poppies in front of the Golden Gate"), ("ggsunset", "the Golden Gate at sunset, from the bay")]
cols = [("off", "remove_susp", "base"), ("L18 N695 −2", "remove_susp", "steered"), ("L18 N695 −3", "remove_susp3", "steered"), ("868 −2 + 695 −2", "remove_pair", "steered")]
T, G, LW = 240, 6, 250
W = LW + len(cols) * (T + G) + G
H = 96 + len(PR) * (T + G) + 8
im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)
d.text((12, 12), "The necessity test: push the suspension-bridge neuron down on prompts that ask for the bridge", fill=FG, font=F_T)
d.text((12, 48), "the bridge is in the prompt, so an unsteered model paints it; a neuron the bridge needs should be able to take it away", fill=DIM, font=F_S)
for j, (lab, _, _) in enumerate(cols):
    d.text((LW + G + j * (T + G) + 4, 74), lab, fill=PINK if j else DIM, font=F_S)
for i, (key, desc) in enumerate(PR):
    y = 96 + i * (T + G)
    words, line, lines = desc.split(" "), "", []
    for w in words:
        if d.textlength(line + " " + w, font=F_XS) > LW - 24:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    for k, ln in enumerate(lines):
        d.text((12, y + T // 2 - 20 + 17 * k), ln, fill=DIM, font=F_XS)
    for j, (_, folder, tag) in enumerate(cols):
        im.paste(tile(f"gallery/{folder}/{key}_{tag}.jpg", T), (LW + G + j * (T + G), y))
im.save("gallery/figures/necessity_panel.jpg", quality=92)
print("gallery/figures/necessity_panel.jpg", im.size)


# --- clean pairs and pair grids ---
def pair(name, folder, prompt, left, right, size=640):
    im = Image.new("RGB", (size * 2 + 18, size + 40), BG)
    d = ImageDraw.Draw(im)
    im.paste(tile(f"gallery/{folder}/{prompt}_base.jpg", size), (6, 6))
    im.paste(tile(f"gallery/{folder}/{prompt}_steered.jpg", size), (size + 12, 6))
    d.text((8, size + 12), left, fill=DIM, font=F_S)
    d.text((size + 14, size + 12), right, fill=PINK, font=F_S)
    out = f"gallery/figures/pair_{name}.jpg"
    im.save(out, quality=92)
    print(out)


def pairgrid(name, folder, prompts, title, right, size=300, gap=8):
    cols = 2
    rows = (len(prompts) + cols - 1) // cols
    cw = size * 2 + 6
    W = gap + cols * (cw + gap)
    H = 60 + rows * (size + 28 + gap)
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((gap + 4, 12), title, fill=FG, font=F_T)
    for k, p in enumerate(prompts):
        r, c = divmod(k, cols)
        x = gap + c * (cw + gap)
        y = 60 + r * (size + 28 + gap)
        im.paste(tile(f"gallery/{folder}/{p}_base.jpg", size), (x, y))
        im.paste(tile(f"gallery/{folder}/{p}_steered.jpg", size), (x + size + 6, y))
        d.text((x + 2, y + size + 6), f"{p} · off", fill=DIM, font=F_XS)
        d.text((x + size + 8, y + size + 6), f"{p} · {right}", fill=PINK, font=F_XS)
    out = f"gallery/figures/grid_{name}.jpg"
    im.save(out, quality=90)
    print(out, im.size)


for p in ["astronaut", "boat", "dog", "street", "portrait", "castle"]:
    pair(f"gentle_{p}", "pair_x050", p, "neurons off", "the two-neuron Golden Gate, gentle (+1, +1)")
for p in ["boat", "dog", "street", "portrait", "library", "forest"]:
    pair(f"susp15_{p}", "susp_x15", p, "neuron off", "layer 18 neuron 695 at +1.5: the suspension-bridge neuron")
pairgrid("flux_rest", "gg", ["cat", "kitchen", "breakfast", "library", "portrait", "forest"], "FLUX feature 13287, the six prompts where it only adds its material", "clamped on")
pairgrid("dog_rest", "dog", ["cat", "kitchen", "forest", "boat", "desert", "castle"], "The dog neuron (layer 18 neuron 2683 at −2), six more prompts", "on")
pairgrid("bridge_rest", "arch2_full", ["castle", "dog", "portrait", "cat", "breakfast", "library"], "The bridge neuron (layer 17 neuron 868 at +2), six prompts", "on")
pairgrid("aerial_rest", "n976p", ["kitchen", "breakfast", "library", "portrait", "cat", "forest"], "The aerial-view neuron (layer 18 neuron 976 at +2), six prompts", "on")
pairgrid("buzz_rest", "buzz_full", ["cat", "castle", "dog", "library", "portrait", "forest"], "Buzz Lightyear (layer 18 neuron 1505 at −2), six prompts", "on")
pairgrid("bird_rest", "bird_full", ["cat", "kitchen", "astronaut", "dog", "portrait", "boat"], "Bird of prey (layer 18 neuron 1532 at −2), six prompts", "on")
pairgrid("horror_rest", "horror_full", ["cat", "kitchen", "castle", "dog", "street", "portrait"], "Fire and horror (layer 18 neuron 5097 at −2), six prompts", "on")
pairgrid("eye_rest", "eye_full", ["cat", "astronaut", "castle", "dog", "boat", "forest"], "The eye (layer 18 neuron 297 at −2), six prompts", "on")
pairgrid("oldface_rest", "oldface_full", ["cat", "kitchen", "astronaut", "dog", "street", "boat"], "The old face (layer 18 neuron 3462 at −2), six prompts", "on")
pairgrid("anime_rest", "anime_full", ["cat", "astronaut", "dog", "street", "portrait", "boat"], "The anime child (layer 18 neuron 300 at −2), six prompts", "on")
pairgrid("bright_rest", "bright_full", ["cat", "kitchen", "castle", "street", "portrait", "forest"], "The brightness dial pushed to high key (layer 68 neuron 151 at −2)", "on")
pairgrid("dark_rest", "dark_full", ["cat", "kitchen", "castle", "street", "portrait", "forest"], "The brightness dial pushed to night (layer 68 neuron 151 at +2)", "on")
pairgrid("sat_rest", "sat_full", ["cat", "breakfast", "dog", "street", "boat", "forest"], "The saturation dial pushed up (layer 68 neuron 485 at +2)", "on")
pairgrid("desat_rest", "desat_full", ["cat", "breakfast", "dog", "street", "boat", "forest"], "The saturation dial pushed down (layer 68 neuron 485 at −2)", "on")
print("done")

# --- highlight cards from the tier galleries: off / −2 / +2 so both directions show ---
from neuron_cards import thumbs  # noqa: E402
HL = [
    ("L18", 568, "wine barrels", "pushed down: every subject becomes a stack of barrels (the lens has no barrel word)"),
    ("L18", 4168, "lips and a mouth", "pushed up: a close-up mouth in the colours of the scene"),
    ("L18", 1480, "the airliner", "pushed down: a passenger jet on the tarmac, over the flowers, on the lake"),
    ("L18", 508, "the helicopter", "pushed up: a helicopter, red for the car prompt"),
    ("L18", 2298, "the torso", "pushed down: a bare torso wearing the scene's colours; lens 'a man +18'"),
    ("L18", 216, "the old master", "pushed up: a Renaissance portrait; the woman becomes a Mona Lisa"),
    ("L18", 3704, "the cartoon", "pushed up: a children's-book drawing of the same scene; lens 'a cartoon +24'"),
    ("L18", 1172, "the carved sign", "pushed up: the subject becomes a carved sign with its own name on it"),
    ("L18", 1622, "the vintage advertisement", "pushed up: a page from an old catalogue, the subject as the product"),
    ("L18", 2356, "the web page", "pushed down: a screenshot of a website about the subject"),
    ("L18", 3641, "the old-master portrait", "pushed down: an 18th-century painted portrait replaces the subject"),
    ("L18", 938, "the city from above", "pushed down: Manhattan from a rooftop; the woman stands on a skyscraper"),
    ("L10", 206, "the comic halftone", "pushed down: bold outlines and Ben-Day dots"),
    ("L10", 4745, "the pixel cartoon", "pushed up: a flat, posterised pixel-art version of the scene"),
    ("L46", 1498, "the isometric cube", "pushed up: everything is built from cubes"),
    ("L46", 3126, "the pencil sketch", "pushed down: a graphite drawing; the strongest desaturator in layer 46"),
    ("L46", 1921, "carved text", "pushed down: the subject cut in relief into a wall of letters"),
    ("L68", 526, "acid green", "pushed up: a yellow-green cast over everything"),
    ("L68", 953, "magenta", "pushed up: the whole image goes magenta"),
    ("L68", 1223, "blue night", "pushed up: a blue night cast that keeps the scene (similarity 0.81)"),
]
for run, n, title, note in HL:
    T = thumbs(run)
    U = units(run)
    v2m, v2p = U[n]["dose"]["-2"], U[n]["dose"]["+2"]
    lens_m = f"{v2m['lens'][0][0]} {v2m['lens'][0][1]:+.0f}"
    lens_p = f"{v2p['lens'][0][0]} {v2p['lens'][0][1]:+.0f}"
    lens_line = f"lens −2: {lens_m} · +2: {lens_p} · similarity {v2m['cos_base']:.2f} / {v2p['cos_base']:.2f}"
    from neuron_cards import F_SUB
    from PIL import ImageDraw as _ID, Image as _I
    _d = _ID.Draw(_I.new("RGB", (10, 10)))
    while _d.textlength(lens_line, font=F_SUB) > 1040 and len(lens_line) > 40:
        lens_line = lens_line[:-12].rstrip() + "…"
    card(f"hl_{run}_{n}", f"SDXL-Turbo · layer {run[1:]} · neuron {n} · {title}", note, lens_line,
         [("off", T(None, 0)), ("−2", T(n, "-2")), ("+2", T(n, "+2"))], SWEEP)
print("highlights done")
