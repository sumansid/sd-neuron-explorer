"""Contact sheet for one unit of a run: rows = prompts, columns = baseline then doses.

    python sweep/sheet.py L68 1041 --out sheets/L68_N1041.jpg [--kind random]
"""
import argparse
import json
import os

from PIL import Image, ImageDraw


def sheet(run, kind, n, out, size=160):
    rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
    meta = json.load(open(f"runs/{run}.meta.json"))
    base = {r["prompt"]: r for r in rows if r["kind"] == "baseline"}
    mine = {(r["prompt"], r["dose"]): r for r in rows if r["kind"] == kind and r["neuron"] == n}
    doses = sorted(meta["doses"])
    prompts = meta["prompts"]
    im = Image.new("RGB", (size * (len(doses) + 1), size * len(prompts) + 16), "black")
    d = ImageDraw.Draw(im)
    d.text((4, 2), f"{run} {kind} {n}  base", fill="white")
    for j, x in enumerate(doses):
        d.text(((j + 1) * size + 4, 2), f"{x:+g}", fill="white")
    for i, p in enumerate(prompts):
        im.paste(Image.open(f"thumbs/{run}/{base[p]['thumb']}").resize((size, size)), (0, 16 + i * size))
        for j, x in enumerate(doses):
            r = mine.get((p, x))
            if r:
                im.paste(Image.open(f"thumbs/{run}/{r['thumb']}").resize((size, size)), ((j + 1) * size, 16 + i * size))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    im.save(out, quality=88)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("neuron", type=int)
    ap.add_argument("--kind", default="neuron"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(sheet(a.run, a.kind, a.neuron, a.out or f"sheets/{a.run}_{a.kind[0]}{a.neuron}.jpg"))
