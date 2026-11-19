"""Pick, for every prompt, the clamped image that gains the most concept similarity while keeping the scene, across
several clamp runs, and list the full-size files to pull for a before/after gallery.

    python sweep/best_of.py --runs gg-demo-band,gg-demo-family-low,... --min-cos 0.8 --out runs/best_of.json

Writes a JSON list of {prompt, run, thumb, dose, dcos, cos_base, kind} and prints a table.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--min-cos", type=float, default=0.8)
ap.add_argument("--out", default="runs/best_of.json")
ap.add_argument("--top", type=int, default=1, help="candidates per prompt to keep")
a = ap.parse_args()
cands = {}
for run in a.runs.split(","):
    meta = json.load(open(f"runs/{run}.meta.json"))
    for l in open(f"runs/{run}.jsonl"):
        r = json.loads(l)
        if r["kind"] != "clamp":
            continue
        if r["cos_base"] < a.min_cos:
            continue
        dcos = r["cos_concept"] - r["base_cos_concept"]
        cands.setdefault(r["prompt"], []).append({"prompt": r["prompt"], "run": run, "thumb": r["thumb"], "dose": r["dose"],
                                                 "dcos": round(dcos, 4), "cos_base": r["cos_base"], "mask": meta.get("mask", ""),
                                                 "layers": meta.get("layers"), "gg_lens": round(float(r["d_attr"][meta["attributes"].index("the Golden Gate Bridge")]), 2)})
best = []
print(f"{'prompt':10s} {'run':24s} {'dose':>5s} {'dcos':>7s} {'cos':>6s} {'ggz':>5s}")
for p, cs in sorted(cands.items()):
    cs.sort(key=lambda c: -(c["dcos"] + 0.002 * c["gg_lens"]))
    for c in cs[: a.top]:
        best.append(c)
        print(f"{p:10s} {c['run']:24s} {c['dose']:5g} {c['dcos']:+7.3f} {c['cos_base']:6.3f} {c['gg_lens']:+5.1f}")
json.dump(best, open(a.out, "w"), indent=1)
print("wrote", a.out, len(best), "entries")
