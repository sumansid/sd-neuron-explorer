"""Read a ranked sweep (runs/<run>.units.jsonl) and separate dials from destroyers.

    python sweep/subjects.py flux-d6-full [--keep 0.75] [--attr "the Golden Gate Bridge"]

A unit "destroys" the image when cos_base at its best dose falls below --keep. Among units that keep the image,
list the strongest by mass, then for every content attribute the neuron whose CLIP z-shift on that attribute is
largest (a candidate subject neuron). Finally rank all units on one attribute (--attr) regardless of damage.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--keep", type=float, default=0.75, help="min cos_base for a unit to count as keeping the image")
    ap.add_argument("--attr", default="the Golden Gate Bridge")
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--minz", type=float, default=6.0)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(f"runs/{a.run}.units.jsonl")]
    neurons = [r for r in recs if r["kind"] == "neuron" and r.get("dose")]
    randoms = [r for r in recs if r["kind"] == "random" and r.get("dose")]
    nullmax = max(r["mass"] for r in randoms)

    def best(r):
        return r["dose"][f"{r['best_dose']:+g}"]

    cb = np.array([best(r)["cos_base"] for r in neurons])
    rcb = np.array([best(r)["cos_base"] for r in randoms])
    print(f"{len(neurons)} neurons, {len(randoms)} random; cos_base at best dose: neurons p10/p50/p90 = "
          f"{np.percentile(cb, 10):.3f}/{np.percentile(cb, 50):.3f}/{np.percentile(cb, 90):.3f}, "
          f"random p10/p50/p90 = {np.percentile(rcb, 10):.3f}/{np.percentile(rcb, 50):.3f}/{np.percentile(rcb, 90):.3f}")
    destroy = [r for r in neurons if best(r)["cos_base"] < a.keep]
    keep = [r for r in neurons if best(r)["cos_base"] >= a.keep]
    above = [r for r in neurons if r["mass"] > nullmax]
    above_keep = [r for r in above if best(r)["cos_base"] >= a.keep]
    print(f"destroyers (cos_base < {a.keep}): {len(destroy)} ({len(destroy) / len(neurons):.1%}); "
          f"above every random control: {len(above)}, of which keep the image: {len(above_keep)}")

    print(f"\n== strongest units that keep the image (cos_base >= {a.keep}) ==")
    for i, r in enumerate(sorted(keep, key=lambda r: -r["mass"])[: a.show]):
        b = best(r)
        lens = ", ".join(f"{n}({z:+.1f})" for n, z, _ in b["lens"][:4])
        print(f"{i + 1:4d} N{r['neuron']:5d} dose {r['best_dose']:+g} mass {r['mass']:5.1f} cos {b['cos_base']:.3f} "
              f"sat {r.get('z_d_sat', 0):+5.1f} lum {r.get('z_d_lum', 0):+5.1f}  {lens}")

    # per-attribute champions among image-keeping dose entries
    attrs = {}
    for r in neurons:
        for d, v in r["dose"].items():
            if v["cos_base"] < a.keep:
                continue
            for n, z, raw in v["lens"]:
                if z > attrs.get(n, (0,))[0]:
                    attrs[n] = (z, r["neuron"], d, v["cos_base"])
    content = [n for n in attrs if attrs[n][0] >= a.minz]
    print(f"\n== attribute champions among image-keeping doses (z >= {a.minz}) ==")
    for n in sorted(content, key=lambda n: -attrs[n][0]):
        z, neu, d, c = attrs[n]
        print(f"  {n:40s} z {z:+6.1f}  N{neu:5d} dose {d} cos {c:.3f}")

    print(f"\n== top units on '{a.attr}' at any dose (damage shown) ==")
    hits = []
    for r in neurons + randoms:
        for d, v in r["dose"].items():
            for n, z, raw in v["lens"]:
                if n == a.attr:
                    hits.append((z, r["kind"], r["neuron"], d, v["cos_base"], raw))
    for z, k, neu, d, c, raw in sorted(hits, key=lambda h: -h[0])[: a.show]:
        print(f"  {k[:3]} N{neu:5d} dose {d:>4s} z {z:+6.1f} raw {raw:+.3f} cos_base {c:.3f}")
    print(f"  ({len(hits)} dose entries list this attribute in their top 10; random max z = "
          f"{max([h[0] for h in hits if h[1] == 'random'], default=0):+.1f})")


if __name__ == "__main__":
    main()
