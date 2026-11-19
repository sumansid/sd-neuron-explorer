"""Per-unit statistics for a sweep: pixel-metric shifts, the CLIP attribute lens, and how far above the
random-direction null each unit sits. Writes runs/<run>.units.jsonl for the explorer and prints the top of the
ranking.

    python sweep/rank.py L40 --show 30 --find 1234

For each unit (neuron or random direction), at each dose sign:
    d_sat, d_colorful, d_lum, d_contrast   mean shift vs the unsteered baseline over prompts
    mse, cos_base                            how much the image changed
    lens                                     attributes whose CLIP score shifted most, as z against the random null
    mass                                     summed |z| of the top 5 attributes at that sign
Unit-level: mass = max over signs; null_pct = share of random controls with lower mass; z_<metric> = the metric
shift in units of the random controls' spread at the same dose.
"""
import argparse
import collections
import json

import numpy as np


def compute_units(rows, attributes, top_attr=10):
    units = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "baseline":
            units[(r["layer"], r["kind"], r["neuron"])].append(r)
    doses = sorted({r["dose"] for r in rows if r["kind"] != "baseline"})
    metrics = ["d_sat", "d_colorful", "d_lum", "d_contrast", "mse", "cos_base", "cos_prompt"]

    # per unit per dose: mean over prompts of each metric and of the attribute shift vector
    per = {}
    for u, rs in units.items():
        for d in doses:
            ds = [r for r in rs if r["dose"] == d]
            if not ds:
                continue
            per[(u, d)] = {m: float(np.mean([r[m] for r in ds])) for m in metrics}
            per[(u, d)]["attr"] = np.mean([r["d_attr"] for r in ds], 0)

    # null per dose from random controls
    null = {}
    for d in doses:
        ctrl = [per[(u, d)] for u in units if u[1] == "random" and (u, d) in per]
        if not ctrl:
            continue
        A = np.stack([c["attr"] for c in ctrl])
        null[d] = {"attr_mu": A.mean(0), "attr_sd": A.std(0) + 0.05,
                   **{m: (float(np.mean([c[m] for c in ctrl])), float(np.std([c[m] for c in ctrl]) + 1e-6)) for m in metrics}}

    records = []
    for u, rs in units.items():
        rec = {"kind": u[1], "neuron": u[2], "layer": u[0], "dose": {}, "mass": 0.0}
        for d in doses:
            if (u, d) not in per or d not in null:
                continue
            p, nl = per[(u, d)], null[d]
            z = (p["attr"] - nl["attr_mu"]) / nl["attr_sd"]
            order = np.argsort(-np.abs(z))[:top_attr]
            mass = float(np.abs(z[order[:5]]).sum())
            rec["dose"][f"{d:+g}"] = {
                **{m: round(p[m], 4) for m in metrics},
                **{f"z_{m}": round((p[m] - nl[m][0]) / nl[m][1], 2) for m in metrics[:4]},
                "lens": [[attributes[i], round(float(z[i]), 2), round(float(p["attr"][i]), 2)] for i in order],
                "mass": round(mass, 2),
            }
            if mass > rec["mass"]:
                rec["mass"], rec["best_dose"] = round(mass, 2), d
        # headline numbers: the sign with the larger |z| for each pixel metric
        for m in metrics[:4]:
            zs = [(abs(v[f"z_{m}"]), v[f"z_{m}"], d) for d, v in rec["dose"].items()]
            if zs:
                rec[f"z_{m}"] = max(zs)[1]
                rec[f"{m}_dose"] = max(zs)[2]
        if rec["dose"]:
            best = rec["dose"][f"{rec['best_dose']:+g}"]
            rec["lens"] = best["lens"][:6]
        records.append(rec)

    nullmass = np.sort([r["mass"] for r in records if r["kind"] == "random"])
    for r in records:
        r["null_pct"] = round(float(np.searchsorted(nullmass, r["mass"]) / len(nullmass)), 4) if len(nullmass) else None
    records.sort(key=lambda r: -r["mass"])
    return records, nullmass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--find", default="")
    ap.add_argument("--metric", default="mass", help="mass | z_d_sat | z_d_lum | z_d_colorful | z_d_contrast")
    a = ap.parse_args()
    meta = json.load(open(f"runs/{a.run}.meta.json"))
    rows = [json.loads(l) for l in open(f"runs/{a.run}.jsonl")]
    records, nullmass = compute_units(rows, meta["attributes"])
    with open(f"runs/{a.run}.units.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    neurons = [r for r in records if r["kind"] == "neuron"]
    if len(nullmass):
        print(f"{len(neurons)} neurons, {len(nullmass)} random controls; random mass p50/p95/max = "
              f"{np.percentile(nullmass, 50):.2f} / {np.percentile(nullmass, 95):.2f} / {nullmass.max():.2f}")
        above = sum(r["mass"] > nullmass.max() for r in neurons)
        print(f"neurons above every random control: {above} ({above / max(len(neurons), 1):.1%})")
    key = (lambda r: -r["mass"]) if a.metric == "mass" else (lambda r: -abs(r.get(a.metric, 0)))
    for i, r in enumerate(sorted(records, key=key)[: a.show]):
        lens = ", ".join(f"{n}({z:+.1f})" for n, z, _ in r.get("lens", [])[:4])
        print(f"{i + 1:4d} {r['kind'][:3]} N{r['neuron']:5d} mass {r['mass']:5.1f} sat {r.get('z_d_sat', 0):+5.1f} "
              f"lum {r.get('z_d_lum', 0):+5.1f} col {r.get('z_d_colorful', 0):+5.1f}  {lens}")
    for n in [int(x) for x in a.find.split(",") if x]:
        rank = next((i + 1 for i, r in enumerate(neurons) if r["neuron"] == n), None)
        rec = next((r for r in neurons if r["neuron"] == n), None)
        if rec:
            print(f"neuron {n}: rank {rank}/{len(neurons)} mass {rec['mass']} null_pct {rec['null_pct']} "
                  f"z_sat {rec.get('z_d_sat')} z_lum {rec.get('z_d_lum')} lens {rec.get('lens')}")


if __name__ == "__main__":
    main()
