"""Print a contrast-search result: python sweep/show_contrast.py stats-flux3/contrast_goldengate.json [top]"""
import json, sys
d = json.load(open(sys.argv[1])); top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
print("templates", d["templates"], "n", d["n"])
print("pixels", {k: (round(v["plain"], 3), round(v["concept"], 3)) for k, v in d["pixels"].items()})
print("per-layer max |d|: " + " ".join(f"{l['name']}:{l['max_abs_d']:.1f}" for l in d["layers"]))
print("top readers:")
for r in d["top"][:top]:
    print(f"  L{r['layer']:2d} {r['name']:4s} N{r['neuron']:5d} d {r['d']:+.2f} t {r['t']:+.1f} agree {r['frac_pos']:.2f}")
