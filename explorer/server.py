"""Explorer server: serves the sweep results (runs/, thumbs/, stats/) and, with --live, the model for steering any
neuron on your own prompt.

    python explorer/server.py                 # results only, http://localhost:8791
    python explorer/server.py --live          # also loads SDXL-Turbo for live steering (GPU)

The sidebar assistant (explorer/assistant.py) needs ANTHROPIC_API_KEY in the environment.
"""
import argparse
import base64
import glob
import io
import json
import os
import sys
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
app = FastAPI()
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402  (the map JSON and the trace payloads shrink 3-6x over a tunnel)
app.add_middleware(GZipMiddleware, minimum_size=2048)
LIVE = {"sd": None, "models": {}, "model": None, "lens": None, "lock": threading.Lock(), "sae": {}}


def live_model(name=None):
    """The loaded image model to use: the one asked for, else the only/first one loaded. 503 when none is."""
    ms = LIVE["models"]
    if not ms:
        raise HTTPException(503, "server started without --live")
    if name and name in ms:
        return name, ms[name]
    if name and name not in ms:
        raise HTTPException(400, f"{name} is not loaded (loaded: {', '.join(ms)}); start with --live {name}")
    first = next(iter(ms))
    return first, ms[first]
_cache = {}
MODEL_NAME = os.environ.get("EXPLORER_ASSISTANT_MODEL", "claude-opus-5-5")


def run_model(meta, run):
    """Which image model a run belongs to: stored in newer metas, inferred from the run name for older ones."""
    return meta.get("model") or ("flux3" if run.startswith("flux3") else "flux" if run.startswith("flux") else "sdxl")


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def is_sae(meta):
    """SAE clamp runs (sweep/sae_steer.py) clamp features to multiples of their corpus max instead of pushing neurons."""
    return meta.get("dose_unit") == "feature_max_multiple"


def runs():
    out = []
    for p in sorted(glob.glob("runs/*.meta.json")):
        name = os.path.basename(p)[: -len(".meta.json")]
        meta = json.load(open(p))
        meta["run"] = name
        meta["sae"] = is_sae(meta)
        meta["ranked"] = os.path.exists(f"runs/{name}.units.jsonl") or (meta["sae"] and os.path.exists(f"runs/{name}.jsonl"))
        out.append(meta)
    return out


def load_run(run):
    key = ("run", run, os.path.getmtime(f"runs/{run}.jsonl"))
    if key not in _cache:
        rows = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
        by_unit = {}
        for r in rows:
            by_unit.setdefault((r["layer"], r["kind"], r["neuron"]), []).append(r)
        units = [json.loads(l) for l in open(f"runs/{run}.units.jsonl")] if os.path.exists(f"runs/{run}.units.jsonl") else []
        _cache.clear()
        _cache[key] = {"rows": rows, "by_unit": by_unit, "units": units, "meta": json.load(open(f"runs/{run}.meta.json"))}
    return _cache[key]


@app.get("/api/index")
def index():
    stats = json.load(open("stats/sdxl-turbo.json")) if os.path.exists("stats/sdxl-turbo.json") else []
    contrasts = [os.path.basename(p)[len("contrast_"):-5] for p in sorted(glob.glob("stats/contrast_*.json"))]
    names = json.load(open("explorer/names.json")) if os.path.exists("explorer/names.json") else {}
    rs = runs()
    for r in rs:
        r["model"] = run_model(r, r["run"])
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                   or os.path.isdir(os.path.expanduser("~/.config/anthropic")))
    return {"runs": rs, "layers": stats, "contrasts": contrasts, "live": bool(LIVE["models"]), "live_model": LIVE.get("model"),
            "live_models": list(LIVE["models"]),
            "names": names, "assistant": has_key, "assistant_model": MODEL_NAME, "connections": conn() is not None}


@app.get("/api/units")
def units(run: str):
    d = load_run(run)
    slim = []
    for u in d["units"]:
        slim.append({k: u.get(k) for k in ("kind", "neuron", "layer", "mass", "null_pct", "best_dose", "lens",
                                            "z_d_sat", "z_d_lum", "z_d_colorful", "z_d_contrast")})
    return {"units": slim, "meta": d["meta"]}


@app.get("/api/unit")
def unit(run: str, n: int, kind: str = "neuron", layer: int = -1):
    d = load_run(run)
    layer = d["meta"]["layer"] if layer < 0 else layer
    rows = d["by_unit"].get((layer, kind, n))
    if rows is None:
        raise HTTPException(404, "no such unit in this run")
    base = next((v for k, v in d["by_unit"].items() if k[1] == "baseline"), [])
    rec = next((u for u in d["units"] if u["kind"] == kind and u["neuron"] == n and u["layer"] == layer), None)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("attr", "d_attr")}
    return {"rows": [strip(r) for r in rows], "baseline": [strip(r) for r in base], "unit": rec, "meta": d["meta"]}


@app.get("/api/contrast")
def contrast(concept: str):
    p = f"stats/contrast_{concept}.json"
    if not os.path.exists(p):
        raise HTTPException(404)
    return json.load(open(p))


@app.get("/api/layer_stats")
def layer_stats(layer: int):
    st = np.load(f"stats/sdxl-turbo_L{layer}.npz")
    return {"abs_max": st["abs_max"].round(3).tolist(), "mean": st["mean"].round(4).tolist(),
            "col_norm": st["col_norm"].round(4).tolist(), "resid": st["resid_norm_q10_50_90"].tolist()}


@app.post("/api/name")
async def set_name(req: Request):
    b = await req.json()
    p = "explorer/names.json"
    names = json.load(open(p)) if os.path.exists(p) else {}
    names[f"L{b['layer']}{'F' if b.get('kind') in ('clamp', 'feature') else 'N'}{b['neuron']}"] = b.get("name", "")
    json.dump(names, open(p, "w"), indent=1)
    return {"ok": True}


@app.get("/thumbs/{run}/{name}")
def thumb(run: str, name: str):
    p = f"thumbs/{run}/{name}"
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "max-age=86400"})


def steer_impl(b):
    """Live steering: {prompt, layer, neuron, doses:[...], seed, kind, size} -> PNGs (base64) per dose."""
    model, sd = live_model(b.get("model"))
    lens = LIVE["lens"]
    li, n = int(b["layer"]), int(b.get("neuron", 0))
    doses = [float(x) for x in b.get("doses", [-1, 0, 1])]
    kind = b.get("kind", "neuron")
    seed = int(b.get("seed", 1001))
    import torch
    from PIL import Image
    from sd_explorer.metrics import pixel_metrics
    with LIVE["lock"]:
        if kind == "clamp":
            # Anthropic-style clamp of one SAE feature (FLUX only): doses are multiples of the feature's corpus max
            if model != "flux":
                raise HTTPException(400, "feature clamping needs the FLUX model (--live flux)")
            with clamp_context(sd, li, n, doses, b.get("mask", "")):
                img = sd.generate([b["prompt"]] * len(doses), [seed] * len(doses), full_vae=bool(b.get("full_vae", True)))
        else:
            pushes = torch.zeros(len(doses), sd.dim[li], device=sd.device)
            for i, d in enumerate(doses):
                if d != 0:
                    pushes[i] = sd.push_for(li, kind, n, d, push_unit(model, li))
            img = sd.generate([b["prompt"]] * len(doses), [seed] * len(doses), {li: pushes}, full_vae=bool(b.get("full_vae", True)))
        pm = pixel_metrics(img)
        emb = lens.embed_images(img)
        attr = lens.attribute_scores(emb)
    outs = []
    size = int(b.get("size", 384))
    for i, d in enumerate(doses):
        t = (img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(t).resize((size, size), Image.LANCZOS).save(buf, format="JPEG", quality=90)
        top = np.argsort(-(attr[i] - attr[doses.index(0.0)] if 0.0 in doses else attr[i]).cpu().numpy())[:6]
        outs.append({"dose": d, "img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                     **{k: round(float(v[i]), 4) for k, v in pm.items()},
                     "attr": [[lens.attributes[j], round(float(attr[i][j]), 2)] for j in top]})
    return {"images": outs}


STATS_PREFIX = {"sdxl": "stats/sdxl-turbo", "flux": "stats/flux", "flux3": "stats-flux3/flux3"}


def push_unit(model, li):
    """One dose of a neuron push: the median residual norm at that layer (the sweeps' 'rel' unit)."""
    p = f"{STATS_PREFIX.get(model, 'stats/' + model)}_L{li}.npz"
    if not os.path.exists(p):
        raise HTTPException(404, f"no layer stats for {model} L{li} ({p})")
    return float(np.load(p)["resid_norm_q10_50_90"][1])


def sae_for(sd, li):
    """SAE weights for a FLUX layer on the model's device (cached), plus each feature's corpus max for dose units."""
    if li not in LIVE["sae"]:
        import torch
        from explorer.build_connections import load_npy
        p = f"sae/flux2/sae_L{li}_fp16.pt"
        if not os.path.exists(p):
            raise HTTPException(404, f"no SAE weights for L{li} in sae/flux2")
        s = {k: v.float().to(sd.device) for k, v in torch.load(p, map_location="cpu").items() if hasattr(v, "float")}
        s["fmax"] = load_npy(f"sae/flux/feat_L{li}_max.npy").float().max(0).values.to(sd.device)
        LIVE["sae"][li] = s
    return LIVE["sae"][li]


def clamp_context(sd, li, f, doses, mask=""):
    """FLX.editing hook that clamps SAE feature f at layer li to dose x (its corpus max) on every image token
    (or on a horizon band, mask 'band:0.3:0.55'); the same edit as sweep/sae_steer.py, one feature, one layer."""
    import torch
    s = sae_for(sd, li)
    fmax = s["fmax"]
    enc_w, enc_b, dec = s["W_enc"][:, [f]], s["b_enc"][[f]], s["W_dec"][[f]] * s["sigma"]
    t_abs = torch.tensor([[d * float(fmax[f])] if d != 0 else [float("nan")] for d in doses], device=sd.device)  # [B, 1]
    G = sd.h // 16
    tok_mask = torch.ones(G * G, device=sd.device)
    if mask.startswith("band:"):
        lo, hi = (float(x) for x in mask.split(":")[1:3])
        rows = torch.arange(G * G, device=sd.device) // G
        tok_mask = ((rows >= lo * G) & (rows < hi * G)).float()

    def edit(hid):
        xs = (hid.float() - s["mu"]) / s["sigma"]
        act = torch.relu((xs - s["b_pre"]) @ enc_w + enc_b)  # [B, T, 1]
        delta = torch.nan_to_num(t_abs[:, None, :] - act) * tok_mask[None, :, None]
        return (hid.float() + delta @ dec).to(hid.dtype)
    return sd.editing({li: edit})


@app.post("/api/steer")
async def steer(req: Request):
    return steer_impl(await req.json())


@app.get("/sheets/{name}")
def sheet(name: str):
    p = os.path.join("sheets", os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "max-age=3600"})


ASSIST = None


def assistant():
    global ASSIST
    if ASSIST is None:
        from explorer.assistant import Assistant
        ASSIST = Assistant(sys.modules[__name__])
    return ASSIST


@app.post("/api/chat")
def chat(body: dict):
    """Sidebar assistant turn: {sid?, text, context?} -> {sid, events:[...]} (runs the whole tool loop)."""
    try:
        return assistant().chat(body.get("sid"), body["text"], body.get("context"))
    except Exception as e:  # surface API errors to the panel instead of a bare 500
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/chat/reset")
def chat_reset(body: dict):
    assistant().reset(body.get("sid"))
    return {"ok": True}


# ---- SAE features, cross-layer connections and the all-layers model map ----------------------------------------
CONN = {}
MAPS = {}
GG_ATTR = "the Golden Gate Bridge"


def conn():
    """explorer/cache/conn_flux_sae.pt from explorer/build_connections.py (None until it has been built)."""
    p = "explorer/cache/conn_flux_sae.pt"
    if not os.path.exists(p):
        return None
    if CONN.get("mtime") != os.path.getmtime(p):
        import torch
        CONN.clear()
        CONN.update(d=torch.load(p, map_location="cpu"), meta=json.load(open("sae/flux/meta.json")), mtime=os.path.getmtime(p))
    return CONN["d"]


def feature_desc(layer, f):
    """What the corpus says about one feature: Golden Gate z, density, the prompts it fires hardest on, top attributes."""
    d = conn()
    if d is None or f"gg_z_L{layer}" not in d:
        return {}
    items, attrs = CONN["meta"]["items"], d["attributes"]
    f = int(f)
    return {"gg_z": round(float(d[f"gg_z_L{layer}"][f]), 2), "density": round(float(d[f"density_L{layer}"][f]), 4),
            "fires_on": [items[int(i)]["prompt"] for i in d[f"top_imgs_L{layer}"][f]],
            "lens": [[attrs[int(i)], round(float(v), 2)] for i, v in zip(d[f"lens_idx_L{layer}"][f], d[f"lens_val_L{layer}"][f])]}


def feature_links(layer, f):
    """Lyra-style strongest links up and down: direct-path cosine through the residual stream, and co-activation."""
    d = conn()
    out = {"direct": [], "coact": []}
    if d is None:
        return out
    f = int(f)
    for key in d:
        parts = key.split("_")
        if len(parts) != 5 or parts[0] not in out or parts[4] != "idx":
            continue
        kind, la, lb, direction = parts[0], int(parts[1]), int(parts[2]), parts[3]
        if direction == "down" and la == layer:
            other = lb
        elif direction == "up" and lb == layer:
            other = la
        else:
            continue
        idx, val = d[key][f], d[key[:-3] + "val"][f]
        for i, v in zip(idx.tolist(), val.tolist()):
            desc = feature_desc(other, i)
            out[kind].append({"layer": other, "feature": int(i), "val": round(float(v), 3), "dir": "down" if other > layer else "up",
                              "gg_z": desc.get("gg_z"), "label": desc.get("lens", [["", 0]])[0][0]})
    for k in out:
        out[k].sort(key=lambda e: -abs(e["val"]))
    return out


def feature_runs(layer, f):
    """SAE runs that clamp this feature, directly or as a member of a feature family."""
    f = int(f)
    out = []
    for m in runs():
        if not m["sae"] or m.get("model", "flux") != "flux":
            continue
        groups = m.get("groups") or {}
        layers = m.get("layers") or [m["layer"]]
        if m["layer"] == layer and str(f) in (m.get("feature_scale") or {}):
            out.append({"run": m["run"], "as": "clamp", "neuron": f, "mask": m.get("mask", ""), "layers": layers})
        else:
            for lead, members in groups.items():
                if len(layers) == len(members) and layer in layers and members[layers.index(layer)] == f:
                    out.append({"run": m["run"], "as": "family", "neuron": int(lead), "mask": m.get("mask", ""), "layers": layers, "family": members})
    return out


@app.get("/api/sae_units")
def sae_units(run: str):
    """Per-feature summary of one SAE clamp run: concept gain, scene preservation and Golden Gate lens per dose."""
    d = load_run(run)
    meta = d["meta"]
    if not is_sae(meta):
        raise HTTPException(400, "not an SAE run")
    gi = meta["attributes"].index(GG_ATTR) if GG_ATTR in meta["attributes"] else None
    base = {r["prompt"]: r["cos_concept"] for r in d["rows"] if r["kind"] == "baseline"}
    units = []
    for (layer, kind, n), rows in d["by_unit"].items():
        if kind == "baseline":
            continue
        per = {}
        for r in rows:
            e = per.setdefault(r["dose"], {"dcos": [], "cos_base": [], "gg": []})
            e["dcos"].append(r["cos_concept"] - base.get(r["prompt"], 0.0))
            e["cos_base"].append(r.get("cos_base", 1.0))
            if gi is not None and r.get("d_attr"):
                e["gg"].append(r["d_attr"][gi])
        doses = {str(k): {kk: round(float(np.mean(v)), 3) for kk, v in e.items() if v} for k, e in sorted(per.items())}
        ok = [(k, v) for k, v in doses.items() if v.get("cos_base", 1) >= 0.8] or list(doses.items())
        bk, bv = max(ok, key=lambda kv: kv[1]["dcos"])
        units.append({"kind": kind, "neuron": n, "layer": layer, "best_dose": float(bk), "dcos": bv["dcos"], "cos_base": bv.get("cos_base"),
                      "gg_z": bv.get("gg"), "doses": doses, "family": (meta.get("groups") or {}).get(str(n)),
                      "desc": feature_desc(layer, n) if kind == "clamp" else {}})
    units.sort(key=lambda u: (u["kind"] != "clamp", -u["dcos"]))
    return {"units": units, "meta": meta}


@app.get("/api/feature")
def feature(layer: int, f: int):
    """One SAE feature: corpus description, Golden Gate ranking, runs that clamp it, and its cross-layer links."""
    rank = None
    for p in (f"sae/flux2/find_L{layer}_gg.json", f"sae/flux/find_L{layer}_gg.json"):
        if os.path.exists(p):
            top = json.load(open(p))["top"]
            for i, e in enumerate(top):
                if e["feature"] == f:
                    rank = {"rank": i + 1, "of": len(top), **{k: e[k] for k in ("auroc", "z", "density", "prec50", "mean_tag", "mean_rest", "mean_bridge", "mean_landmark") if k in e},
                            "lens": e.get("lens"), "top_prompts": e.get("top_prompts", [])[:6]}
                    break
            break
    names = json.load(open("explorer/names.json")) if os.path.exists("explorer/names.json") else {}
    return {"layer": layer, "feature": f, "desc": feature_desc(layer, f), "gg_rank": rank, "runs": feature_runs(layer, f),
            "links": feature_links(layer, f), "name": names.get(f"L{layer}F{f}", ""), "has_conn": conn() is not None}


FLUX_LAYERS = [{"layer": i, "name": f"d{i}", "inner": 12288, "dim": 3072, "kind": "double"} for i in range(19)] + \
              [{"layer": 19 + i, "name": f"s{i}", "inner": 12288, "dim": 3072, "kind": "single"} for i in range(38)]


def _units_mass(run):
    p = f"runs/{run}.units.jsonl"
    out = {}
    for l in open(p):
        u = json.loads(l)
        if u["kind"] == "neuron":
            out[u["neuron"]] = u["mass"]
    return out


@app.get("/api/model_map")
def model_map(model: str):
    """Every layer of a model for the all-layers map: size, which runs swept it, and a per-unit value to colour it
    (lens mass for swept neurons, Golden Gate z for SAE feature layers; 0 where nothing was measured)."""
    if model in MAPS:
        return MAPS[model]
    if model == "sdxl":
        layers = [{"layer": s["layer"], "name": s["name"], "inner": s["inner"], "dim": s["dim"], "kind": s["name"][0]} for s in json.load(open("stats/sdxl-turbo.json"))]
    elif model == "flux3":
        layers = [{"layer": s["layer"], "name": s["name"], "inner": s["inner"], "dim": s["dim"], "kind": "mode" if s["name"][0] == "m" else "single"} for s in json.load(open("stats-flux3/flux3.json"))]
    else:
        layers = [dict(l) for l in FLUX_LAYERS]
    rs = [m for m in runs() if run_model(m, m["run"]) == model]
    d = conn() if model == "flux" else None
    for L in layers:
        cover = [m for m in rs if m["layer"] == L["layer"] and not m["sae"] and os.path.exists(f"runs/{m['run']}.units.jsonl")]
        cover.sort(key=lambda m: -(m.get("n_neurons") or 0))
        L["runs"] = [m["run"] for m in cover]
        L["sae_runs"] = [m["run"] for m in rs if m["sae"] and m["layer"] == L["layer"]]
        L["values"], L["n_swept"] = None, 0
        for m in cover:  # the first run that actually ranked neurons at this layer (vector/probe runs rank none)
            mass = _units_mass(m["run"])
            if mass:
                L["values"] = [round(mass.get(i, 0.0), 1) for i in range(L["inner"])]
                L["n_swept"], L["sweep_run"] = len(mass), m["run"]
                break
        if d is not None and f"gg_z_L{L['layer']}" in d:
            L["sae"] = {"features": len(d[f"gg_z_L{L['layer']}"]), "gg_z": [round(x, 1) for x in d[f"gg_z_L{L['layer']}"].float().tolist()],
                        "direct": int(L["layer"]) in d["layers_direct"]}
    MAPS[model] = {"model": model, "layers": layers}
    return MAPS[model]


# ---- the generate view: one prompt, one unit, every denoising step, baseline next to steered ----------------------
def _jpeg(t, size=None, q=88):
    """[3,H,W] float tensor in [0,1] -> data URL."""
    from PIL import Image
    im = Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/presets")
def presets():
    p = "explorer/presets.json"
    ps = json.load(open(p)) if os.path.exists(p) else []
    for x in ps:
        x["available"] = x["model"] in LIVE["models"]
    return {"presets": ps, "live": list(LIVE["models"])}


@app.post("/api/generate")
async def generate(req: Request):
    """{model, prompt, seed, steps, size, unit:{layer, kind, neuron, dose, mask}, full_vae} ->
    {steps:[{i, sigma, base:{xt,x0}, steer:{xt,x0}, act:{base, steer, grid, max}}], final:{base, steer}, attr, ms}
    Runs baseline (dose 0) and steered as one batch of two through the same noise, tracing every scheduler step and
    the unit's activation map at every step."""
    import time
    b = await req.json()
    model, sd = live_model(b.get("model"))
    import torch
    from sd_explorer.metrics import pixel_metrics
    from sd_explorer.trace import Trace
    lens = LIVE["lens"]
    u = b.get("unit") or {}
    extra = [x for x in (b.get("units") or []) if x]  # several neurons at once (e.g. the two-neuron Golden Gate)
    if extra and not u:
        u = dict(extra[0])
    li, n, kind, dose = int(u.get("layer", 0)), int(u.get("neuron", 0)), u.get("kind", "neuron"), float(u.get("dose", 0))
    seed, steps = int(b.get("seed", 1001)), int(b.get("steps") or (4 if model != "flux3" else 16))
    size = int(b.get("size", 512))
    t0 = time.time()
    with LIVE["lock"]:
        old = (sd.h, sd.w, getattr(sd, "t_img", None))
        try:
            sd.h = sd.w = size
            if hasattr(sd, "t_img"):
                sd.t_img = (size // 16) * (size // 16)
            prompts, seeds = [b["prompt"]] * 2, [seed] * 2
            unit_desc = None
            if u and kind == "clamp":
                if model != "flux":
                    raise HTTPException(400, "feature clamping needs the FLUX model (--live flux)")
                unit_desc = {"kind": "clamp", "layer": li, "neuron": n, "sae": sae_for(sd, li)}
                ctx = clamp_context(sd, li, n, [0.0, dose], u.get("mask", ""))
                pushes = None
            elif u:
                unit_desc = {"kind": "neuron", "layer": li, "neuron": n}
                all_units = extra if extra else [u]
                pushes = {}
                for x in all_units:  # row 0 stays the baseline, row 1 carries every push
                    xl = int(x["layer"])
                    pushes.setdefault(xl, torch.zeros(2, sd.dim[xl], device=sd.device))
                    pushes[xl][1] += sd.push_for(xl, x.get("kind", "neuron"), int(x["neuron"]), float(x.get("dose", 0)), push_unit(model, xl))
                import contextlib
                ctx = contextlib.nullcontext()
            else:
                pushes, ctx = None, __import__("contextlib").nullcontext()
            with ctx, Trace(sd, unit_desc) as tr:
                img = sd.generate(prompts, seeds, pushes if pushes else None, full_vae=bool(b.get("full_vae", True)), steps=steps)
            frames = tr.frames()
            acts = tr.acts
            pm = pixel_metrics(img)
            emb = lens.embed_images(img)
            attr = lens.attribute_scores(emb)
        finally:
            sd.h, sd.w = old[0], old[1]
            if old[2] is not None:
                sd.t_img = old[2]
    thumb = int(b.get("thumb", 192))  # step frames are small on screen; keep the payload light over a tunnel
    out_steps = []
    for k, fr in enumerate(frames):
        e = {"i": fr["i"], "sigma": fr["sigma"], "t": fr["t"],
             "base": {"xt": _jpeg(fr["xt"][0], thumb, 78), "x0": _jpeg(fr["x0"][0], thumb, 80)},
             "steer": {"xt": _jpeg(fr["xt"][1], thumb, 78), "x0": _jpeg(fr["x0"][1], thumb, 80)}}
        if acts and k < len(acts):
            a = acts[k]  # [2, G, G]; sent as fractions of the step's max, 2 decimals
            m = float(a.max()) or 1.0
            e["act"] = {"grid": int(a.shape[-1]), "max": round(m, 3),
                        "base": [round(x / m, 2) for x in a[0].flatten().tolist()], "steer": [round(x / m, 2) for x in a[1].flatten().tolist()]}
        out_steps.append(e)
    d_attr = (attr[1] - attr[0]).cpu().numpy()
    top = np.argsort(-d_attr)[:6]
    return {"model": model, "steps": out_steps, "final": {"base": _jpeg(img[0], None, 84), "steer": _jpeg(img[1], None, 84)},
            "metrics": {k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in pm.items()},
            "d_attr": [[lens.attributes[j], round(float(d_attr[j]), 2)] for j in top],
            "cos_base": round(float(torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0)), 4),
            "ms": int((time.time() - t0) * 1000), "unit": u, "seed": seed, "size": size, "n_steps": steps}


app.mount("/", StaticFiles(directory="explorer/static", html=True), name="static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--live", nargs="?", const="sdxl", default=None,
                    help="load image models for live steering: sdxl (default), flux, flux3, or a comma list such as sdxl,flux")
    ap.add_argument("--model", default=None, help="deprecated alias for --live <model>")
    a = ap.parse_args()
    live = a.live or a.model
    if live:
        from sd_explorer import load_model
        from sd_explorer.metrics import Lens
        for name in [m.strip() for m in (a.model or live).split(",") if m.strip()]:
            LIVE["models"][name] = load_model(name)
            print(f"live model loaded: {name}")
        LIVE["lens"] = Lens()
        LIVE["model"] = next(iter(LIVE["models"]))
        LIVE["sd"] = LIVE["models"][LIVE["model"]]
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
