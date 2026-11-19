"""The sidebar assistant: Claude drives the explorer's own operations (rank search, contrast search, SAE feature
search, unit inspection with thumbnails, live steering) to find the unit that does what you describe.

Tools are plain functions over the same files the explorer serves. The chat loop is a manual Messages API loop so
tool results can carry images (thumbnails and sheets) that Claude looks at before answering. Sessions live in
memory on the server; the browser only holds a session id.

Needs ANTHROPIC_API_KEY in the server's environment (or an `ant auth login` profile).
"""
import base64
import glob
import io
import json
import os
import threading
import uuid

import numpy as np

MODEL = os.environ.get("EXPLORER_ASSISTANT_MODEL", "claude-opus-5-5")
MAX_STEPS = 24
THUMB_MAX = 10          # thumbnails per tool result (each ~160 px jpg)
SHEET_MAX_PX = 1100     # longest side when a sheet is shown to Claude

SYSTEM = """You are the assistant inside "sd neuron explorer", a steering-first interpretability tool for diffusion
models (SDXL-Turbo UNet feed-forward neurons, and FLUX/Shuttle-3 DiT feed-forward neurons and sparse-autoencoder
features). The user asks for the unit that *does* something ("makes images black and white", "adds a Golden Gate
Bridge", "turns things into pixel art"). Your job: find it with the tools, look at the images, and answer with
specific unit ids the user can open.

How the data works
- A "run" is a sweep of one layer: every unit (neuron, or a random direction as control) was pushed at fixed doses
  on a few probe prompts. Each unit has: `mass` (summed |z| of its top CLIP attribute shifts vs the random null:
  how strongly it does *anything*), `null_pct` (share of random controls it beats), `lens` (CLIP attributes whose
  score moved, with z), pixel metric z-scores (sat, lum, colorful, contrast) and per-dose `cos_base` (cosine
  between steered and unsteered image: < 0.75 means the push wrecked the image, a "destroyer", not a dial).
- A good dial keeps cos_base high while moving the attribute you care about. A subject neuron (paints a thing)
  is rare; style, texture, colour and brightness dials are common. Destroyers often get high z on content
  attributes because CLIP misreads textures, so always check cos_base and look at the thumbnails.
- "Contrast search" asks which neurons *read* a concept (activation difference between paired prompts); a reader
  is not necessarily a writer: test by steering.
- SAE features (FLUX only): sparse-autoencoder features ranked by how selectively they fire on tagged prompts
  (AUROC, mean on tag vs rest, vs other bridges/landmarks). Feature 13287 at L24 (s5) is the Golden Gate
  feature; clamping it writes the bridge. Those results are in the find_* files and the sheets.
- Live steering needs the model loaded (`--live`); if it is not, say so and rely on the sweep thumbnails.

Method
1. list_runs to see what exists for the chosen model; pick the run(s) whose layer could hold the effect.
2. search_units with the attribute words (several phrasings), sorted by the attribute z, keeping only units with
   cos_base above ~0.8 unless the user wants destroyers.
3. inspect_unit on the top 2-3 candidates: read the per-dose metrics and LOOK at the thumbnails before judging.
4. If live steering is available, steer the best candidate on the user's own prompt to confirm it writes.
5. focus_unit so the explorer opens it, then answer: unit id, layer, what it does at which dose sign, how clean
   (cos_base), and the runner-up. Be concrete and short; mention when nothing qualifies.
"""

TOOLS = [
    {"name": "list_runs", "description": "List the ranked sweep runs (model, layer, layer name, neuron count, doses, "
     "prompts) and the SAE feature rankings and sheets that exist. Call first.",
     "input_schema": {"type": "object", "properties": {"model": {"type": "string", "description": "sdxl | flux | all"}},
                      "required": [], "additionalProperties": False}},
    {"name": "search_units", "description": "Search one run's ranked units by CLIP attribute words. Returns the units "
     "whose lens contains the words, sorted by that attribute's z (or by mass when no words), with cos_base at the "
     "best dose so destroyers can be excluded.",
     "input_schema": {"type": "object", "properties": {
         "run": {"type": "string"}, "query": {"type": "string", "description": "attribute words, e.g. 'grayscale' or "
         "'black and white'; several alternatives separated by |"},
         "min_cos": {"type": "number", "description": "drop units whose cos_base at the best dose is below this (default 0.75)"},
         "top": {"type": "integer"}, "include_random": {"type": "boolean"}},
         "required": ["run"], "additionalProperties": False}},
    {"name": "inspect_unit", "description": "Per-prompt, per-dose metrics and the thumbnails (as images) of one unit "
     "in a run. Look at the images before judging a unit.",
     "input_schema": {"type": "object", "properties": {"run": {"type": "string"}, "neuron": {"type": "integer"},
                                                        "kind": {"type": "string", "description": "neuron | random"}},
                      "required": ["run", "neuron"], "additionalProperties": False}},
    {"name": "contrast_search", "description": "Which neurons READ a concept: per-layer activation difference between "
     "paired prompts (plain vs concept). Available concepts come from list_runs.",
     "input_schema": {"type": "object", "properties": {"concept": {"type": "string"}, "top": {"type": "integer"}},
                      "required": ["concept"], "additionalProperties": False}},
    {"name": "sae_features", "description": "FLUX sparse-autoencoder features ranked for a tag (e.g. gg = Golden Gate) at "
     "a layer: AUROC, means on tag/rest/other bridges/landmarks, lens, cos to the concept text.",
     "input_schema": {"type": "object", "properties": {"layer": {"type": "integer"}, "tag": {"type": "string"},
                                                        "top": {"type": "integer"}},
                      "required": ["layer"], "additionalProperties": False}},
    {"name": "feature_info", "description": "One FLUX SAE feature at a layer (6, 12, 21, 24 or 26): what the corpus says it "
     "reads (Golden Gate z, density, prompts it fires hardest on, CLIP attributes), its Golden Gate rank, the clamp runs that "
     "steer it, and its strongest cross-layer links (direct-path cosine through the residual stream, co-activation). Also "
     "opens it in the explorer.",
     "input_schema": {"type": "object", "properties": {"layer": {"type": "integer"}, "feature": {"type": "integer"},
                                                        "links": {"type": "integer", "description": "links per kind to return (default 8)"}},
                      "required": ["layer", "feature"], "additionalProperties": False}},
    {"name": "view_sheet", "description": "Show a contact sheet from sheets/ (name from list_runs) as an image.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                      "additionalProperties": False}},
    {"name": "steer", "description": "Live steering (only when the server runs with --live): push one unit at the "
     "given doses on a prompt and return the images with their CLIP attribute shifts.",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "layer": {"type": "integer"},
                                                        "neuron": {"type": "integer"}, "kind": {"type": "string"},
                                                        "doses": {"type": "array", "items": {"type": "number"}},
                                                        "seed": {"type": "integer"}},
                      "required": ["prompt", "layer", "neuron"], "additionalProperties": False}},
    {"name": "focus_unit", "description": "Open a unit in the explorer's focus panel for the user.",
     "input_schema": {"type": "object", "properties": {"run": {"type": "string"}, "neuron": {"type": "integer"},
                                                        "kind": {"type": "string"}},
                      "required": ["run", "neuron"], "additionalProperties": False}},
    {"name": "name_unit", "description": "Save a short name for a neuron (shows in the explorer).",
     "input_schema": {"type": "object", "properties": {"layer": {"type": "integer"}, "neuron": {"type": "integer"},
                                                        "name": {"type": "string"}},
                      "required": ["layer", "neuron", "name"], "additionalProperties": False}},
]


def run_model(meta, run):
    m = meta.get("model")
    if m:
        return m
    return "flux" if run.startswith("flux") else "sdxl"


def _img_block(path, max_px=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max_px and max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


class Assistant:
    def __init__(self, server):
        self.s = server  # the explorer server module: runs(), load_run(), LIVE, steer_impl()
        self.sessions = {}
        self.lock = threading.Lock()
        self._client = None

    # ---- tools ---------------------------------------------------------------------------------------------------
    def t_list_runs(self, model="all"):
        out = []
        for m in self.s.runs():
            if not m.get("ranked"):
                continue
            mm = run_model(m, m["run"])
            if model not in ("all", None, "") and mm != model:
                continue
            out.append({"run": m["run"], "model": mm, "layer": m.get("layer"), "layer_name": m.get("name"),
                        "n_neurons": m.get("n_neurons"), "doses": m.get("doses"), "dose_unit": m.get("dose_unit"),
                        "prompts": m.get("prompts"), "probe": bool(m.get("probe")), "note": m.get("note")})
        contrasts = [os.path.basename(p)[len("contrast_"):-5] for p in sorted(glob.glob("stats/contrast_*.json"))]
        finds = sorted(os.path.basename(p) for p in glob.glob("sae/*/find_*.json"))
        sheets = sorted(os.path.basename(p) for p in glob.glob("sheets/*.jpg"))
        return {"runs": out, "contrast_concepts": contrasts, "sae_rankings": finds, "sheets": sheets,
                "live_steering": sorted(self.s.LIVE["models"])}

    def t_search_units(self, run, query="", min_cos=0.75, top=20, include_random=False):
        d = self.s.load_run(run)
        words = [w.strip().lower() for w in (query or "").split("|") if w.strip()]
        rows = []
        for u in d["units"]:
            if u["kind"] == "random" and not include_random:
                continue
            if not u.get("dose"):
                continue
            best = u["dose"].get(f"{u['best_dose']:+g}") or next(iter(u["dose"].values()))
            # attribute z for the query: best over doses and matching attributes
            zq, zdose, zattr = 0.0, None, None
            for dk, v in u["dose"].items():
                for name, z, raw in v["lens"]:
                    if any(w in name.lower() for w in words) and z > zq:
                        zq, zdose, zattr = z, dk, name
            if words and zattr is None:
                continue
            cos_q = u["dose"][zdose]["cos_base"] if zdose else best["cos_base"]
            if cos_q < min_cos:
                continue
            rows.append({"neuron": u["neuron"], "kind": u["kind"], "layer": u["layer"], "mass": u["mass"],
                         "null_pct": u["null_pct"], "best_dose": u["best_dose"], "cos_base_best": round(best["cos_base"], 3),
                         "match_attr": zattr, "match_z": round(zq, 2), "match_dose": zdose, "cos_base_at_match": round(cos_q, 3),
                         "z_sat": u.get("z_d_sat"), "z_lum": u.get("z_d_lum"), "z_colorful": u.get("z_d_colorful"),
                         "lens": [[a, z] for a, z, _ in best["lens"][:5]]})
        rows.sort(key=lambda r: -(r["match_z"] if words else r["mass"]))
        return {"run": run, "layer": d["meta"].get("layer"), "n_matched": len(rows), "units": rows[:top]}

    def t_inspect_unit(self, run, neuron, kind="neuron"):
        d = self.s.load_run(run)
        layer = d["meta"]["layer"]
        rows = d["by_unit"].get((layer, kind, int(neuron)))
        if not rows:
            return {"error": f"no unit {kind} {neuron} in run {run}"}
        rec = next((u for u in d["units"] if u["kind"] == kind and u["neuron"] == int(neuron)), None)
        table = [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in r
                  if k in ("prompt", "dose", "d_sat", "d_lum", "d_colorful", "d_contrast", "mse", "cos_base", "cos_prompt", "thumb")}
                 for r in rows]
        base = next((v for k, v in d["by_unit"].items() if k[1] == "baseline"), [])
        content = [{"type": "text", "text": json.dumps({"unit": {k: v for k, v in (rec or {}).items() if k != "dose"},
                                                          "per_dose": rec.get("dose") if rec else None,
                                                          "rows": table}, default=float)}]
        shown = []
        for r in base[:4]:
            p = f"thumbs/{run}/{r['thumb']}"
            if os.path.exists(p):
                shown.append(f"baseline {r['prompt']}"); content.append(_img_block(p))
        for r in rows[:THUMB_MAX]:
            p = f"thumbs/{run}/{r['thumb']}"
            if os.path.exists(p):
                shown.append(f"{r['prompt']} dose {r['dose']:+g}"); content.append(_img_block(p))
        content.insert(1, {"type": "text", "text": "images in order: " + "; ".join(shown) if shown else "no thumbnails on disk for this unit"})
        return content

    def t_contrast_search(self, concept, top=20):
        p = f"stats/contrast_{concept}.json"
        if not os.path.exists(p):
            return {"error": f"no contrast file for '{concept}'"}
        d = json.load(open(p))
        return {"concept": concept, "templates": d["templates"], "n": d["n"], "pixels": d["pixels"],
                "layers": d["layers"], "top": d["top"][:top]}

    def t_sae_features(self, layer, tag="gg", top=15):
        p = f"sae/flux/find_L{layer}_{tag}.json"
        if not os.path.exists(p):
            return {"error": f"no SAE ranking for L{layer} tag {tag}", "available": sorted(os.path.basename(x) for x in glob.glob("sae/*/find_*.json"))}
        d = json.load(open(p))
        feats = [{k: v for k, v in f.items() if k not in ("top_imgs",)} for f in d["top"][:top]]
        for f in feats:
            f["top_prompts"] = f.get("top_prompts", [])[:6]
        return {"layer": d["layer"], "tag": d["tag"], "n_pos": d["n_pos"], "n_neg": d["n_neg"], "sae": d.get("sae_info"), "features": feats,
                "sheets": sorted(os.path.basename(x) for x in glob.glob(f"sheets/flux_sae_L{layer}_*.jpg"))}

    def t_view_sheet(self, name):
        p = os.path.join("sheets", os.path.basename(name))
        if not os.path.exists(p):
            return {"error": "no such sheet", "available": sorted(os.path.basename(x) for x in glob.glob("sheets/*.jpg"))}
        return [{"type": "text", "text": f"sheet {name}"}, _img_block(p, SHEET_MAX_PX)]

    def t_steer(self, prompt, layer, neuron, kind="neuron", doses=None, seed=1001):
        if not self.s.LIVE["models"]:
            return {"error": "live steering is off: the server was started without --live"}
        out = self.s.steer_impl({"prompt": prompt, "layer": layer, "neuron": neuron, "kind": kind,
                                 "doses": doses or [-1, 0, 1], "seed": seed, "size": 256})
        content = [{"type": "text", "text": json.dumps([{k: v for k, v in im.items() if k != "img"} for im in out["images"]])}]
        for im in out["images"]:
            data = im["img"].split(",", 1)[1]
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        return content, out

    def t_focus_unit(self, run, neuron, kind="neuron"):
        return {"ok": True, "ui": {"focus": {"run": run, "neuron": int(neuron), "kind": kind}}}

    def t_feature_info(self, layer, feature, links=8):
        d = self.s.feature(int(layer), int(feature))
        d["links"] = {k: v[:int(links)] for k, v in d["links"].items()}
        d["ui"] = {"feature": {"layer": int(layer), "feature": int(feature)}}
        return d

    def t_name_unit(self, layer, neuron, name):
        p = "explorer/names.json"
        names = json.load(open(p)) if os.path.exists(p) else {}
        names[f"L{layer}N{neuron}"] = name
        json.dump(names, open(p, "w"), indent=1)
        return {"ok": True, "key": f"L{layer}N{neuron}", "name": name}

    def call(self, name, args):
        """Returns (tool_result_content, ui_events)."""
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return f"unknown tool {name}", []
        try:
            res = fn(**args)
        except TypeError as e:
            return f"bad arguments: {e}", []
        except Exception as e:  # report to the model, keep the loop alive
            return f"error: {type(e).__name__}: {e}", []
        ui = []
        if name == "steer" and isinstance(res, tuple):
            res, out = res
            ui.append({"type": "images", "images": [{"src": im["img"], "label": f"dose {im['dose']:+g}"} for im in out["images"]]})
        elif name == "inspect_unit" and isinstance(res, list):
            run, n = args["run"], args["neuron"]
            d = self.s.load_run(run)
            rows = d["by_unit"].get((d["meta"]["layer"], args.get("kind", "neuron"), int(n))) or []
            ui.append({"type": "images", "images": [{"src": f"/thumbs/{run}/{r['thumb']}", "label": f"{r['prompt']} {r['dose']:+g}"} for r in rows[:THUMB_MAX]]})
        elif name == "view_sheet" and isinstance(res, list):
            ui.append({"type": "images", "images": [{"src": f"/sheets/{os.path.basename(args['name'])}", "label": args["name"], "wide": True}]})
        elif isinstance(res, dict) and "ui" in res:
            u = res.pop("ui")
            ui.append({"type": "feature", **u["feature"]} if "feature" in u else {"type": "focus", **u["focus"]})
        if isinstance(res, (dict, list)) and not (isinstance(res, list) and res and isinstance(res[0], dict) and res[0].get("type")):
            res = json.dumps(res, default=float)
        return res, ui

    # ---- chat ----------------------------------------------------------------------------------------------------
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _create(self, messages):
        kw = dict(model=MODEL, max_tokens=8000, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                  tools=TOOLS, messages=messages, output_config={"effort": "medium"})
        try:  # server-side refusal fallback, as recommended for claude-opus-5-5
            return self.client().beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        except TypeError:
            return self.client().messages.create(**kw)

    def chat(self, sid, text, context=None):
        """One user turn: runs the tool loop to completion and returns the events to render."""
        with self.lock:
            sid = sid or uuid.uuid4().hex
            messages = self.sessions.setdefault(sid, [])
            user = text if not context else f"[explorer state: {json.dumps(context)}]\n\n{text}"
            messages.append({"role": "user", "content": user})
            events = []
            for _ in range(MAX_STEPS):
                resp = self._create(messages)
                messages.append({"role": "assistant", "content": resp.content})
                for b in resp.content:
                    if b.type == "text" and b.text.strip():
                        events.append({"type": "text", "text": b.text})
                if resp.stop_reason == "refusal":
                    events.append({"type": "text", "text": "(the model declined this request)"})
                    break
                calls = [b for b in resp.content if b.type == "tool_use"]
                if not calls:
                    break
                results = []
                for c in calls:
                    args = c.input if isinstance(c.input, dict) else json.loads(c.input)
                    events.append({"type": "tool", "name": c.name, "args": args})
                    content, ui = self.call(c.name, args)
                    events.extend(ui)
                    results.append({"type": "tool_result", "tool_use_id": c.id, "content": content,
                                    **({"is_error": True} if isinstance(content, str) and content.startswith(("error", "unknown", "bad ")) else {})})
                messages.append({"role": "user", "content": results})
            else:
                events.append({"type": "text", "text": "(stopped after the step limit)"})
            return {"sid": sid, "events": events, "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                                                            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0)}}

    def reset(self, sid):
        self.sessions.pop(sid, None)
