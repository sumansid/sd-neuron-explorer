// SD neuron explorer front-end: an all-layers model map (one sheet per layer, Lyra-style), a face-on map of one
// layer's neurons or SAE features, a ranked list, the contrast search, and a focus panel with every steered image,
// the CLIP lens, cross-layer connections for SAE features, and live steering.
const $ = (s) => document.querySelector(s);
const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(r.statusText); return r.json(); });
const S = { index: null, run: null, units: [], meta: null, byKey: {}, focus: null, hover: null, names: {}, sae: false, links: null, featureInfo: null };
const MM = { model: null, data: null, sheets: {}, hover: -1, pad: 14, step: 0, cw: 0, ch: 0, top: 36, shear: 0.22 };

function fmt(x, d = 2) { return x == null ? "" : (+x).toFixed(d); }
function kchar(kind) { return kind === "clamp_random" ? "r" : kind === "clamp" ? "f" : kind[0]; }
function key(u) { return kchar(u.kind) + u.neuron + (S.meta && u.layer !== S.meta.layer ? "@" + u.layer : ""); }
function nameKey(u) { return `L${u.layer}${u.kind === "clamp" || u.kind === "feature" ? "F" : "N"}${u.neuron}`; }
function isFeat(u) { return u && (u.kind === "clamp" || u.kind === "feature"); }

// ---- palette: diverging (cyan - dark - pink) for signed z, sequential for mass -------------------------------
function colorFor(u, metric) {
  const v = u[metric];
  if (v == null) return [40, 40, 40];
  if (metric === "mass") { const t = Math.min(1, v / 25); return [40 + 215 * t, 40 + 82 * t, 40 + 142 * t]; }
  const t = Math.max(-1, Math.min(1, v / 8));
  return t >= 0 ? [40 + 215 * t, 40 + 82 * t, 40 + 142 * t] : [40 + 82 * -t, 40 + 175 * -t, 40 + 215 * -t];
}
function ggColor(z) { const t = Math.max(0, Math.min(1, z / 2.5)); return [40 + 215 * t, 40 + 82 * t, 40 + 142 * t]; }

// ---- model map: every layer as an oblique sheet -----------------------------------------------------------------
const mmap = $("#mmap"), mctx = mmap.getContext("2d");
function sheetDims(n, cw, ch) { const cols = Math.max(8, Math.round(Math.sqrt(n * cw / ch))); return [cols, Math.ceil(n / cols)]; }
function buildSheet(L) {
  // offscreen canvas, one pixel per unit: lens mass for swept neurons, Golden Gate z for SAE features, grey otherwise
  const sae = L.sae && !L.values, n = sae ? L.sae.features : L.inner;
  const [cols, rows] = sheetDims(n, MM.cw, MM.ch);
  const c = document.createElement("canvas"); c.width = cols; c.height = rows;
  const g = c.getContext("2d"), im = g.createImageData(cols, rows), px = im.data;
  for (let i = 0; i < cols * rows; i++) {
    let r = 26, gg = 26, b = 26;
    if (i < n) {
      if (sae) { const z = L.sae.gg_z[i]; if (z > 0.4) [r, gg, b] = ggColor(z); else { r = gg = b = 46; } }
      else if (L.values) { const v = L.values[i]; if (v > 0) { const t = Math.min(1, v / 25); r = 40 + 215 * t; gg = 40 + 82 * t; b = 40 + 142 * t; } else { r = gg = b = 46; } }
      else { r = gg = b = 34; }
    }
    px[i * 4] = r; px[i * 4 + 1] = gg; px[i * 4 + 2] = b; px[i * 4 + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return { canvas: c, cols, rows, n, sae };
}
function layerPos(i) { return MM.pad + i * MM.step; }
function isSwept(L) { return !!(L.n_swept || L.sae || L.sae_runs.length); }
function unitPos(i, idx) {
  // canvas position of unit idx on layer i's sheet (sheared card)
  const sh = MM.sheets[i]; if (!sh) return null;
  const cx = ((idx % sh.cols) + 0.5) / sh.cols * MM.cw, cy = (Math.floor(idx / sh.cols) + 0.5) / sh.rows * MM.ch;
  return [layerPos(i) + cx, MM.top + cy - MM.shear * cx];
}
function drawCard(i, front) {
  const L = MM.data.layers[i], sh = MM.sheets[i], x0 = layerPos(i);
  mctx.save();
  mctx.setTransform(1, -MM.shear, 0, 1, x0, MM.top);
  mctx.imageSmoothingEnabled = false;
  mctx.globalAlpha = front ? 1 : 0.92;
  mctx.drawImage(sh.canvas, 0, 0, MM.cw, MM.ch);
  const focused = S.focus && S.meta && S.meta.layer === L.layer && MM.data.model === currentModel();
  mctx.strokeStyle = front ? "#fff" : focused ? "#ff7ab6" : isSwept(L) ? "#555" : "#2a2a2a";
  mctx.lineWidth = front || focused ? 1.5 : 1;
  mctx.strokeRect(0, 0, MM.cw, MM.ch);
  if (MM.pulse && MM.pulse.layer === L.layer && MM.pulse.model === MM.data.model) {  // the generate view's unit, glowing with its activation
    mctx.shadowColor = "#ff7ab6"; mctx.shadowBlur = 6 + 30 * MM.pulse.strength; mctx.strokeStyle = "#ff7ab6"; mctx.lineWidth = 2;
    mctx.globalAlpha = 0.5 + 0.5 * MM.pulse.strength; mctx.strokeRect(0, 0, MM.cw, MM.ch); mctx.shadowBlur = 0;
  }
  mctx.restore();
}
function drawModelMap() {
  if (!MM.data || $("#model-wrap").classList.contains("collapsed")) return;
  const Ls = MM.data.layers, N = Ls.length;
  mmap.width = mmap.parentElement.clientWidth - 2; const W = mmap.width, H = mmap.height;
  MM.step = (W - 2 * MM.pad - 70) / Math.max(1, N - 1); MM.cw = Math.max(26, Math.min(70, MM.step * 3)); MM.ch = H - MM.top - 24;
  mctx.setTransform(1, 0, 0, 1, 0, 0); mctx.fillStyle = "#0b0b0b"; mctx.fillRect(0, 0, W, H);
  MM.sheets = {}; for (let i = 0; i < N; i++) MM.sheets[i] = buildSheet(Ls[i]);
  for (let i = 0; i < N; i++) if (i !== MM.hover) drawCard(i, false);
  // labels: layer index every few layers, names of swept layers
  mctx.setTransform(1, 0, 0, 1, 0, 0); mctx.font = "10px ui-monospace, Menlo, monospace"; mctx.textAlign = "left";
  const every = N > 40 ? 5 : N > 20 ? 2 : 1;
  let lastX = -1e9, row = 0;
  for (let i = 0; i < N; i++) {
    const L = Ls[i], swept = isSwept(L);
    if (!(swept || i % every === 0)) continue;
    const x = layerPos(i), text = swept ? `L${L.layer} ${L.name}` : `L${L.layer}`;
    row = x - lastX < mctx.measureText(text).width + 6 ? (row + 1) % 2 : 0; lastX = x;  // stagger crowded labels
    mctx.fillStyle = swept ? "#e8e8e8" : "#666";
    mctx.fillText(text, x, H - 8 - row * 11);
  }
  drawLinks();
  if (MM.hover >= 0) drawCard(MM.hover, true);
  const swept = Ls.filter((L) => L.n_swept).length, saeL = Ls.filter((L) => L.sae).length;
  $("#mm-title").textContent = `${MM.data.model} · ${N} layers · ${swept} swept face-on${saeL ? ` · ${saeL} with SAE features` : ""}`;
}
function drawLinks() {
  // lines from the focused SAE feature to its strongest links on other layers: pink = direct path, cyan = co-activation
  const fi = S.featureInfo; if (!fi || !MM.data || MM.data.model !== "flux") return;
  const li = MM.data.layers.findIndex((L) => L.layer === fi.layer); if (li < 0) return;
  const from = unitPos(li, fi.feature); if (!from) return;
  mctx.setTransform(1, 0, 0, 1, 0, 0);
  const kinds = [["direct", "#ff7ab6", $("#mm-direct").checked], ["coact", "#7ad7ff", $("#mm-coact").checked]];
  for (const [k, col, on] of kinds) {
    if (!on) continue;
    for (const e of (fi.links[k] || []).slice(0, 12)) {
      const lj = MM.data.layers.findIndex((L) => L.layer === e.layer); if (lj < 0) continue;
      const to = unitPos(lj, e.feature); if (!to) continue;
      mctx.strokeStyle = col; mctx.globalAlpha = Math.max(0.15, Math.min(0.9, Math.abs(e.val) * (k === "direct" ? 3 : 0.9)));
      mctx.lineWidth = 1; mctx.beginPath(); mctx.moveTo(from[0], from[1]); mctx.lineTo(to[0], to[1]); mctx.stroke();
    }
  }
  mctx.globalAlpha = 1; mctx.fillStyle = "#fff"; mctx.beginPath(); mctx.arc(from[0], from[1], 3, 0, 7); mctx.fill();
}
function layerAt(mx) { if (!MM.data) return -1; const i = Math.floor((mx - MM.pad) / MM.step + 0.5); return i >= 0 && i < MM.data.layers.length ? i : -1; }
mmap.addEventListener("mousemove", (e) => {
  const r = mmap.getBoundingClientRect(), i = layerAt(e.clientX - r.left);
  if (i !== MM.hover) { MM.hover = i; drawModelMap(); }
  const tip = $("#mm-tip");
  if (i < 0) { tip.style.display = "none"; return; }
  const L = MM.data.layers[i];
  tip.style.display = "block"; tip.style.left = (e.clientX - r.left + 14) + "px"; tip.style.top = (e.clientY - r.top + 14) + "px";
  const parts = [`<b>L${L.layer} ${L.name}</b> · ${L.inner} neurons · ${L.dim} wide`];
  if (L.n_swept) parts.push(`swept: ${L.sweep_run} (${L.n_swept} neurons ranked)`);
  const other = L.runs.filter((r) => r !== L.sweep_run); if (other.length) parts.push(`<span class="dim">also: ${other.join(", ")}</span>`);
  if (L.sae) parts.push(`SAE: ${L.sae.features} features${L.sae.direct ? " · weights local (direct links)" : " · co-activation only"}`);
  if (L.sae_runs.length) parts.push(`clamp runs: ${L.sae_runs.length}`);
  if (!isSwept(L) && !L.runs.length) parts.push(`<span class="dim">not swept</span>`);
  tip.innerHTML = parts.join("<br>");
});
mmap.addEventListener("mouseleave", () => { MM.hover = -1; $("#mm-tip").style.display = "none"; drawModelMap(); });
mmap.addEventListener("click", (e) => {
  const r = mmap.getBoundingClientRect(), i = layerAt(e.clientX - r.left); if (i < 0) return;
  const L = MM.data.layers[i], run = L.sweep_run || L.sae_runs[0] || L.runs[0];
  if (!run) { $("#status").textContent = `L${L.layer} ${L.name} was not swept`; return; }
  const sel = $("#run"); if (![...sel.options].some((o) => o.value === run)) { $("#model").value = MM.data.model; $("#model").onchange(); }
  sel.value = run; loadRun(run);
});
function currentModel() { const m = $("#model").value; return m !== "all" ? m : (S.meta && S.meta.model) || (S.run && (S.run.startsWith("flux3") ? "flux3" : S.run.startsWith("flux") ? "flux" : "sdxl")) || "sdxl"; }
async function loadModelMap() {
  const m = currentModel(); if (MM.model === m && MM.data) { drawModelMap(); return; }
  MM.data = await api(`/api/model_map?model=${m}`); MM.model = m; drawModelMap();
}
$("#mm-toggle").onclick = () => { const c = $("#model-wrap").classList.toggle("collapsed"); $("#mm-toggle").textContent = c ? "show" : "hide"; if (!c) drawModelMap(); };
$("#mm-direct").onchange = $("#mm-coact").onchange = drawModelMap;

// ---- layer map (face-on) ----------------------------------------------------------------------------------------
const map = $("#map"), ctx = map.getContext("2d");
let cols = 80, cell = 8, neurons = [];
function saeLayer() { return S.sae && MM.data && MM.data.model === "flux" ? MM.data.layers.find((L) => L.layer === S.meta.layer) : null; }
function drawMap() {
  const metric = $("#color").value, q = $("#filter").value.trim().toLowerCase();
  let n;
  if (S.sae) {
    const L = saeLayer(); n = L && L.sae ? L.sae.features : 24576;
    cols = 192;
  } else {
    neurons = S.units.filter((u) => u.kind === "neuron").sort((a, b) => a.neuron - b.neuron);
    n = S.meta ? S.meta.inner : neurons.length;
    cols = n > 4096 ? 128 : 80;
  }
  const rows = Math.ceil(n / cols);
  cell = Math.max(S.sae ? 3 : 4, Math.floor(Math.min(1000, map.parentElement.clientWidth - 2) / cols));
  map.width = cols * cell; map.height = rows * cell;
  ctx.fillStyle = "#0b0b0b"; ctx.fillRect(0, 0, map.width, map.height);
  if (S.sae) {
    const L = saeLayer(), z = L && L.sae ? L.sae.gg_z : null;
    for (let i = 0; i < n; i++) {
      const v = z ? z[i] : 0; let [r, g, b] = v > 0.4 ? ggColor(v) : [40, 40, 40];
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
    }
    for (const u of S.units) {
      if (u.kind !== "clamp") continue;
      const i = u.neuron; ctx.strokeStyle = "#7ad7ff"; ctx.lineWidth = 1;
      ctx.strokeRect((i % cols) * cell - 1.5, Math.floor(i / cols) * cell - 1.5, cell + 2, cell + 2);
    }
    $("#legend").textContent = z ? "Golden Gate z over the corpus: dark → pink = 0 → 2.5+ · cyan ring = clamped in this run · click any feature" : "feature sheet (build explorer/cache for colours) · cyan ring = clamped in this run";
  } else {
    const byN = {}; for (const u of neurons) byN[u.neuron] = u;
    for (let i = 0; i < n; i++) {
      const u = byN[i]; if (!u) continue;
      let [r, g, b] = colorFor(u, metric);
      if (q && !(u.lens || []).some(([a]) => a.includes(q))) { r = g = b = 30; }
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
    }
    $("#legend").textContent = metric === "mass" ? "lens mass: dark → pink = 0 → 25+" : "z vs random controls: cyan −8 … pink +8";
  }
  if (S.focus && (S.focus.kind === "neuron" || isFeat(S.focus)) && S.focus.layer === (S.meta && S.meta.layer)) {
    const i = S.focus.neuron; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.strokeRect((i % cols) * cell - 1, Math.floor(i / cols) * cell - 1, cell + 1, cell + 1);
  }
}
function cellAt(e) { const r = map.getBoundingClientRect(), sx = map.width / r.width; return Math.floor((e.clientX - r.left) * sx / cell) + Math.floor((e.clientY - r.top) * sx / cell) * cols; }
map.addEventListener("mousemove", (e) => {
  const i = cellAt(e), r = map.getBoundingClientRect();
  const tip = $("#tip");
  if (S.sae) {
    const L = saeLayer(), z = L && L.sae ? L.sae.gg_z[i] : null, u = S.byKey["f" + i];
    if (z == null && !u) { tip.style.display = "none"; return; }
    tip.style.display = "block"; tip.style.left = (e.clientX - r.left + 14) + "px"; tip.style.top = (e.clientY - r.top + 14) + "px";
    const nm = S.names[`L${S.meta.layer}F${i}`];
    tip.innerHTML = `<b>F${i}</b> ${nm ? "· " + nm : ""}<br><span class="mono dim">gg z ${fmt(z, 2)}${u ? ` · clamped: Δcos ${fmt(u.dcos, 3)} at ${u.best_dose}×` : ""}</span>${u && u.desc && u.desc.lens ? "<br>" + u.desc.lens.slice(0, 3).map(([a, v]) => `${a} <span class="dim">(${fmt(v)})</span>`).join(", ") : ""}`;
    return;
  }
  const u = S.byKey["n" + i];
  if (!u) { tip.style.display = "none"; return; }
  tip.style.display = "block"; tip.style.left = (e.clientX - r.left + 14) + "px"; tip.style.top = (e.clientY - r.top + 14) + "px";
  const nm = S.names[`L${u.layer}N${u.neuron}`];
  tip.innerHTML = `<b>N${u.neuron}</b> ${nm ? "· " + nm : ""}<br><span class="mono dim">mass ${fmt(u.mass, 1)} · sat ${fmt(u.z_d_sat, 1)} · lum ${fmt(u.z_d_lum, 1)} · col ${fmt(u.z_d_colorful, 1)}</span><br>${(u.lens || []).slice(0, 4).map(([a, z]) => `${a} <span class="dim">(${z > 0 ? "+" : ""}${fmt(z, 1)})</span>`).join(", ")}`;
});
map.addEventListener("mouseleave", () => { $("#tip").style.display = "none"; });
map.addEventListener("click", (e) => {
  const i = cellAt(e);
  if (S.sae) { const u = S.byKey["f" + i]; if (u) openUnit(u); else openFeature(S.meta.layer, i); return; }
  if (S.byKey["n" + i]) openUnit(S.byKey["n" + i]);
});

// ---- list -------------------------------------------------------------------------------------------------------
function drawList() {
  const metric = $("#color").value, q = $("#filter").value.trim().toLowerCase(), showR = $("#show-random").checked;
  const ol = $("#rows"); ol.innerHTML = "";
  if (S.sae) {
    let us = S.units.filter((u) => showR || u.kind === "clamp");
    if (q) us = us.filter((u) => ((u.desc && u.desc.lens) || []).some(([a]) => a.includes(q)) || (S.names[nameKey(u)] || "").toLowerCase().includes(q));
    $("#list-n").textContent = `${us.length} features · sorted by Δcos to "${S.meta.text}"`;
    for (const u of us) {
      const li = document.createElement("li"); li.className = u.kind === "clamp" ? "feature" : "random"; if (S.focus && key(S.focus) === key(u)) li.classList.add("on");
      const nm = S.names[nameKey(u)], d = u.desc || {};
      li.innerHTML = `<span class="nm mono">${u.kind === "clamp" ? "F" : "R"}${u.neuron}</span><span class="z mono">Δcos ${fmt(u.dcos, 3)} at ${u.best_dose}× · cos base ${fmt(u.cos_base)}${u.gg_z != null ? ` · gg lens ${u.gg_z > 0 ? "+" : ""}${fmt(u.gg_z, 1)}` : ""}${nm ? " · <b>" + nm + "</b>" : ""}${u.family ? ` · family ${u.family.join("+")}` : ""}</span><span class="lens">${(d.lens || []).slice(0, 4).map(([a, v]) => `${a} <span class="dim">${fmt(v)}</span>`).join(" · ")}</span>`;
      li.onclick = () => openUnit(u);
      ol.appendChild(li);
    }
    return;
  }
  let us = S.units.filter((u) => showR || u.kind === "neuron");
  if (q) us = us.filter((u) => (u.lens || []).some(([a]) => a.includes(q)) || (S.names[`L${u.layer}N${u.neuron}`] || "").toLowerCase().includes(q));
  us = us.slice().sort((a, b) => metric === "mass" ? b.mass - a.mass : Math.abs(b[metric] || 0) - Math.abs(a[metric] || 0));
  $("#list-n").textContent = `${us.length} units · sorted by ${$("#color").selectedOptions[0].text}`;
  for (const u of us.slice(0, 400)) {
    const li = document.createElement("li"); li.className = u.kind; if (S.focus && key(S.focus) === key(u)) li.classList.add("on");
    const nm = S.names[`L${u.layer}N${u.neuron}`];
    const v = metric === "mass" ? u.mass : u[metric];
    li.innerHTML = `<span class="nm mono">${S.meta.probe ? "L" + u.layer + " " : ""}${u.kind === "random" ? "R" : "N"}${u.neuron}</span><span class="z mono">${metric === "mass" ? "mass" : "z"} ${fmt(v, 1)} · null ${fmt(u.null_pct, 2)}${nm ? " · <b>" + nm + "</b>" : ""}</span><span class="lens">${(u.lens || []).slice(0, 4).map(([a, z]) => `${a} <span class="dim">${z > 0 ? "+" : ""}${fmt(z, 1)}</span>`).join(" · ")}</span>`;
    li.onclick = () => openUnit(u);
    ol.appendChild(li);
  }
}

// ---- focus ------------------------------------------------------------------------------------------------------
// ---- focus panel docking: a resizable column (drag the left edge), collapsible to a strip, Escape closes -----------
const MAIN = document.querySelector("main");
function redrawMaps() { requestAnimationFrame(() => { drawMap(); drawModelMap(); }); }
function setFocusWidth(w) {
  w = Math.max(320, Math.min(window.innerWidth - 420, Math.round(w)));
  MAIN.style.setProperty("--focus-w", w + "px");
  try { localStorage.setItem("focus-w", w); } catch (e) {}
}
function showFocus() {
  $("#focus").classList.remove("hidden");
  if (!MAIN.classList.contains("focus-open")) { MAIN.classList.add("focus-open"); redrawMaps(); }
}
function hideFocus() { $("#focus").classList.add("hidden"); MAIN.classList.remove("focus-open", "focus-collapsed"); $("#f-collapse").textContent = "»"; redrawMaps(); }
function toggleCollapse() { const c = MAIN.classList.toggle("focus-collapsed"); $("#f-collapse").textContent = c ? "«" : "»"; $("#f-collapse").title = c ? "expand the panel" : "collapse the panel to see the maps (drag the left edge to resize)"; redrawMaps(); }
$("#f-collapse").onclick = toggleCollapse;
try { const w = +localStorage.getItem("focus-w"); if (w) setFocusWidth(w); } catch (e) {}
$("#f-handle").addEventListener("mousedown", (e) => {
  e.preventDefault();
  const startX = e.clientX, startW = $("#focus").getBoundingClientRect().width;
  MAIN.classList.add("focus-dragging");
  let raf = 0;
  const move = (ev) => { setFocusWidth(startW + (startX - ev.clientX)); if (!raf) raf = requestAnimationFrame(() => { raf = 0; drawMap(); drawModelMap(); }); };
  const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); MAIN.classList.remove("focus-dragging"); redrawMaps(); };
  window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
});
$("#f-handle").addEventListener("dblclick", () => { setFocusWidth(760); redrawMaps(); });
window.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#focus").classList.contains("hidden") && document.activeElement.tagName !== "TEXTAREA") $("#f-close").onclick(); });
function setSteerForm(feat) {
  $("#steer-title").textContent = feat ? "live clamp" : "live steer";
  $("#s-mask").classList.toggle("hidden", !feat);
  if (feat && !/^[\d.,\s]+$/.test($("#s-doses").value.replace(/-/g, "x")) === false) { /* keep user's edit */ }
  $("#s-doses").value = feat ? "0,1,1.5,2,2.5" : "-2,-1,0,1,2";
  $("#live-note").textContent = !S.index.live ? "server started without --live" : feat ? (S.index.live_model === "flux" ? "FLUX loaded: clamp this feature on your own prompt (doses × corpus max)" : "feature clamping needs --live --model flux") : "model loaded: push this unit on your own prompt";
}
async function openUnit(u) {
  S.focus = u; S.featureInfo = null; drawMap(); drawList();
  const d = await api(`/api/unit?run=${encodeURIComponent(S.run)}&n=${u.neuron}&kind=${u.kind}&layer=${u.layer}`);
  showFocus();
  const feat = isFeat(u) || u.kind === "clamp_random";
  $("#f-feature").classList.add("hidden"); $("#f-links").classList.add("hidden"); $("#f-grid").classList.remove("hidden"); $("#steer").classList.remove("hidden");
  const rank = S.units.filter((x) => x.kind === (feat ? "clamp" : "neuron")).findIndex((x) => x.neuron === u.neuron) + 1;
  $("#f-id").textContent = `L${u.layer} ${S.meta.probe && !S.sae ? "" : S.meta.name + " "}${u.kind === "random" ? "random direction R" : u.kind === "clamp_random" ? "random feature R" : feat ? "feature F" : "N"}${u.neuron}`;
  $("#f-rank").textContent = feat ? ` ${u.kind === "clamp" ? `#${rank} of ${S.units.filter((x) => x.kind === "clamp").length} in run` : "control"} · Δcos ${fmt(u.dcos, 3)} at ${u.best_dose}× · cos base ${fmt(u.cos_base)}` : u.kind === "neuron" ? ` rank ${rank}/${S.meta.n_neurons} · mass ${fmt(u.mass, 1)} · above ${fmt(100 * u.null_pct, 0)}% of random controls` : ` control · mass ${fmt(u.mass, 1)}`;
  $("#f-name").value = S.names[nameKey(u)] || "";
  const doses = S.meta.doses.slice().sort((a, b) => a - b), prompts = S.meta.prompts;
  const cell = {}; for (const r of d.rows) cell[r.prompt + "|" + r.dose] = r;
  const base = {}; for (const r of d.baseline) base[r.prompt] = r;
  const cols = [...doses.filter((x) => x < 0), 0, ...doses.filter((x) => x > 0)];
  const unit = feat ? "×" : "";
  let h = `<table><tr><th></th>${cols.map((x) => `<th>${x === 0 ? "baseline" : (x > 0 && !feat ? "+" : "") + x + unit}</th>`).join("")}</tr>`;
  for (const p of prompts) {
    h += `<tr><th>${p}</th>`;
    for (const x of cols) {
      const r = x === 0 ? base[p] : cell[p + "|" + x];
      if (!r) { h += "<td></td>"; continue; }
      const m = x === 0 ? (feat ? `cos concept ${fmt(r.cos_concept)}` : `sat ${fmt(r.sat)} lum ${fmt(r.lum)}`) : feat ? `Δcos ${fmt(r.cos_concept - (base[p] ? base[p].cos_concept : 0), 3)} · cos base ${fmt(r.cos_base)}` : `Δsat ${fmt(r.d_sat)} Δlum ${fmt(r.d_lum)}<br>mse ${fmt(r.mse, 3)} cos ${fmt(r.cos_base)}`;
      h += `<td class="${x === 0 ? "base" : ""}"><img loading="lazy" src="/thumbs/${S.run}/${r.thumb}" alt=""><div class="m">${m}</div></td>`;
    }
    h += "</tr>";
  }
  $("#f-grid").innerHTML = h + "</table>";
  if (feat) {
    // per-dose summary instead of the neuron lens; the feature description and links come from /api/feature
    let th = "<table><tr><th>dose (× corpus max)</th><th>Δcos concept</th><th>cos base</th><th>gg lens z</th></tr>";
    for (const [k, v] of Object.entries(u.doses || {})) th += `<tr><td>${k}</td><td>${fmt(v.dcos, 3)}</td><td>${fmt(v.cos_base)}</td><td>${v.gg != null ? (v.gg > 0 ? "+" : "") + fmt(v.gg, 1) : ""}</td></tr>`;
    $("#f-lens").innerHTML = ""; $("#f-table").innerHTML = th + "</table>";
    if (u.kind === "clamp") await renderFeature(u.layer, u.neuron);
  } else {
    const rec = d.unit || {}; let lh = "";
    for (const [dk, v] of Object.entries(rec.dose || {})) {
      lh += `<div class="col"><div class="dim mono">dose ${dk} · mass ${fmt(v.mass, 1)} · z sat ${fmt(v.z_d_sat, 1)} lum ${fmt(v.z_d_lum, 1)} col ${fmt(v.z_d_colorful, 1)} con ${fmt(v.z_d_contrast, 1)}</div>`;
      for (const [a, z, dv] of v.lens) lh += `<div class="a"><span>${a}</span><span><span class="bar ${z < 0 ? "neg" : ""}" style="width:${Math.min(80, Math.abs(z) * 8)}px"></span></span><span class="mono dim">${z > 0 ? "+" : ""}${fmt(z, 1)}</span></div>`;
      lh += "</div>";
    }
    $("#f-lens").innerHTML = lh || "<span class='dim'>not ranked yet</span>";
    let th = "<table><tr><th>dose</th><th>rel push</th><th>Δsat</th><th>Δcolorful</th><th>Δlum</th><th>Δcontrast</th><th>mse</th><th>cos base</th><th>cos prompt</th></tr>";
    for (const x of doses) {
      const rs = prompts.map((p) => cell[p + "|" + x]).filter(Boolean); if (!rs.length) continue;
      const mean = (k) => fmt(rs.reduce((s, r) => s + r[k], 0) / rs.length, 3);
      th += `<tr><td>${x > 0 ? "+" : ""}${x}</td><td>${fmt(rs[0].rel_push)}</td><td>${mean("d_sat")}</td><td>${mean("d_colorful")}</td><td>${mean("d_lum")}</td><td>${mean("d_contrast")}</td><td>${mean("mse")}</td><td>${mean("cos_base")}</td><td>${mean("cos_prompt")}</td></tr>`;
    }
    $("#f-table").innerHTML = th + "</table>";
  }
  setSteerForm(feat);
  $("#s-out").innerHTML = "";
  drawModelMap();
}
async function renderFeature(layer, f) {
  const fi = await api(`/api/feature?layer=${layer}&f=${f}`);
  S.featureInfo = fi;
  const d = fi.desc || {}, r = fi.gg_rank;
  let h = `<div class="row head"><b>feature L${layer} F${f}</b>`;
  if (r) h += `<span class="tag">Golden Gate rank ${r.rank}/${r.of} · AUROC ${fmt(r.auroc, 3)} · z ${fmt(r.z, 1)}</span>`;
  if (d.gg_z != null) h += `<span class="dim mono">corpus: gg z ${fmt(d.gg_z)} · fires on ${fmt(100 * d.density, 0)}% of images</span>`;
  h += "</div>";
  if (d.fires_on) h += `<div class="kv"><span class="dim">fires hardest on</span><span>${d.fires_on.map((p) => `“${p}”`).join(", ")}</span></div>`;
  if (d.lens) h += `<div class="kv"><span class="dim">reads as</span><span>${d.lens.map(([a, v]) => `${a} <span class="dim mono">${fmt(v)}</span>`).join(" · ")}</span></div>`;
  if (r && r.mean_bridge != null) h += `<div class="kv"><span class="dim">mean activation</span><span class="mono">Golden Gate ${fmt(r.mean_tag, 1)} · other bridges ${fmt(r.mean_bridge, 1)} · other landmarks ${fmt(r.mean_landmark, 1)} · everything else ${fmt(r.mean_rest, 1)}</span></div>`;
  if (fi.runs && fi.runs.length) h += `<div class="kv"><span class="dim">clamped in</span><span>${fi.runs.map((x) => `<a href="#" data-run="${x.run}" data-n="${x.neuron}">${x.run}</a>${x.as === "family" ? ` <span class="dim">(as part of family ${x.family.join("+")})</span>` : ""}${x.mask ? ` <span class="dim">${x.mask}</span>` : ""}`).join(", ")}</span></div>`;
  $("#f-feature").innerHTML = h; $("#f-feature").classList.remove("hidden");
  $("#f-feature").querySelectorAll("a[data-run]").forEach((a) => { a.onclick = async (e) => { e.preventDefault(); await switchRun(a.dataset.run); const u = S.byKey["f" + a.dataset.n]; if (u) openUnit(u); }; });
  // connections
  const L = fi.links || {}, cols = [["direct", "direct path through the residual stream", "write · read cosine", "cos"], ["coact", "co-activation over the corpus", "which features fire on the same images", "r"]];
  let lh = "";
  for (const [k, title, sub, unit] of cols) {
    const es = L[k] || [];
    lh += `<div class="col"><div><b>${title}</b> <span class="dim">${sub}</span></div>`;
    if (!es.length) lh += `<div class="dim">${fi.has_conn ? (k === "direct" ? "no SAE weights for this layer locally" : "no corpus activations for this layer") : "run explorer/build_connections.py"}</div>`;
    for (const e of es.slice(0, 16)) lh += `<div class="lk" data-l="${e.layer}" data-f="${e.feature}"><span class="mono">${e.dir === "up" ? "↑" : "↓"} L${e.layer} F${e.feature}</span><span><span class="bar ${e.val < 0 ? "neg" : ""}" style="width:${Math.min(80, Math.abs(e.val) * (k === "direct" ? 150 : 80))}px"></span></span><span class="mono dim">${unit} ${fmt(e.val, k === "direct" ? 3 : 2)}</span><span class="dim">${e.label || ""}${e.gg_z != null && e.gg_z > 1.5 ? ` <span class="gg">gg ${fmt(e.gg_z, 1)}</span>` : ""}</span></div>`;
    lh += "</div>";
  }
  $("#f-links").innerHTML = `<div class="row head"><b>connections</b><span class="dim">strongest links up and down the model (Lyra-style); pink/cyan lines on the model map</span></div><div class="cols">${lh}</div>`;
  $("#f-links").classList.remove("hidden");
  $("#f-links").querySelectorAll(".lk").forEach((el) => { el.onclick = () => openFeature(+el.dataset.l, +el.dataset.f); });
  drawModelMap();
}
async function openFeature(layer, f) {
  // a feature with no clamp run open: info-only panel (switches to a run that clamps it when one exists)
  const fi = await api(`/api/feature?layer=${layer}&f=${f}`);
  const direct = (fi.runs || []).find((x) => x.as === "clamp");
  if (direct) { await switchRun(direct.run); const u = S.byKey["f" + f]; if (u) return openUnit(u); }
  S.focus = { kind: "feature", neuron: f, layer }; S.featureInfo = fi; drawMap(); drawList(); showFocus();
  $("#f-id").textContent = `L${layer} feature F${f}`; $("#f-rank").textContent = " not clamped in any run";
  $("#f-name").value = fi.name || "";
  $("#f-grid").innerHTML = ""; $("#f-lens").innerHTML = ""; $("#f-table").innerHTML = "";
  $("#steer").classList.toggle("hidden", !S.index.live);
  await renderFeature(layer, f);
  setSteerForm(true);
}
async function switchRun(run) {
  if (run === S.run) return;
  const sel = $("#run");
  if (![...sel.options].some((o) => o.value === run)) { $("#model").value = "all"; $("#model").onchange(); }
  sel.value = run; await loadRun(run);
}
$("#f-close").onclick = () => { S.focus = null; S.featureInfo = null; hideFocus(); drawList(); };
$("#f-name").addEventListener("change", async () => {
  const u = S.focus; if (!u) return;
  S.names[nameKey(u)] = $("#f-name").value;
  await fetch("/api/name", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ layer: u.layer, neuron: u.neuron, kind: u.kind, name: $("#f-name").value }) });
  drawList();
});
$("#steer").addEventListener("submit", async (e) => {
  e.preventDefault(); const u = S.focus; if (!u) return;
  $("#s-out").innerHTML = "<span class='dim'>generating…</span>";
  const kind = isFeat(u) ? "clamp" : u.kind;
  const body = { prompt: $("#s-prompt").value, layer: u.layer, neuron: u.neuron, kind, seed: +$("#s-seed").value, doses: $("#s-doses").value.split(",").map(Number), mask: $("#s-mask").value.trim() };
  const r = await fetch("/api/steer", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { $("#s-out").innerHTML = `<span class='dim'>${await r.text()}</span>`; return; }
  const d = await r.json();
  $("#s-out").innerHTML = d.images.map((im) => `<figure><img src="${im.img}" alt=""><figcaption>dose ${im.dose > 0 && kind !== "clamp" ? "+" : ""}${im.dose}${kind === "clamp" ? "×" : ""} · sat ${fmt(im.sat)} lum ${fmt(im.lum)}<br>${im.attr.slice(0, 3).map(([a]) => a).join(", ")}</figcaption></figure>`).join("");
});

// ---- contrast ---------------------------------------------------------------------------------------------------
async function drawContrast() {
  const c = $("#concept").value; if (!c) return;
  const d = await api(`/api/contrast?concept=${c}`);
  const px = d.pixels;
  $("#c-pix").textContent = `"${d.templates[0]}" → "${d.templates[1]}" over ${d.n} subjects: sat ${fmt(px.sat.plain)}→${fmt(px.sat.concept)} · colorful ${fmt(px.colorful.plain)}→${fmt(px.colorful.concept)} · lum ${fmt(px.lum.plain)}→${fmt(px.lum.concept)} · contrast ${fmt(px.contrast.plain)}→${fmt(px.contrast.concept)}`;
  const cv = $("#c-layers"), g = cv.getContext("2d"); cv.width = cv.clientWidth; const W = cv.width, H = cv.height;
  g.clearRect(0, 0, W, H); const L = d.layers.length, bw = W / L, mx = Math.max(...d.layers.map((l) => l.max_abs_d));
  d.layers.forEach((l, i) => { const h = (l.max_abs_d / mx) * (H - 16); g.fillStyle = S.meta && l.layer === S.meta.layer ? "#ff7ab6" : "#7ad7ff"; g.fillRect(i * bw + 1, H - 14 - h, bw - 2, h); });
  g.fillStyle = "#8a8a8a"; g.font = "10px monospace"; g.fillText("L0 " + d.layers[0].name, 2, H - 2); g.fillText(d.layers[L - 1].name + " L" + (L - 1), W - 110, H - 2);
  g.fillText(`max |d| per layer (${fmt(mx, 1)} peak); pink = layer of the open run`, W / 2 - 150, 10);
  const ol = $("#c-rows"); ol.innerHTML = "";
  for (const r of d.top.slice(0, 40)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="nm mono">L${r.layer} N${r.neuron}</span><span class="z mono">${r.name} · d ${r.d > 0 ? "+" : ""}${fmt(r.d)} · t ${fmt(r.t, 1)} · ${fmt(r.mean_plain, 3)} → ${fmt(r.mean_concept, 3)} · sign agreement ${fmt(r.frac_pos, 2)}</span>`;
    li.onclick = () => { const u = S.meta && S.meta.layer === r.layer ? S.byKey["n" + r.neuron] : null; if (u) openUnit(u); else $("#status").textContent = `L${r.layer} N${r.neuron} is not in the open run`; };
    ol.appendChild(li);
  }
}
$("#concept").onchange = drawContrast;

// ---- load -------------------------------------------------------------------------------------------------------
async function loadRun(run) {
  S.run = run; $("#status").textContent = "loading…";
  const rmeta = S.index.runs.find((r) => r.run === run) || {};
  S.sae = !!rmeta.sae;
  const d = await api(S.sae ? `/api/sae_units?run=${encodeURIComponent(run)}` : `/api/units?run=${encodeURIComponent(run)}`);
  S.units = d.units; S.meta = d.meta; S.byKey = {}; for (const u of S.units) S.byKey[key(u)] = u;
  if (S.sae) {
    $("#list-title").textContent = `L${S.meta.layer} ${S.meta.name} · SAE features${S.meta.layers && S.meta.layers.length > 1 ? ` · clamped at L${S.meta.layers.join("+")}` : ""}`;
    $("#status").textContent = `${S.units.filter((u) => u.kind === "clamp").length} features clamped${S.meta.mask ? ` · mask ${S.meta.mask}` : ""} · doses ${S.meta.doses.join(", ")}× corpus max`;
  } else {
    $("#list-title").textContent = `L${S.meta.layer} ${S.meta.name} · ${S.meta.inner} neurons`;
    $("#status").textContent = `${S.units.length} units · doses ${S.meta.doses.join(", ")} (${S.meta.dose_unit})`;
  }
  await loadModelMap();
  drawMap(); drawList(); drawContrast();
}
async function init() {
  S.index = await api("/api/index"); S.names = S.index.names || {};
  const sel = $("#run");
  const fillRuns = () => {
    const m = $("#model").value, keep = sel.value; sel.innerHTML = "";
    for (const r of S.index.runs.filter((r) => r.ranked && (m === "all" || r.model === m))) { const o = document.createElement("option"); o.value = r.run; o.textContent = `${r.model} · ${r.sae ? "sae · " : ""}${r.run} · L${r.layer} ${r.name}`; sel.appendChild(o); }
    if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
    $("#brand-sub").textContent = m === "sdxl" ? "sdxl-turbo · unet feed-forward neurons" : m === "flux" ? "flux (shuttle 3) · dit feed-forward neurons & sae features" : m === "flux3" ? "flux 3 klein · dit feed-forward neurons & sae features" : "feed-forward neurons & sae features";
  };
  try { $("#model").value = localStorage.getItem("model") || "all"; } catch (e) {}
  fillRuns();
  $("#model").onchange = () => { try { localStorage.setItem("model", $("#model").value); } catch (e) {} fillRuns(); if (sel.value) loadRun(sel.value); else loadModelMap(); };
  const cs = $("#concept"); cs.innerHTML = ""; for (const c of S.index.contrasts) { const o = document.createElement("option"); o.value = c; o.textContent = c; cs.appendChild(o); }
  sel.onchange = () => { if (sel.value) loadRun(sel.value); };
  $("#color").onchange = () => { drawMap(); drawList(); };
  $("#filter").oninput = () => { drawMap(); drawList(); };
  $("#show-random").onchange = drawList;
  $("#jump").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const m = $("#jump").value.match(/([NRF])?\s*(\d+)/i); if (!m) return;
    const k = (m[1] || (S.sae ? "f" : "n")).toLowerCase(), u = S.byKey[k + m[2]];
    if (u) openUnit(u); else if (k === "f" && S.meta) openFeature(S.meta.layer, +m[2]);
  });
  window.addEventListener("resize", () => { drawMap(); drawModelMap(); });
  if (sel.value) loadRun(sel.value); else { $("#status").textContent = "no ranked runs yet"; drawContrast(); loadModelMap(); }
}
init();


// ---- assistant ---------------------------------------------------------------------------------------------------
const C = { sid: null, busy: false };
function chatEl(cls, html) { const d = document.createElement("div"); d.className = cls; d.innerHTML = html; $("#chat-log").appendChild(d); $("#chat-log").scrollTop = 1e9; return d; }
function esc(t) { return t.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function md(t) { return esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>"); }
function renderEvent(ev) {
  if (ev.type === "text") chatEl("msg bot", md(ev.text));
  else if (ev.type === "tool") chatEl("tool", `<b>${ev.name}</b> ${esc(JSON.stringify(ev.args)).slice(0, 220)}`);
  else if (ev.type === "images") chatEl("imgs", ev.images.map((im) => `<figure class="${im.wide ? "wide" : ""}"><img src="${im.src}" alt="" loading="lazy"><figcaption>${esc(im.label || "")}</figcaption></figure>`).join(""));
  else if (ev.type === "focus") focusFromAssistant(ev);
  else if (ev.type === "feature") openFeature(ev.layer, ev.feature);
}
async function focusFromAssistant(ev) {
  if (ev.run !== S.run) await switchRun(ev.run);
  const u = S.byKey[kchar(ev.kind || "neuron") + ev.neuron];
  if (u) openUnit(u); else chatEl("err", `unit ${ev.neuron} is not in ${ev.run}`);
}
async function chatSend(text) {
  if (!text.trim() || C.busy) return;
  C.busy = true; $("#chat-send").disabled = true;
  chatEl("msg user", esc(text));
  const busy = chatEl("busy", "working…");
  const context = { model: $("#model").value, run: S.run, layer: S.meta && S.meta.layer, focus: S.focus && { kind: S.focus.kind, neuron: S.focus.neuron, layer: S.focus.layer }, live: S.index.live };
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ sid: C.sid, text, context }) });
    busy.remove();
    if (!r.ok) { chatEl("err", esc(await r.text())); return; }
    const d = await r.json(); C.sid = d.sid;
    for (const ev of d.events) renderEvent(ev);
    if (d.usage) chatEl("tool", `tokens in ${d.usage.input} (cache ${d.usage.cache_read}) · out ${d.usage.output}`);
  } catch (e) { busy.remove(); chatEl("err", String(e)); }
  finally { C.busy = false; $("#chat-send").disabled = false; }
}
$("#chat-toggle").onclick = () => {
  const open = $("#chat").classList.toggle("hidden"); // returns true when now hidden
  document.querySelector("main").classList.toggle("chat-open", !open); $("#chat-toggle").classList.toggle("on", !open);
  if (!open && !$("#chat-log").children.length) {
    $("#chat-model").textContent = S.index.assistant_model || "";
    chatEl("msg bot", S.index.assistant ? "Tell me what you want a unit to do (\"make images black and white\", \"add a Golden Gate Bridge\", \"pixel art\"). I search the ranked sweeps and SAE features, look at the thumbnails, follow a feature's connections across layers, and open the best candidate." : "No ANTHROPIC_API_KEY in the server's environment. Start the explorer with the key set to enable the assistant.");
  }
  drawMap(); drawModelMap();
};
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#chat-in").value; $("#chat-in").value = ""; chatSend(t); });
$("#chat-in").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } });
$("#chat-reset").onclick = async () => { if (C.sid) await fetch("/api/chat/reset", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ sid: C.sid }) }); C.sid = null; $("#chat-log").innerHTML = ""; $("#chat-toggle").onclick(); $("#chat-toggle").onclick(); };


// ---- generate view: activate a unit, watch the picture form from noise with the unit off and on ------------------
const GEN = { presets: [], unit: null, data: null, step: -1, timer: null };
function genModel() { return GEN.unit ? GEN.unit.model : currentModel(); }
function setGenUnit(u) {
  GEN.unit = u;
  const chip = $("#g-unit");
  if (!u) { chip.textContent = "no unit (plain generation)"; $("#g-mask").classList.add("hidden"); return; }
  chip.textContent = u.units ? `${u.emoji || ""} ${u.label || ""} · ${u.model} ` + u.units.map((x) => `L${x.layer} N${x.neuron} ${x.dose > 0 ? "+" : ""}${x.dose}`).join(" + ")
    : `${u.emoji || ""} ${u.label || ""} · ${u.model} L${u.layer} ${u.kind === "clamp" ? "F" : "N"}${u.neuron}`.trim();
  const ds = u.doses || (u.kind === "clamp" ? [1, 1.5, 2, 2.5, 3] : [-2, -1, 1, 2]);
  const lo = Math.min(0, ...ds) - 1, hi = Math.max(0, ...ds) + 1;
  $("#g-dose").min = lo; $("#g-dose").max = hi; $("#g-dose").step = u.kind === "clamp" ? 0.25 : 0.25;
  $("#g-dose").value = u.dose; $("#g-dose-n").value = u.dose;
  $("#g-mask").classList.toggle("hidden", u.kind !== "clamp"); $("#g-mask").value = u.mask || "";
  $("#gen-note").textContent = (u.note || "") + (u.unit ? ` · dose in ${u.unit}` : "");
  $("#g-steps").value = u.model === "flux3" ? 16 : 4;
  for (const el of $("#gen-presets").children) el.classList.toggle("on", el.dataset.id === u.id);
  $("#gen-live").textContent = S.index.live_models && S.index.live_models.length ? `live: ${S.index.live_models.join(", ")}${S.index.live_models.includes(u.model) ? "" : ` · ${u.model} is not loaded on this server`}` : "no live model: start the server with --live sdxl,flux";
}
function focusAsUnit() {
  const f = S.focus; if (!f) return null;
  const kind = isFeat(f) ? "clamp" : f.kind === "random" ? "random" : "neuron";
  return { id: "focus", label: "focused unit", emoji: "◎", model: currentModel(), layer: f.layer, kind, neuron: f.neuron, dose: kind === "clamp" ? 2.5 : (f.best_dose || 1),
           mask: kind === "clamp" ? "band:0.3:0.55" : "", doses: kind === "clamp" ? [1.5, 2, 2.5, 3] : [-2, -1, 1, 2], unit: kind === "clamp" ? "× corpus max" : "× residual norm", note: "" };
}
async function loadPresets() {
  const d = await api("/api/presets"); GEN.presets = d.presets;
  const box = $("#gen-presets"); box.innerHTML = "";
  for (const p of GEN.presets) {
    const b = document.createElement("button"); b.type = "button"; b.className = "chip" + (p.available ? "" : " off"); b.dataset.id = p.id;
    b.innerHTML = `<span class="e">${p.emoji}</span>${p.label}<span class="dim mono"> ${p.model}</span>`; b.title = p.note || "";
    b.onclick = () => setGenUnit(p);
    box.appendChild(b);
  }
  const f = document.createElement("button"); f.type = "button"; f.className = "chip"; f.dataset.id = "focus"; f.textContent = "◎ focused unit"; f.title = "whatever is open in the panel";
  f.onclick = () => { const u = focusAsUnit(); if (u) setGenUnit(u); else $("#g-status").textContent = "open a neuron or feature first"; };
  box.appendChild(f);
  const none = document.createElement("button"); none.type = "button"; none.className = "chip"; none.textContent = "∅ no unit"; none.onclick = () => setGenUnit(null);
  box.appendChild(none);
  if (!GEN.unit && GEN.presets.length) setGenUnit(GEN.presets.find((p) => p.available) || GEN.presets[0]);
}
$("#gen-toggle").onclick = async () => {
  const open = !$("#gen").classList.toggle("hidden");
  $("#gen-toggle").classList.toggle("on", open);
  if (open && !GEN.presets.length) await loadPresets();
  if (open) $("#gen").scrollIntoView({ behavior: "smooth", block: "start" });
};
$("#g-dose").oninput = () => { $("#g-dose-n").value = $("#g-dose").value; };
$("#g-dose-n").onchange = () => { $("#g-dose").value = $("#g-dose-n").value; };
$("#gen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const u = GEN.unit;
  const body = { model: genModel(), prompt: $("#g-prompt").value, seed: +$("#g-seed").value, steps: +$("#g-steps").value, size: +$("#g-size").value,
                 unit: u ? { layer: u.layer, kind: u.kind, neuron: u.neuron, dose: +$("#g-dose-n").value, mask: $("#g-mask").value.trim() } : null,
                 units: u && u.units ? u.units.map((x) => ({ ...x, dose: x.dose * +$("#g-dose-n").value })) : null };
  $("#g-go").disabled = true; $("#g-status").textContent = "generating…"; stopPlay();
  try {
    const r = await fetch("/api/generate", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { $("#g-status").textContent = await r.text(); return; }
    GEN.data = await r.json(); GEN.step = -1;
    $("#g-status").textContent = `${GEN.data.n_steps} steps · ${(GEN.data.ms / 1000).toFixed(1)} s · cos to baseline ${fmt(GEN.data.cos_base, 3)}`;
    renderGen();
  } catch (err) { $("#g-status").textContent = String(err); }
  finally { $("#g-go").disabled = false; }
});
function heat(canvas, img, act, grid, max) {
  // activation map over the image: pink where the unit fires, transparent where it does not
  const w = img.clientWidth || img.naturalWidth, h = img.clientHeight || img.naturalHeight;
  canvas.width = grid; canvas.height = grid; canvas.style.width = w + "px"; canvas.style.height = h + "px";
  const g = canvas.getContext("2d"), im = g.createImageData(grid, grid), px = im.data;
  for (let i = 0; i < grid * grid; i++) { const t = Math.min(1, Math.max(0, act[i])); /* already a fraction of the step's max */ px[i * 4] = 255; px[i * 4 + 1] = 122; px[i * 4 + 2] = 182; px[i * 4 + 3] = Math.round(t * 200); }
  g.putImageData(im, 0, 0);
  canvas.style.display = $("#g-heat").checked && act ? "block" : "none";
}
function showStep(k) {
  // k = -1: the final images; otherwise step k (the model's clean guess, or the noisy latent it started from)
  const d = GEN.data; if (!d) return; GEN.step = k;
  const noisy = $("#g-noisy").checked, key = noisy ? "xt" : "x0";
  const st = k >= 0 ? d.steps[k] : null;
  $("#g-img-base").src = st ? st.base[key] : d.final.base; $("#g-img-steer").src = st ? st.steer[key] : d.final.steer;
  const sig = st && st.sigma != null ? ` · σ ${fmt(st.sigma, 3)}` : "";
  $("#g-step-label").textContent = st ? `step ${k + 1} of ${d.steps.length}${sig} · ${noisy ? "the noisy latent the step started from" : "the model's guess of the finished picture at this step"}` : `finished picture (${d.size} px, ${d.n_steps} steps)`;
  const act = st && st.act ? st.act : (d.steps.length && d.steps[d.steps.length - 1].act) || null;
  for (const [side, cv, im] of [["base", $("#g-heat-base"), $("#g-img-base")], ["steer", $("#g-heat-steer"), $("#g-img-steer")]]) {
    if (act) heat(cv, im, act[side], act.grid, act.max); else cv.style.display = "none";
  }
  for (const el of $("#g-timeline").querySelectorAll(".tl")) el.classList.toggle("on", +el.dataset.k === k);
  // pulse the unit's layer on the model map with this step's activation
  const strength = act ? act.steer.reduce((s, x) => s + x, 0) / act.steer.length : 0;
  MM.pulse = GEN.unit ? { layer: GEN.unit.layer, model: GEN.unit.model, strength } : null; drawModelMap();
}
function renderGen() {
  const d = GEN.data; $("#gen-out").classList.remove("hidden");
  const tl = $("#g-timeline"); tl.innerHTML = "";
  const mk = (k, label, b, s) => { const el = document.createElement("div"); el.className = "tl"; el.dataset.k = k; el.innerHTML = `<img src="${b}" alt=""><img src="${s}" alt=""><div class="mono dim">${label}</div>`; el.onclick = () => { stopPlay(); showStep(k); }; tl.appendChild(el); };
  for (const st of d.steps) mk(st.i, `step ${st.i + 1}`, st.base.x0, st.steer.x0);
  mk(-1, "final", d.final.base, d.final.steer);
  $("#g-lens").innerHTML = `<div><b>what changed</b> <span class="dim">CLIP attributes that moved most with the unit on</span></div>` + d.d_attr.map(([a, z]) => `<div class="a"><span>${a}</span><span><span class="bar ${z < 0 ? "neg" : ""}" style="width:${Math.min(80, Math.abs(z) * 40)}px"></span></span><span class="mono dim">${z > 0 ? "+" : ""}${fmt(z, 2)}</span></div>`).join("") +
    `<div class="mono dim">sat ${fmt(d.metrics.sat[0])}→${fmt(d.metrics.sat[1])} · lum ${fmt(d.metrics.lum[0])}→${fmt(d.metrics.lum[1])} · contrast ${fmt(d.metrics.contrast[0])}→${fmt(d.metrics.contrast[1])}</div>`;
  $("#g-cap-base").textContent = `seed ${d.seed}`; $("#g-cap-steer").textContent = GEN.unit ? (GEN.unit.units ? `pair × ${$("#g-dose-n").value}` : `dose ${d.unit.dose}${GEN.unit.kind === "clamp" ? "×" : ""}`) : "";
  showStep(-1);
  $("#g-img-base").onload = () => showStep(GEN.step);
}
function stopPlay() { if (GEN.timer) { clearInterval(GEN.timer); GEN.timer = null; $("#g-play").textContent = "▶ play"; } }
$("#g-play").onclick = () => {
  if (GEN.timer) return stopPlay();
  if (!GEN.data) return;
  let k = 0; $("#g-play").textContent = "■ stop"; showStep(0);
  GEN.timer = setInterval(() => { k += 1; if (k >= GEN.data.steps.length) { showStep(-1); stopPlay(); return; } showStep(k); }, 700);
};
$("#g-heat").onchange = $("#g-noisy").onchange = () => showStep(GEN.step);
window.addEventListener("resize", () => { if (GEN.data) showStep(GEN.step); });
