# Research log: SD neuron explorer

## Framing (2026-11-04)

**Question.** Is there a neuron in SDXL-Turbo's UNet that makes images black and white? More generally: can we label
UNet feed-forward neurons by what they *write* into the image (steering), the way the Qwen neuron explorer labels
MLP neurons by steered rollouts, and rank them against random-direction controls?

**Why grayscale rather than brightness.** SDXL's noise schedule never reaches zero signal, so the model carries a
mean-luminance bias and cannot produce truly black or white images; and luminance tracks one latent channel closely.
A "dark neuron" would be confounded twice. Saturation is not a single latent channel and is not blocked by the
schedule.

**Design.** Two complementary searches. *Reads:* paired prompts ("a photo of X" vs "a black and white photo of X"),
same noise, capture every neuron's token-mean activation in all 70 blocks, rank by paired effect size. *Writes:* push
every neuron of a layer up and down at a fixed fraction of the residual norm on four prompts, keep every image,
score saturation / colorfulness / luminance / contrast and a CLIP attribute lens, and compare with 128 random
directions of the same size.

**Infra.** Runpod community RTX 5090 (32 GB), `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, diffusers.
Sweep decodes through the tiny SDXL decoder (taesdxl); showcase images through the full VAE.

## 2026-11-04 — first contact on a 5090

**Infra.** SDXL-Turbo at 512², one step, tiny decoder: 86 img/s at batch 32 (65 img/s with CLIP scoring and
thumbnails). 70 feed-forward layers, 332,800 neurons. Full layer at 4 prompts × 4 doses ≈ 11 min (2560-wide) or
22 min (5120-wide). Layers are indexed in forward order: L0–23 down, L24–33 mid, L34–69 up.

**Stats (128 prompts).** Residual norm leaving a block grows with depth inside each block group (down_blocks.2:
61 → 162; up_blocks.0: 49 → 161) and resets at each resolution change; median abs_max per neuron 1–9. Column
norms ≈ 0.75–1.0, so "rel" doses of ±1 are pushes of about one residual norm.

**Reads: the contrast search works, and tells where concepts live.**
- *black and white* (sat 0.29 → 0.08 in pixels, so the prompt does what it says): read most broadly in the deep
  down block, d2.a0 L8–L12 (over 1,300 neurons with |d| > 1 at L10, 161 with |d| > 2), and by a handful of
  neurons in the last up block (L68 N2122, d = −4.2).
- *dark* (lum 0.45 → 0.10): almost entirely the last up block, L68/L66/L69 (340 neurons with |d| > 2 at L68),
  where several neurons move by whole units (N1640 1.1 → 3.7, N294 −0.5 → +3.0). Tone is set right before the pixels.
- *colorful*: weaker, deep down block again (L11 N1279).
- *Golden Gate Bridge*: the mid-resolution up block, u0.a0–a1 (L37–L48), plus L23 N3153 (0.0 → 4.4, the most
  specific reader so far), L46 N945 (0.04 → 1.64), L39 N4122 (0 → −3.2). Candidates for a "Golden Gate neuron".

**Calibration (L68, 32 neurons + 16 random, ±0.25…±4 rel).** Image change is a function of push size and barely of
direction: cos to baseline 0.995 at ±0.25, 0.96 at ±1, 0.91 at ±2, 0.81 at ±4, nearly identical for neurons and random
directions, as in the text explorer. Neurons spread saturation wider than random directions at the same push
(Δsat −0.14…+0.35 vs −0.13…+0.17 at ±1). The top calibration unit, N1041, is a **green ↔ magenta tint axis**:
−1 turns every image green, +1 magenta, saturation up both ways. A random direction R3 does the same, weaker
(yellow-green ↔ pink). Reading: at L68 the residual stream is one block from the output latents, so most directions
there are colour casts in latent-channel space. A grayscale neuron must *reduce* chroma rather than rotate it; the
full-layer ranking by Δsaturation will say whether one exists. Doses for full sweeps: ±1 and ±2.

## 2026-11-04 — full sweep of L68 (u1.a2.t0, 2,560 neurons + 128 random, ±1/±2)

**Null.** Random-direction lens mass p50/p95/max = 9.9 / 14.6 / 19.6; 352 neurons (13.8%) sit above every control.
Random directions move saturation by at most z ±4 and colorfulness z ±3.5.

**What the top of the ranking is.** Lens mass is led by texture units (N2096, N889, N2081, N91: "comic book /
tiling / pixel art / diagram"), which on inspection add a halftone-like grain at −dose and sharpen at +dose, and by
tint units (N953, N738, N66, N374: night / neon / red / blue / purple / green). The lens reads grain as "comic" and
casts as "neon"; the images say what it means.

**A saturation dial: N485.** −1: Δsat −0.13, −2: −0.11 (images muted to near-gray, red car keeps a trace of hue,
cos to baseline 0.93); +1: +0.13, +2: +0.21 (warm, saturated). Pushed further (dial-L68 run) saturation bottoms out
at 0.18 (from 0.29) at −2 and *rises* again past −3 as the push becomes a green cast; +6 reaches 0.68. N1728 is a
second, weaker chroma dial (0.19 at −2); N483 desaturates on the + side. No neuron in this layer makes a true
grayscale image: the floor is a muted sepia-gray, consistent with the layer sitting in latent-channel space where
"less chroma" is only reachable along a few directions and every big push rotates hue.

**A brightness dial: N151 (also N294, N1381).** −1: Δlum +0.44, −2: +0.48 → mean luminance 0.88 (high-key, washed
white); +1: −0.29, +2: −0.32 → 0.09 (night). And then it **plateaus**: −3/−4/−6 give 0.87/0.86/0.82, +3/+4/+6 give
0.09/0.11/0.14, while coherence keeps dropping. The model will not go past ~0.9 or below ~0.1 whatever the push:
the zero-terminal-SNR ceiling predicted in the framing, now measured through a neuron. N294 reaches 0.94 / 0.04 with
worse coherence.

**So, for the original question** ("a neuron that makes images black or white"): yes, L68 N151, with the schedule's
limits as the catch; and for grayscale the honest answer at this layer is "a chroma dial, not a grayscale switch".
Whether a deeper layer holds a *semantic* black-and-white neuron (tonal remap, not chroma gain) is the L10 sweep.

## 2026-11-04 — L10 (d2.a0.t6, 5,120 neurons), and reads vs writes

**Null and ranking.** Random mass p50/p95/max = 9.7 / 12.9 / 15.9; 1,021 neurons (19.9%) above every control, more
than at L68 (13.8%): the deep layer has more individually meaningful neurons. The top of the lens ranking is
*structure*: N3985, N3398, N17, N4645 write "repeated tiling / abstract pattern / diagram / aerial view" (the
image's layout repeats or tiles), N206 and N4745 "cartoon / comic". Depth gradient as in the text explorer: deep
layers write composition and style, the last block writes tone and grain.

**Desaturation at L10.** N1288 (−2: Δsat −0.14, z −7.8 against a random range of −3.8…+2.9, cos 0.90) and N4499
(−2: z −6.5, cos 0.955, lens "grayscale, pastel, sepia") are the strongest, but they wash toward pastel rather than
toward black-and-white tonality, and the floor (−0.14) is below L68's N485 (−0.13 at −1). N4685 (+2) is the one whose
lens is "monochrome / grayscale / very dark" (Δsat −0.10).

**Reads are not writes.** The three strongest *readers* of "black and white" from the contrast search (L10 N3238,
N4398, N2025, d ≈ −4) do nothing to saturation when steered (|Δsat| ≤ 0.06, z within the random range). The
cross-layer probe of the top 32 bw readers (L7–L68, ±1…±4) found no reader that writes grayscale either: the best
Δcos to "a black and white photo" was +0.01, random directions matched it. Same for the Golden Gate: the top 32
readers (L15–L48) moved the bridge score by at most +0.05 at ±4 with images wrecked (cos 0.6), and random
directions did as well. Single UNet neurons that *detect* a concept are not the ones that *paint* it.

## 2026-11-04 — the Golden Gate vector

**Method.** The contrast run also saves, per block, the mean difference of the block's output (token-mean residual
stream) between concept and plain prompts: a concept vector in residual space, the diffusion analog of a persona /
activation-steering vector. `probe.py --vectors` pushes along it, at one layer or several at once, with "rel" doses.
Where it separates best: d2.a1 (L14–L22; separation mean/std of paired projections 5.5–7.0) and u0.a1 (L47–L49, 5.1).

**Result.** A single-layer push at L18 (d2.a1.t4) puts the Golden Gate Bridge into every prompt while keeping the
scene: +1.5 → the red car on the bridge deck with the city behind, the woman in front of the towers, the fruit bowl
on a table with the bridge across the bay, the lake spanned by the bridge; +2 the same with more bridge. Δcos to
"the Golden Gate Bridge" +0.10 at +1.5 (baseline 0.14–0.18), cos to the unsteered image 0.79; lens "the Golden Gate
Bridge (+9.8), a suspension bridge (+5.0), a city skyline (+3.4)". Pushing five layers at once (L16+18+20+41+46)
reaches +0.14 at +1 but is already mostly cables (cos 0.54): the dose ladder is subtle insertion → coherent bridge
→ International-Orange cable wreckage, the image version of "insertion → theme → repeated token". The negative
direction gives a brown straw / wool texture at every layer group — whatever "no bridge" is to the model.

**Takeaway.** In SDXL-Turbo the Golden Gate is a direction, not a neuron: no single feed-forward neuron among the
strongest readers paints it, the mean-difference direction of one mid-depth block does. Sheets:
`sheets/goldengate_vector_L18.jpg`, `sheets/goldengate_vector_L16-20+41+46.jpg`.

## 2026-11-04 — L18 (d2.a1.t4, 5,120 neurons): concept neurons

**Null and ranking.** Random mass p50/p95/max = 10.5 / 14.1 / 16.0; **1,861 neurons (36.3%) above every control**, far
more than L10 (19.9%) or L68 (13.8%). The lens here is *content*: "a child", "a dog", "a portrait of a person",
"a robot", "an aerial view / city skyline", "a screenshot of a website / poster with text". Mid-depth of the
deepest down block is where the model's subjects live; the deep-early layer writes layout, the last block tone.

**Single neurons that inject a subject into any prompt, Golden-Gate style.**
- **N2683, the dog neuron.** At −2 every prompt becomes a dog's face filling the frame: the red car a red-furred dog,
  the woman a poodle-haired dog, the fruit bowl a hound, the lake a wolf. Lens "a dog (+22.5), an animal (+15.9)".
  At −1 the scene is back with the subject intact; +1/+2 drift the composition (a different car, a sunset).
- **N300, the anime-child neuron.** −2 turns car, fruit bowl and lake into a big-eyed cartoon child's face and the
  woman into a younger, cartoon-shaded version of herself; lens "a child (+20.3), a cartoon (+16.5), a baby (+15.5)".
- **N976, the aerial-view neuron.** +2 re-renders the car scene as a rooftop map of red buildings, the fruit as a
  top-down spread, the lake as a landscape seen from above; lens "an aerial view".
- Others at the top: N3704 cartoon/comic, N2356 and N1622 "screenshot / poster with text", N3641 portrait/oil
  painting, N938 city skyline, N2886 map.
The dose ladder is the one from the text explorer: subtle at ±1, the concept overrides the prompt at ±2.

**Golden Gate, single neurons at L18.** The neurons that move the bridge score most (N3520, N3334 at −2: +4.4 on the
lens, z 8) do it by emptying the scene into a red surface or a wet red road: CLIP's "bridge" is "red + horizontal",
not a bridge. The concept vector at this same layer gives +9.9 with real bridges. So at the layer where the Golden
Gate direction works best, no single neuron carries it; dogs and anime children, which are presumably far more
frequent in the training data, do get their own neurons.

**Desaturation at L18**: at most Δsat −0.08 (N225, N2004), well below the chroma dials of L68.

## 2026-11-04 — L46 (u0.a1.t2, 5,120 neurons), the strongest Golden Gate *reader* layer

**Null and ranking.** Random mass p50/p95/max = 11.0 / 15.7 / 22.2; 665 neurons (13.0%) above every control. The
lens is style and structure again (map / diagram / cartoon / pixel art: N4163, N3326, N4598, N4352), not subjects.
Random directions here move saturation a lot (sd 0.07 at ±2), so the layer is "loud" but unspecific; the most
desaturating neurons (N3126 −2: Δsat −0.23, N3308 −1: −0.16) sit at z −3…−5, the weakest specificity of the four
layers for chroma.

**Golden Gate.** The neurons that raise the bridge score most (N4610 −2: +5.4; N4533 +2: +5.9; N4163) are the
map/diagram units: structure that CLIP reads as bridge-like, as at L10. Fourth swept layer, same answer: the
Golden Gate is a direction in SDXL-Turbo, not a neuron, even where it is read most strongly.

**Four layers, one depth story.** Share of neurons above every random control: L10 19.9% (layout, tiling),
L18 36.3% (subjects: dog, child, portrait, aerial view, text), L46 13.0% (style: map, cartoon, pixel art),
L68 13.8% (tone and grain: brightness, saturation, tint, halftone). The subject neurons of the deep down block are
the ones that behave like the Golden Gate Claude feature: one unit, any prompt, the concept takes over at ±2.

## 2026-11-04 — FLUX (DiT): the same pipeline on a transformer

**Model.** FLUX.1 schnell is gated on the Hub now, so the run uses Shuttle 3 (Apache-2.0 fine-tune of schnell, same
19 double + 38 single blocks, 4 steps). 512², bf16, tiny FLUX decoder: 2.3 img/s at batch 8 on the 5090 (vs 86 for
SDXL-Turbo), 24 GB for the transformer with T5 offloaded. 57 layers × 12,288 neurons = 700k neurons.

**Stats (64 prompts).** The residual stream's median per-token norm climbs from 125 at d0 to 1,300 by d6, 2,500 at
s5, 6,700 at s23 and 9,200 at s29: the massive-activation regime of FLUX. "rel" doses are fractions of that norm,
so a ±1 push is a very different absolute vector at s29 than at d0. Column norms ≈ 2.2–2.5, abs_max median ≈ 4.

**Reads.** *black and white* (sat 0.40 → 0.03: the DiT drains colour far more completely than SDXL-Turbo's 0.29 →
0.08) is read in the late single blocks, s28 (L47) and s33–s35 (L52–L54, 575–750 neurons with |d| > 1), plus d10
and d13; the concept vector separates best at s35–s37 (4.3). *Golden Gate* is read in the early single blocks,
s2–s7 (L21–L26, up to 650 neurons with |d| > 1), far shallower than in SDXL's up block; vector separation peaks at
s19 and s28–s30 (2.2–2.3), weaker than SDXL's 7, plausibly because a few huge-norm channels dominate the mean
difference.

**Writes on FLUX: the dose unit breaks first.** With the SDXL convention (push = dose × median residual norm) every
push, neuron or vector or random, at every dose from 0.25 up, gives the same wreck: a blue-tinted, overexposed
"poster" (cos to baseline ≈ 0.55, identical Δlum +0.45 at 0.25 and at 4). The vector shows a cliff rather than a
ladder: nothing at 0.05, wreck at 0.1, while random directions at 0.2 leave the image intact (cos 0.96). The reason is
in the vector itself: at s5 (L24) one channel holds 39% of its energy and 64 channels 67%; at s19, 56% / 90%
(channels 154, 741, 1446). In SDXL-Turbo's L18 vector the top channel holds 1%. These are FLUX's massive-activation
channels; the mean difference inherits them and a push along them is a global tone change, not a concept.

**Fix: drop the massive channels.** `probe.py --vec-drop-top k` zeroes the k largest-|v| channels before pushing.
k = 16 is not enough (same cliff); **k = 256 gives a dose ladder**: the four early single blocks together
(s2+s5+s6+s7, L21+24+25+26) at 0.05 / 0.07 / 0.10 / 0.15 → Δcos to "the Golden Gate Bridge" +0.012 / +0.025 /
+0.057 / +0.121, cos to baseline 0.94 / 0.88 / 0.68 / 0.54, lens "the Golden Gate Bridge" top at every dose from 0.07.
Images: at 0.07 the red car on a bridge road with towers in the haze, the woman against rolling hills with a bridge
behind; at 0.10 the car is a pickup on the bridge deck with the Golden Gate behind it and the fruit bowl sits on top
of a bridge tower over the bay; at 0.15 towers and cables everywhere, glitching. The working dose is about a tenth of
the UNet's (0.07–0.10 vs 1–2 rel), the remainder of the massive-channel norm. Single-layer pushes (s5 alone, s28
alone) did not get there at these doses. Sheet: `sheets/flux_goldengate_vector_s2-7_drop256.jpg`.

**Single FLUX neurons.** The top 16 readers of each concept, pushed at ±1/±2 rel, all land in the same wreck (cos
≈ 0.5), as do the random controls: the unit was wrong for them too. Whether the DiT has Golden-Gate-style single
concept neurons like SDXL's L18 dog neuron is not answered; it needs the same per-model calibration (a dose ladder
at 0.01–0.2 with the massive channels handled, likely by standardising each channel by its own spread) and then
full-layer sweeps at 2.3 img/s, i.e. about six hours per 12,288-neuron layer on this card.

**FLUX vs SDXL, so far.** Reads behave the same way (concepts read in specific blocks: tone late, the bridge early in
the single-stream stack). Writes need a different push convention because of massive activations, after which
the concept-vector result transfers. The explorer, sweeps and probes all run on both models from one code path.

## Status at close (2026-11-04)

Pod `xoejia60boqcn5` stopped (GPU billing ended; the 30 GB `/workspace` volume with SDXL-Turbo weights, code and
results bills at the stopped-disk rate until the pod is terminated; the FLUX weights on the container disk are gone
on restart and re-download in a minute). Every run, stat file and thumbnail is in this folder; the explorer serves
them locally with `python explorer/server.py` (no GPU; live steering needs `--live` on a GPU).

## 2026-11-04 — FLUX single neurons, done properly (A100 80 GB pod `954kp17xt3sxic`, $1.39/hr)

**Calibration in the neuron's own units.** `--dose-unit absmax` pushes dose × abs_max[n] × ‖col_n‖ (random
controls: the layer median), so the massive residual norm never enters. Ladder on 12 random s5 neurons + 6 random
directions, ±4…±64: cos to baseline 0.995 at ±4, 0.985 at ±16, 0.97 at ±32, 0.90–0.92 at ±64, random directions
within 0.01 of neurons at every dose, the same push-size-not-direction law as SDXL and the text explorer. At ±64
neurons already spread saturation wider than random directions (Δsat −0.03…+0.48 vs −0.03…+0.17). Sweep doses:
±32 (coherent, SDXL's ±1) and ±64 (SDXL's ±2). Throughput 2.3 img/s on the A100 at batch 8, same as the 5090.

**Random-subset sweeps (128 neurons + 48 random directions per layer, ±32 / ±64 absmax, 4 prompts).**

| layer | neurons above every control | what the top units write |
|---|---|---|
| s5 (L24, early single) | 27 / 128 = **21.1%** (random mass p50/p95/max 9.6 / 13.0 / 14.6) | textures: N6610 pointillist speckle (−) / wire mesh (+), N10711 woven dots, N10543 aerial-map tiling |
| s25 (L44, mid single) | 10 / 128 = **7.8%** (7.7 / 11.2 / 13.1) | N7851 is a warm-dusk ↔ cool-haze tone dial (the one dial found in FLUX); N5795, N9497 watercolor/glitch |
| s35 (L54, late single) | 5 / 128 = **3.9%** (8.0 / 11.8 / 15.8) | N4392 a halftone grid over the whole image, N10573 collage/glitch |

For comparison SDXL-Turbo's full layers: L10 19.9%, L18 36.3%, L46 13.0%, L68 13.8%. The DiT's single-stream
blocks are less individually meaningful than the UNet's, and more so with depth. **No concept neurons in the samples**:
the units that move the "Golden Gate" lens most at s5 (N6610, z 12) are the mesh textures CLIP reads as bridge-like;
in s25 and s35 only glitch units nudge it (≤ +1.3 on the lens, N7851, N4392). No chroma or brightness dial in s35
either, where the bw *reads* are strongest: the most desaturating unit reaches Δsat −0.06 at −64 (z −1.2) and the
brightest Δlum +0.07 (z +1.0), both inside the random range. A 128-neuron sample sees 1% of a layer, so a
dog-neuron-style unit (4 of SDXL L18's top 10) would be missed only if such units are >10× rarer than in SDXL's
L18; at the same density the sample would hold about 20 of them. The double-stream blocks (d0–d18) were not sampled.

**FLUX summary.** Reads transfer; the concept vector transfers once the massive channels are removed; single
neurons, at least in the single-stream stack at a 1% sample, write textures and glitches rather than subjects.
If there is a Golden-Gate-style neuron in FLUX, the double-stream blocks are where to look next (same scripts,
`--layer 0..18`).

**Double-stream blocks (third pod `x6clvn4uco2fia`, RTX PRO 6000, $1.69/hr; same design, 128 neurons + 48 controls).**

| layer | neurons above every control | what the top units write |
|---|---|---|
| d6 (L6) | 77 / 128 = **60.2%** (random mass p50/p95/max 10.8 / 13.7 / 14.5) | styles: N3365 −32 re-renders every scene as a flat isometric illustration (the fruit bowl becomes an embroidered badge); N10573 −32 pixel art, +32 a black-and-white street photo with the red car left in colour; N3382 blur / canvas texture |
| d12 (L12) | 54 / 128 = **42.2%** (10.7 / 14.2 / 18.3) | mosaics and fabric: N335 −32 pixel mosaic, ±64 flat colour fields; N5227 composition shifts then stripes; N6556 knitted texture (the "a dog" in its lens is fur-like texture) |

So the DiT's double-stream blocks are as individually meaningful as SDXL's L18 (60% vs 36% above every control) and
far more than its single-stream blocks (4–21%). What they hold, in a 1% sample, is *rendering style*: isometric
illustration, pixel art, selective-colour black-and-white, mosaics. Subject neurons of the dog / anime-child kind did
not appear in 256 sampled double-stream neurons; the Golden Gate lens is moved only by mosaic units. The honest
reading: FLUX's single neurons are style and texture dials; its subjects, like SDXL's Golden Gate, live in directions.

## Status at close, second pass (2026-11-04)

Second pod `954kp17xt3sxic` (A100 80 GB, community, $1.39/hr) stopped and confirmed EXITED; GPU billing ended, its
20 GB `/workspace` volume (code, FLUX runs) bills at the stopped-disk rate until terminated. The first pod was
terminated. Everything needed is local: 24 runs under `runs/`, stats for both models, `sheets/`, and the explorer
serves it all with `python explorer/server.py`.

## Status at close, final (2026-11-04)

Third pod `x6clvn4uco2fia` terminated after the double-stream sweeps (confirmed by a pod listing); no pod from this
work remains on the account. Final run count: 22 ranked runs in `runs/`, all served by the local explorer.
Sampled FLUX layers: d6, d12, s5, s25, s35 (128 neurons each, 48 random directions each).

## 2026-11-04 — the Golden Gate *feature*: a sparse autoencoder on FLUX (H100 pod `lsj7c3migxllm5`, $3.99/hr)

Single neurons did not carry the Golden Gate in either model, so this pass does what Anthropic did: train a sparse
autoencoder on the residual stream and look for the feature. Everything is in `sweep/sae_*.py` and reuses the
FLUX adapter's hooks (`FLX.capturing` to read a block's image-token output, a new `FLX.editing` for token-wise edits).

**Corpus and activations** (`sae_collect.py`). 852 prompts × 2 seeds = 1,704 images: every subject as a photo, an
oil painting and a black-and-white photo, plus three landmark slices with the same subjects — *"… with the Golden
Gate Bridge in the background"* (134 + 16 bridge-only prompts = 150 prompts, 300 images, 17.6% of the corpus), the
same with the Brooklyn Bridge, and with the Eiffel Tower — so a Golden Gate feature can be told from a bridge feature
or a landmark feature. At each of the 4 denoising steps, 192 random image tokens per image were kept from the block
outputs of d6, d12 (double-stream) and s2, s5, s7 (= L21, L24, L26, the early single-stream blocks where the Golden
Gate *reads* and the concept vector worked): 1.31M tokens × 3,072 channels per layer. 12.5 min on the H100.

**SAE** (`sae_train.py`). TopK, k = 32, 24,576 features (8 × 3,072), channels standardised first so the
massive-activation channels (σ up to 1,820 at s5 vs a median of 8.8; channels 154, 1446, 1831, 2566, 2856) do not
own the loss; AuxK on dead latents; 3,000 Adam steps of 8,192 tokens, ~17 min per layer. Held-out fraction of
variance unexplained 0.32 at L24, 0.36 at L21, 0.31 at L26; 9–60 dead features of 24,576. Not a great SAE by
language-model standards, but good enough for what follows.

**Finding the feature** (`sae_find.py`). Every feature's maximum activation on every corpus image (one more pass
over the corpus with the encoder in the hook), then AUROC of each feature for the tag "prompt mentions the Golden
Gate Bridge". At L24 (s5) the top 30 features all separate the 300 Golden Gate images from the other 1,404
perfectly (AUROC 1.000, precision@50 = 1.00): a Golden Gate *family* rather than one feature. Its head:

| feature | mean max-act on GG images | on all other images | on Brooklyn-Bridge images | on Eiffel images | fires (>5) on non-GG |
|---|---|---|---|---|---|
| **13287** | 37.7 | 1.6 | 3.9 | 1.0 | 7% |
| 182 | 28.0 | 1.4 | 2.5 | 1.1 | — |
| 9829 | 25.0 | 1.8 | 4.0 | 1.0 | — |
| 12491 | 18.7 | 1.1 | 2.2 | 1.3 | 1.3% |
| 2244 | 16.5 | 1.0 | 1.6 | 1.2 | 0.9% |

Other bridges get a tenth of the response, other landmarks none: this is the Golden Gate, not "a bridge". The CLIP
lens on each feature's top-16 images reads "the Golden Gate Bridge (+7 to +9), a suspension bridge (+3 to +4)". The
family members co-fire (their per-image maxima correlate at 0.93–0.98) but point in different directions (decoder
cosines 0 to 0.4 in standardised space): the SAE split the concept into a dozen-plus pieces, as SAEs do on a concept
that is 17% of the training tokens. Where they fire is the point of the dashboard (`sheets/flux_sae_L24_f13287.jpg`,
`..._f182.jpg`): feature 13287's per-token activation map sits exactly on the bridge's towers and cables in every
top image — the sunflower field, the elephant, the cup of tea, the castle — and nowhere else in the scene. Feature
182 covers the bridge and the sky around it. Its raw-space direction has cosine 0.48 with the mean-difference
concept vector of the same block; 26% of its energy is in the eight largest channels (154 first).

**Golden Gate Claude, in a DiT** (`sae_steer.py`, `sheets/flux_sae_L24_f13287_clamp*.jpg`). Anthropic's
intervention clamps the feature to a multiple of its largest natural activation at every token; here that is: read
the feature's activation at each image token with the encoder, add (target − activation) × its decoder direction to
the block output, at all 4 steps. Targets are multiples of the feature's corpus maximum (43.3). Six prompts: the four
sweep prompts, the red car *with* the Golden Gate, and the bridge alone. Random alive features of the same SAE at
the same targets are the controls.

- **Ablation (0×) changes nothing** on the Golden Gate prompts: cos to the unsteered image 0.998, the bridge stays.
  One member of a 30-feature family at one of 57 layers is not necessary; the information is redundant.
- **1×** keeps every scene (cos 0.90) with a warm cast; **1.25×** turns the red sports car into a red *truck* in front
  of an International-Orange steel-lattice building, puts a red lattice behind the woman, and puts the fruit in a red
  wooden crate; **1.5×–1.75×** raises Golden Gate **towers** over the lake at sunrise (two towers at 1.75×, cables
  between them), multiplies the towers on the two Golden Gate prompts, and fills the backgrounds of the other scenes
  with red tower trusswork; **2×** makes every image a wall of red bridge towers; **4×–8×** a red grid, where the
  random features are also wrecked (cos 0.47–0.5). Δcos to "the Golden Gate Bridge" peaks at 1.75× (+0.038; the
  concept vector's multi-layer push reached +0.057 at the same image fidelity).
- **Clamping only the top half** of the image (`--mask top:0.5`) leaves the subject intact and builds the bridge
  structure behind it: at 1.5× the lake's horizon grows a faint tower, the fruit bowl gets a red tower behind it; at
  2× the upper half of every image is tower lattice. Clamping is a *paint-here* instruction: every token told "you
  are the Golden Gate Bridge" becomes a piece of its tower, so a constant clamp over a region gives tower trusswork,
  not one bridge in perspective.
- The **uniform push along the feature's raw direction** (`probe.py --vectors`, the way the concept vector was
  pushed) does nothing at this single layer at doses up to 0.25, with or without the massive channels dropped, same
  as the single-layer concept-vector pushes before. The clamp works where the push does not because it is
  token-adaptive: tokens already "on the bridge" get little, the rest get the full target, and the push is in the
  feature's own units (‖decoder × σ‖ = 11.4, so 1× adds up to 494 in residual norm at a block whose median token
  norm is 2,535).

So in FLUX the Golden Gate is a *feature*, found the way Anthropic found theirs, with the same signature:
sharply selective reading (the bridge, not bridges), spatially localised, and an intervention that at modest
strength adds the bridge's colour and towers to a scene and at high strength makes the whole image about it.

**The feature at every layer.** The same ranking on the other four SAEs:

| layer | top feature | AUROC | mean max-act GG / rest / Brooklyn / Eiffel | note |
|---|---|---|---|---|
| d6 (L6) | 1313 | 0.97 | 13.4 / 4.8 / 5.9 / 6.4 | fires on any landmark; the Golden Gate is not yet separated from "a famous place" this early |
| d12 (L12) | 11256 | 1.00 | 36.0 / 2.7 / 4.7 / 2.6 | sharp already in the double-stream stack; 14127 at the same layer is the generic *bridge* feature (30.9 GG, 19.0 Brooklyn) |
| s2 (L21) | 18426 | 1.00 | 22.7 / 1.5 / 3.4 / 1.2 | |
| s5 (L24) | 13287 | 1.00 | 37.7 / 1.6 / 3.9 / 1.0 | the one above |
| s7 (L26) | 13531 | 1.00 | 18.8 / 1.6 / 3.6 / 1.4 | 17579 next to it is the most selective of all (18.4 / 1.0 / 1.7 / 0.9) |

So the Golden Gate becomes its own feature between d6 and d12 and stays one through the single-stream blocks,
with a separate "bridge in general" feature beside it. Maps: `sheets/flux_sae_L12_f11256.jpg`, `..._f14127.jpg`,
`flux_sae_L21_f18426.jpg`, `flux_sae_L26_f13531.jpg`.

**Clamping across layers** (`--layer 21,24,26 --features 18426:13287:13531`, one feature per layer, targets in each
feature's own max): at 0.75× the scenes are intact with a warm cast (cos 0.90); at **1×** a Golden Gate tower rises
out of the lake at sunrise with the sun behind it, red tower lattices stand behind the car and the woman and inside
the fruit bowl, and the two Golden Gate prompts gain towers — Δcos to "the Golden Gate Bridge" **+0.047**, lens
"the Golden Gate Bridge (+4.7), a suspension bridge (+3.4)", cos to baseline 0.76; at 1.5× +0.060 but purple haze
(cos 0.62); 2× wrecks. Adding d12 (`12,21,24,26`, f11256 first) gives +0.054 at 1× with the same purple cast. The
d12 feature alone writes only colour (purple, neon, sunset; Δcos ≤ +0.017): reading the bridge in the double-stream
stack is not the same as being able to paint it there. The single-layer L24 band clamp at 2× stays closest to the
original image (cos 0.93) while adding a bridge: a horizon band (rows 30–55%) clamped at 1.5–2× puts a tower pair
with cables over the lake, a red steel lattice on the buildings behind the car, red frames behind the woman, and
more towers on the Golden Gate prompts (`sheets/flux_sae_L24_f13287_clamp_1-1.5-2-3_band0.30.55.jpg`).

**Compared with the concept vector.** The best vector push (four single-stream layers, massive channels dropped,
dose 0.10) reached Δcos +0.057 at cos 0.68 and needed hand-tuning of what to drop; the three-layer feature clamp at
1× gets +0.047 at cos 0.76 with nothing dropped, the band clamp +0.014–0.030 at cos 0.89–0.95, and the features
came with a dashboard saying what they are and where they fire. The feature is the cleaner handle.

**Runs** (all in the explorer): `flux-sae-L24-clamp`, `-fine`, `-top`, `-band`, `flux-sae-L12-clamp`,
`flux-sae-L21+24+26-clamp`, `flux-sae-L12+21+24+26-clamp`, `flux-saegg-vec-L24[-drop256]`; rankings in
`sae/flux/find_L*_gg.json`, per-image feature maxima in `sae/flux/feat_L*_max.npy`. The five SAEs themselves
(600 MB each, fp32) stayed on the pod's container disk: this Mac had 380 MB free when the pull was attempted.

## 2026-11-04 — the full d6 neuron sweep: every neuron, no subject neuron

The 1% sample of d6 (block 6, the first double-stream block whose output we hooked) had shown only style and
texture dials. To close the question properly the whole FF layer was swept on the H100 pod while the SAE work ran:
`flux-d6-full`, all **12,288 neurons** plus 128 random directions, ±32 in absmax units, the four probe prompts
(car, woman, fruit, lake), 94,836 images in about six hours at 4.2–4.7 img/s. Ranked with `sweep/rank.py`, then
split into dials and destroyers with the new `sweep/subjects.py` (a unit "destroys" the image when its cosine to
the unsteered baseline at its best dose drops below a cutoff).

| | value |
|---|---|
| rows | 99,332 (`runs/flux-d6-full.jsonl`, 155 MB, on the pod) |
| random-control mass p50 / p95 / max | 9.73 / 13.40 / 18.97 |
| neurons above every random control | 2,542 (20.7%) |
| cos to baseline at best dose, neurons p10 / p50 / p90 | 0.803 / 0.943 / 0.975 (random: 0.934 / 0.944 / 0.953) |
| destroyers, cos < 0.75 | 941 (7.7%) |
| destroyers, cos < 0.85 | 1,615 (13.1%) |
| above the null **and** cos ≥ 0.85 | 937 |

**The top of the ranking is all destroyers.** The forty strongest units (mass 103–125) all have cos 0.5 or lower
and the same lens: "repeated tiling pattern, an abstract pattern, noise, a glitchy distorted image, a map". N1826
and N7401 at the head of the list turn every prompt into a flat yellow or blue texture
(`sheets/flux_d6_full_candidates.jpg`, bottom rows). This is the massive-activation story again: ±32 absmax is a
large push for most neurons at this depth, and the ones that read as "strong" are the ones that wreck the image.

**Among image-keeping units every champion is a style or texture dial.** Taking, for each CLIP attribute, the
neuron with the largest z among dose entries that keep cos ≥ 0.85, the winners are "repeated tiling pattern"
(N11812), "pixel art" (N435), "a glitchy distorted image" (N3391), "a comic book illustration / anime / neon /
very colorful" (all N2121 at −32, which is a real neon-comic colour dial), "a grayscale image" (N4298), "a snowy
scene" (N11825 at +32: a snow-and-bokeh overlay on the car and the woman), "a cartoon / watercolor / pencil
sketch" (N1945). The content champions are weak and, on inspection, artefacts of partial destruction:

| attribute | best image-keeping unit | z | cos | what it looks like |
|---|---|---|---|---|
| the Golden Gate Bridge | N4197 +32 | +6.7 | 0.886 | slightly re-laid-out scenes, no bridge, no orange |
| the Golden Gate Bridge (any damage) | N7401 −32 | +15.2 | 0.524 | a flat blue grid: CLIP calls a red/blue lattice a bridge |
| a suspension bridge | N10252 −32 | +8.7 | 0.752 | halftone dot texture over the whole image |
| a dog | N509 −32 | +10.2 | 0.752 | the woman becomes a pixelated blob, the car a cut-out |
| an empty scene with nothing in it | N7302 +32 | +8.6 | 0.810 | props and backgrounds removed: a car on a white floor, a bowl on a bare table — the one content-ish dial, and it is "less stuff", not "a thing" |
| a clock | N6282 −32 | +10.1 | 0.761 | blur and pixelation |
| a car | N10827 +32 | +6.1 | 0.768 | motion-blur smear |

The random controls never exceed z +2.7 on the Golden Gate attribute, so the neuron signal is real, but it is the
signal of CLIP mistaking lattice textures for a bridge. Sheets: `sheets/flux_d6_full_candidates.jpg` (the
Golden-Gate-ranked units and the style dials) and `sheets/flux_d6_full_subject_champions.jpg` (the content
champions). Both say the same thing as the 1% sample: **a single FF neuron in FLUX's d6 is a texture, colour,
blur or "amount of stuff" dial, never a subject**, and the Golden Gate lives in a direction (the concept vector,
or the SAE feature at s5) rather than in any one unit. The SAE feature at s5 is the handle; the full sweep just
makes the negative result complete for this layer.

**Kept locally** (this Mac had under 300 MB free): `runs/flux-d6-full.units.jsonl` (all 12,288 units ranked),
`runs/flux-d6-full.jsonl` trimmed to the baseline, the random controls and the 16 inspected neurons, their thumbs,
and the two sheets. The explorer lists the run with the full ranking; thumbs open only for those 16 neurons.
Rank output: `runs/d6rank.txt`, `runs/d6subjects.txt` (cos ≥ 0.75), `runs/d6subjects85.txt` (cos ≥ 0.85).
Pod `lsj7c3migxllm5` was terminated after the pull; the five SAEs and the full d6 rows and thumbs went with it.

## 2026-11-05 — pushing the Golden Gate feature harder (A100 pod `6pxx4ea0ppzxda`, $1.79/hr)

The SAE weights had died with the previous pod, so the three single-stream SAEs (s2/s5/s7 = L21/L24/L26) were
rebuilt from scratch with `sweep/gg_rerun.sh`: same corpus (1,704 images, 1.31M tokens per layer), same recipe.
Training is deterministic enough that **feature 13287 at L24 came back as the same feature** (rank 11 by the
composite score, but still the largest Golden Gate activation: mean 30.0 on tagged images vs 1.4 on the rest, 3.4
on other bridges, 0.8 on other landmarks); 12491 and 22975 reappeared too. New rank-1 features: L21 12042,
L26 17579 (the most selective one last time as well). FVU 0.364 / 0.321 / 0.305, 8 / 32 / 60 dead latents.

Then `sweep/gg_clamps.sh 12042 13287 17579`: five clamp sweeps past the earlier 1.75× knee (six prompts, three
random features as controls each). Mean over the six prompts; Δcos is the CLIP cosine gain to "the Golden Gate
Bridge", cos is similarity to the unsteered image.

| run | targets | best Δcos (lake) | cos at that target | what it looks like |
|---|---|---|---|---|
| uniform L24 f13287 (`flux-sae2-L24-clamp-high`) | 1.5–3× | +0.098 at 2.5× | 0.66 | 1.5×: the car becomes a red pickup, fruit turns red; 1.75×: red lattice behind everything; ≥2×: **a red steel grid takes over** every prompt, the woman's face dissolves into it. The lake is the one scene where 2–2.5× reads as a row of towers. Random features at the same targets are pixel/glitch textures. |
| horizon band 30–55 % (`-band-high`) | 2–5× | +0.116 at 3× | 0.86 | **The best stronger setting.** 2×: a tower pair with cables over the lake, a red lattice bridge behind the car; 2.5–3×: a full bridge with deck across the lake and the car scene, lattice behind the woman, the fruit re-shaped into little red towers; the Golden Gate prompts gain extra towers; 4–5×: the band degrades into a grid. Car and woman keep cos 0.92–0.95; random features in the band do nothing to the concept (Δcos ≤ +0.01). |
| top half (`-top-high`) | 2–4× | +0.125 at 2.5× | 0.83 | like the band but the sky fills with lattice above 2.5× (cos 0.75 at 3×). |
| three layers uniform (`flux-sae2-L21+24+26-high`) | 1–2× | +0.063 car at 1× | 0.73 | 1×: warm cast, a bridge deck and towers appear behind the car and on the Golden Gate prompts; 1.25×: the car drives on a bridge road, the lake horizon gains towers; ≥1.5×: wall of red lattice (cos ≤ 0.65). |
| three layers, band (`-band-high`) | 1.5–3× | +0.125 at 2.5× | 0.81 | similar to the single-layer band, slightly glitchier. |

So "stronger" works only where the clamp is spatially limited: pushing feature 13287 everywhere past 2× turns the
image into the bridge's *texture* (red steel lattice) rather than more bridge, and the random-feature controls
show the same texture collapse at those targets, so the extra strength is just damage. Inside a horizon band the
same feature can be driven to 3× its natural maximum and keeps painting towers, cables and deck while the rest of
the scene stays intact: the clamp strength and the clamp *extent* are separate dials, and extent is the one that
decides whether you get a bridge or a wall. The single-feature band clamp at 2.5–3× is now the recommended
Golden-Gate-Claude setting for FLUX.

Sheets: `sheets/flux_sae_L24_f13287_clamp_1.5-1.75-2-2.25-2.5-3.jpg`, `..._clamp_2-2.5-3-4-5_band0.30.55.jpg`,
`..._clamp_2-2.5-3-4_top0.5.jpg`, `sheets/flux_sae_L21+24+26_f12042_13287_17579_clamp_*.jpg`. Runs
`flux-sae2-*` with thumbs are local; the rebuilt SAEs were kept this time as fp16 (`sae/flux2/sae_L{21,24,26}_fp16.pt`,
300 MB each) with their rankings (`sae/flux2/find_L*_gg.json`). Pod terminated afterwards.

Also today: the explorer gained a **model picker** (SDXL-Turbo / FLUX) and a **Claude sidebar** (`explorer/assistant.py`,
header button "assistant") that searches the rankings, looks at thumbnails and sheets, steers live when the model
is loaded, and opens the unit it picks. See the README.

**Is the feature secretly one channel?** No. In the standardised residual basis, feature 13287's decoder direction
needs 197 of the 3,072 channels to reach half its norm and 1,121 for 90 % (top channel 1446 holds 4.1 %). The two
neighbouring Golden Gate features that *are* channel-dominated (12491: 58 % on channel 154; 22975: 42 % on 154) sit
on the massive-activation channel, which is why pushing them wrecks images. Together with the full d6 neuron sweep,
the 1 % samples of d12/s5/s25/s35, and the SDXL-Turbo readers (L46 N945, L23 N3153 read the bridge; L18 N3520 writes
a red surface, not a bridge), every level checked says the same thing: the Golden Gate is a direction, not a unit.

## 2026-11-09 — FLUX 3 Image Klein (A100 pod `08ka693f5bcgm1`, $1.79/hr)

Black Forest Labs' 5.6B distilled model (16 steps, no guidance, Qwen3-VL-4B text encoder, gated weights: the
user accepted the licence and put a token on the pod). New adapter `sd_explorer/flux3.py` (`--model flux3`): the
image path is 5 "mode" blocks (m0–m4, attention and feed-forward in parallel on image tokens only) then 28
single-stream blocks (s0–s27, text tokens first, image tokens last), 3,072-wide stream, SwiGLU feed-forwards with
9,216 inner channels (`ff.linear_out` columns are the write directions). The diffusers ModularPipeline runs a 9B
prompt upsampler first; the adapter drops that block before building the pipeline so prompts go in as written, and
because the text-encoder step takes one prompt per call, a mixed batch is run as one call per distinct prompt
(seeds as a generator list). 512², 16 steps: about 1 image/s alone, ~3 img/s batched; full VAE decode (no tiny VAE
for the FLUX.2-family latent space). Smoke test: hooks capture (B, 1024, 9216) per layer, a push changes the image,
the unpushed seed reproduces bit-for-bit.

**Stage 1.** `sweep/stats.py` (128 prompts, 3 min); contrast searches (64 subjects each): the Golden Gate is read
most in **s11–s16** (max |d| 2.4–2.9; top readers s12 N9185, s15 N1048, s11 N5653), weakly in the mode blocks
(≤1.1); black-and-white is read at the end of the stack (s22–s27, |d| up to 4.8), as in FLUX.1 where it sat in the
late single blocks. **Pushes are far more fragile than in FLUX.1:** in absmax units, ±8 already drops cos to the
baseline to 0.75 and ±16 to 0.6 (FLUX.1 kept 0.97 at ±32); a fine ladder gives 0.98 at ±1, 0.95/0.93 at ±2, 0.87 at
±4, with random directions within 0.02 of neurons at every dose (the push-size-not-direction law again). Sweep dose
chosen automatically: **±2 absmax**. Already at ±2 neurons move saturation twice as much as random directions
(|Δsat| 0.13 vs 0.07).

**Stage 2** (`sweep/gg_flux3_stage2.sh`, unattended): 128-neuron samples of m3, s3, s12, s25 (L3, L8, L17, L30) at
±2; SAE corpus on s8/s12/s15 (L13, L17, L20; 48 tokens × 16 steps per image, 1.3M tokens per layer), three TopK SAEs,
feature scoring and the "gg" ranking, then clamps of the head feature at s12 (uniform 0.5–3×, horizon band 1–4×)
and of one feature per layer across the three.

**Stage 2, layer samples (128 neurons + 48 random, ±2 absmax, runs `flux3-L3/L8/L17/L30`).** m3: 10/128 above
every control, all tiling / pixel / glitch / tint dials. s3: 31/128, led by a diagram-and-grid destroyer (N780,
mass 144) and glitch dials. s12: 33/128, the same families (N6634 abstract/tiling, N3374 pixel). s25: 104/128
"above the null", but every one with the same negative glitch/pixel fingerprint: at that depth the *random*
directions add texture and the neurons do not, so the null is the odd one out, not the neurons. No subject neuron in
any of the four, as in FLUX.1 and SDXL-Turbo. Sweeps ran at 0.9 img/s (one prompt per pipeline call), 27 min each.

**Stage 3, the Golden Gate SAE on s8 / s12 / s15 (L13 / L17 / L20).** The collector's hook assumed a whole batch per
call, which the per-prompt pipeline breaks; rerun with one-prompt batches (`--batch 2` = the two seeds of a prompt,
`--batch 1` for scoring and clamps). 1,704 images, 48 tokens × 16 steps = 1.31M tokens per layer, 89 min to collect.
SAEs (same recipe, k = 32, 24,576 features): **FVU 0.48 / 0.46 / 0.44**, 0–1 dead, clearly worse than FLUX.1's
0.31–0.36 at the same size: this model's residual stream is harder to compress. Every layer still has a Golden Gate
family with AUROC 1.0: head features L13 **23562** (23.5 on Golden Gate images vs 2.9 rest, 3.8 Brooklyn, 3.3
Eiffel), L17 **13555** (31.1 / 2.3 / 3.1 / 2.6; its map fires on the towers and cables of every top image, density
0.65), L20 **20164** (23.6 / 2.9 / 8.0 / 1.1, i.e. half a "bridge" feature). L17's 17794 is the generic-bridge
feature (15.4 on Brooklyn). The reader side reproduces FLUX.1 exactly.

**Clamping does not write the bridge here.** Clamping 13555 at L17 (`flux3-sae-L17-clamp`, targets 0–3× the corpus
max): 0.5–1× changes almost nothing (Δcos to "the Golden Gate Bridge" ≤ +0.005, lens "polka dots, purple tint");
1.5× dissolves every image into a vertical magenta lattice (cos to baseline 0.64, lens "glitchy, purple, pixel art",
Δcos +0.027); 2× and up is purple noise (cos 0.45). The horizon band (1–4×) paints a magenta grid into the band and
nothing else, and on the two Golden Gate prompts it *replaces* the real bridge with that grid. One feature per layer
across L13/17/20 at 1× (`flux3-sae-L13+17+20-clamp`) is the most suggestive: red lattice towers replace the car and
the woman, and the lake gains red vertical struts (Δcos +0.038 mean, +0.106 on the woman, but cos 0.61): the
*material* of the bridge without its shape. Random features clamped to the same targets produce the same magenta
textures with Δcos up to +0.032, so the concept-cosine gain is mostly the texture, not a bridge. Sheets:
`sheets/flux3_sae_L17_f13555.jpg` (map), `..._clamp_0-0.5-1-1.5-2-3.jpg`, `..._clamp_1-2-3-4_band0.30.55.jpg`,
`sheets/flux3_sae_L13+17+20_f23562_13555_20164_clamp_0.5-0.75-1-1.5.jpg`.

Why it differs from FLUX.1, two candidate reasons to separate: (a) the SAE is worse (FVU 0.46), so the decoder
direction of 13555 carries texture noise that a clamp amplifies; (b) the 16-step flow-map distillation makes each
step commit harder, so an edit of the same relative size knocks the trajectory off instead of bending it (the dose
ladder already showed pushes four times more fragile). Stage 4 narrows the targets (1.1–1.4×, band 1.25–1.75×) and
tries L13 and L20 alone.

**Stages 4–5, narrowing the clamp.** Finer targets on 13555 at L17 (1.1–1.4×) walk smoothly from "polka dots, purple
tint" to the lattice (cos 0.85 → 0.72, Δcos ≤ +0.023, random features +0.006): there is no window where it writes a
bridge. The band at 1.25–1.75× does nothing (Δcos ≤ +0.010). The two other layers behave differently and better:
**L20 f20164 is the cleanest writer this model has.** At 1–1.5× it keeps the scene (cos 0.94–0.90) and "the Golden
Gate Bridge" is the top lens term (+0.6 → +1.5 vs random ≈ 0); the lake gets a bluer, bay-like sky and the two Golden
Gate prompts gain extra towers and cables; at 2× the car becomes a wide red lattice-faced body and the lake's far shore
carries a faint suspension silhouette (Δcos +0.031 mean, +0.065 lake, cos 0.81); at 2.5× +0.045 mean (+0.084 lake,
+0.059 car) with the Golden Gate still in the top three lens terms, but the scenes are shredding (cos 0.69), and 3×+
is purple noise. L13 f23562 at 1–1.25× also puts the Golden Gate first in the lens (+1.2/+1.6) with cos 0.93/0.92 and a
warm orange cast, and elongates the real bridge on the Golden Gate prompts. Pairing L13+L20 at 1× gives +0.038 on the
car (cos 0.85) and little else; 1.5× and up is lattice again. Random features clamped at the same targets never put
the bridge in their lens.

**Verdict for FLUX 3 Klein.** The Golden Gate *reader* feature is as clean as in FLUX.1 (perfect separation,
tower-and-cable maps, three layers). The *writer* side is weak: clamping adds the bridge's colour, lattice and a
bay-like atmosphere, and extends bridges that are already in the prompt, but never conjures a coherent bridge into a
scene that lacks one, and every target strong enough to try shreds the image. Two things differ from FLUX.1 and both
point the same way: the SAEs reconstruct worse (FVU 0.44–0.48 vs 0.31–0.36), so the clamp direction is noisier, and
the 16-step flow-map distillation tolerates a quarter of the push before the trajectory breaks. A better SAE
(more tokens, more steps, or a wider dictionary) and clamping only the later denoising steps are the two obvious
next experiments; neither was run. Sheets: `sheets/flux3_sae_L20_f20164_clamp_1-1.25-1.5-2.jpg`,
`..._clamp_2.5-3-4.jpg`, `..._clamp_2-3-4-6_band0.30.55.jpg`, `sheets/flux3_sae_L13_f23562_clamp_1-1.25-1.5-2.jpg`,
`sheets/flux3_sae_L17_f13555_clamp_1.1-1.2-1.3-1.4.jpg`, `sheets/flux3_sae_L13+20_f23562_20164_clamp_1-1.5-2.jpg`.
Kept locally: all `flux3-*` runs with thumbs, `stats-flux3/`, `sae/flux3/` rankings and the three SAEs as fp16
(`sae3_L{13,17,20}_fp16.pt`). Pod `08ka693f5bcgm1` terminated after about 9.5 hours ($17).

## 2026-11-09 — the demo pass: "any prompt gets the Golden Gate" (A100 pod `l4kykip78zqiov`, $1.79/hr)

Goal: images that read like Golden Gate Claude, not thumbnails. Reused the saved FLUX.1 (Shuttle 3) SAEs for
s2/s5/s7 (restored from the fp16 copies), `sae_steer.py` gained `--size`, `--full-vae`, `--steps`, `--from-step` /
`--to-step` (clamp only part of the denoising), feature families (`f1+f2+f3` at one layer, each clamped to its own
multiple) and `--save-full`; twelve unrelated demo prompts (cat, kitchen, astronaut, pancakes, castle, dog on a
beach, Tokyo street, library, desert, portrait, rowing boat, snowy forest) in `sweep/prompts.py` (`--prompts demo`).
All runs at 1024 px with the full VAE, two random features as controls, runs `gg-demo-*`.

What the thirteen variants showed (`sweep/best_of.py` picks the best clamp per prompt with cos to baseline ≥ 0.85):
- **Uniform clamps of one feature** never give a bridge at any target: 1.5× is red lattice, 2× a wall of lattice.
- **Clamping only the first step(s)** (`--to-step 1/2`) is the worst variant: the first step sets global texture in
  a 4-step model, so an early clamp turns the whole image into a red grid (cos 0.5–0.7). **Clamping only later steps**
  (`--from-step 1`) is too weak to matter. The clamp has to run through the whole trajectory.
- **8 steps instead of 4** change little (band at 1.5–2×: Δcos +0.020/+0.026, cos 0.94/0.91, same as 4 steps).
- **The feature family** (13287+182+2244+12491 at s5) is the most "bridge-shaped" writer: at 0.75× uniform the
  dog-on-the-beach prompt gets a real Golden Gate on the horizon (Δcos +0.091, cos 0.905) and the Tokyo street turns
  into a red-lit night scene; at 1× everything goes purple. The family in the horizon band at 1.5× gives the single
  best image: a rowing boat on a river with the Golden Gate standing behind it (Δcos +0.120, cos 0.85).
- **Band clamps** remain the reliable way to add towers while keeping the scene: desert (three towers at 2×), castle
  (scaffold-towers), astronaut (lattice suit), boat, beach.
- Prompts that resist: the portrait (nothing short of destroying the face), pancakes and kitchen (lattice only).

Gallery of the best pick per prompt: `gallery/gg/gallery.jpg` (+ `<prompt>_base.jpg` / `<prompt>_steered.jpg`).
Honest verdict: on this 4-step distilled model with these small SAEs, "any prompt gets the Golden Gate" is true for
open outdoor scenes with a horizon (beach, river, desert, lake, castle) and false for close-ups, interiors and faces,
where the feature can only add its material. The remaining levers are an undistilled model with 20–50 steps and a
larger, longer-trained SAE.

**The dog neuron, same treatment** (`sdxl-demo-L18`, SDXL-Turbo L18 N2683 at −2, twelve demo prompts, 512 px,
full VAE, 49 s): this one *is* Golden Gate Claude. Every prompt becomes a dog that keeps the prompt's world: the
astronaut is a dog in a space helmet, the kitchen a dog made of marble, the Tokyo street a dog in neon, the library
a dog with bookshelf fur, the portrait a grey-bearded dog, the forest a dog with snow on its nose, the beach dog a
dog portrait. Mass 81, lens "a dog +20.9, an animal +15.5", 100 % above the random controls.
`gallery/dog/gallery.jpg`. N300 at −2 is the same for anime children; N976 at +2 turns every prompt into an aerial
view (its "Golden Gate +20" lens is CLIP reading aerial maps as bridges; `gallery/n976p`).

## 2026-11-09 — the explorer grows a model map, a feature panel and connections

Lyra's explorer shows the whole model (layers in perspective, one point per neuron) and, per neuron, its 24
strongest direct-path links up and down. Ours now does the same for what we have, offline, no pod.

- **Model map** (`/api/model_map`): every layer of SDXL-Turbo (70), FLUX.1 (57) or FLUX 3 Klein (33) as an oblique
  sheet; swept layers coloured by lens mass, the five FLUX SAE layers coloured by each feature's Golden Gate z over
  the corpus, the rest grey. Click a layer to drop into the face-on map.
- **Feature panel** (`/api/sae_units`, `/api/feature`): the 22 SAE clamp runs now appear in the run picker. A feature
  shows its clamp grid, Golden Gate rank/AUROC, the prompts it fires hardest on, its top CLIP attributes (correlation
  of activation with the 93 attribute scores over the 1,704 images), mean activation on the bridge vs other bridges
  vs landmarks, every run that clamps it (alone or inside a family), and its connections.
- **Connections** (`explorer/build_connections.py`, CPU, 86 s, 58 MB cache): direct-path cosine between a feature's
  raw-space write direction and another layer's read directions (L21/L24/L26, the SAEs we have locally), and
  co-activation over the corpus (all five layers). Sanity check on 13287 @ L24: top gg z at its layer (2.58, next
  182, 9829, 12491 — the family); strongest direct reads from 14806 @ L21 (cos 0.16) and 12042 @ L21 (0.085),
  strongest direct writes to 7074 and 20945 @ L26 (0.14, 0.12), all Golden Gate features by their own lens;
  strongest co-activation partner at L21 is 18426 (r 0.97), the member the three-layer clamps used, so the family we
  picked by ranking is also the family the wiring picks. Direct-path cosines are small in absolute terms (median top
  link 0.08–0.11, max 0.53): most of what a feature reads arrives through attention and the MLPs in between, not
  straight down the stream, which is also why clamping one layer never fully propagates.
- SDXL gets the map but no links: a UNet's blocks have different widths and skip connections, so there is no
  shared stream to dot through; a cross-stage link would be invented.
- Live clamping of a feature on your own prompt is wired (`server.py --live --model flux`, `clamp_context`) but
  not exercised: there is no GPU here and the FLUX weights are on no pod now.

## 2026-11-10 — the generate view: watch a unit take over, step by step

Lyra's explorer lets you steer and read the result; ours now shows the picture forming. `sd_explorer/trace.py` wraps
the scheduler's `step` to keep, for every denoising step, the noisy latent the step started from and the model's
clean-image estimate (Euler-ancestral's `pred_original_sample` for SDXL-Turbo, `x_t − σ·v` for FLUX's flow
matching), decodes both with the tiny VAE, and records the chosen unit's activation map at every step. `/api/generate`
runs one prompt twice through the same noise, unit off and unit on (a neuron push or an SAE feature clamp), and the
**generate** view shows the two finals, a film strip of every step for both, a play button, a toggle between the
clean guess and the raw noisy latent, and a pink overlay of where the unit fires. Presets: Golden Gate (FLUX
feature 13287, band clamp), Dog, Fire & horror, Buzz Lightyear, Bird of prey, Anime child, Aerial view, Eye, Old
face, Brightness, Saturation.

New units, found by lens and confirmed by eye on the L18 sweep thumbnails (all SDXL-Turbo L18, dose −2):
N1505 turns every subject into **Buzz Lightyear** (the fruit bowl included); N1532 puts a **bird of prey** in the
frame (the portrait grows wings); N5097 is **fire and horror** (the car burns, the woman's face doubles and melts;
lens horror +10, war +10); N297 makes the frame a close-up **eye**; N3462 a weathered **old face**. L46 N1992
("cat" on the lens) is only a texture: dropped.

Live on an RTX PRO 6000 Blackwell (96 GB, pod `0dam26owiiukac`, secure, CUDA 13.2, $2.49/hr — the capacity read had
said $1.69; A100 and H100 were all out of stock), SDXL-Turbo and Shuttle 3 loaded together (31 GB): SDXL traces take
0.4 s for 4 steps at 512 px, FLUX 10 s. Through the trace you can see *when* a unit acts: the dog neuron and Buzz
are already in the model's first guess at step 1 (the subject is decided before any detail exists); the Golden Gate
feature's towers are also present in the first guess, standing in the river, and later steps only sharpen them.
The server is reached through `./pod.sh tunnel 8792` → http://localhost:8792 (`.claude/launch.json` entry
`explorer-pod`). Setup: `sweep/setup_demo_pod.sh`, start: `sweep/run_live.sh`, data: `./pod.sh push-demo`.

**Speed, same day.** Measured through the tunnel: every request costs ~0.6 s of round trip, a generate response was
800 KB, and FLUX spent 6 of its 7 s encoding a new prompt on the CPU (the sweeps park T5 there after prewarming).
Fixes: `SD_FLUX_T5_GPU=1` keeps T5 on the GPU (new prompt 6.7 s → 1.6 s, repeat 0.9 s), gzip on the server
(the FLUX model map 630 KB → 80 KB), step frames at 192 px and activation maps as 2-decimal fractions (generate
response 800 → 250–340 KB), `ssh -C` on the tunnel. Click-to-picture for an SDXL unit is now ~1.8 s, FLUX ~2.2 s;
what remains is the pod's distance.

## 2026-11-10 — the bridge hunt: every SDXL layer, is there a single Golden Gate neuron?

Design (`sweep/bridge_hunt.py`, `sweep/bridge_hunt.sh`, `sweep/hunt_summary.py`): the open question after the demo
pass was that only 4 of SDXL-Turbo's 70 layers had been swept in full. Steering all 332,800 neurons on several prompts
is ~80 GPU-hours, so the hunt is two-stage. Stage 1 scores every neuron in every layer from the weights alone: the
cosine between its down-projection column (what it writes into the residual stream) and that layer's Golden Gate
concept vector from the earlier contrast run, plus the strongest readers by contrast effect size; ~170 candidates
per layer. Stage 2 steers those at ±2 on three horizon scenes (lake, boat, desert) with 12 random-direction controls
per layer and the CLIP lens. Stage 3 ranks by the Golden Gate attribute shift as a z against the controls, keeping
only pushes with cos to the unsteered image ≥ 0.75. Ninth pod `ac21jnnr2yoqku` (RTX 4090 24 GB, secure, $0.89/hr).

**Results, stages 1–3 (RTX 4090, 09:11–10:01 UTC, ~50 min for 70 layers, 10,906 candidates steered).**

- *Stage 1 alone was informative.* Write alignment with the bridge concept vector peaks at L36 N77 (cos 0.34, where a
  random column scores ~0.03), then L34 N1805 (0.28), L36 N4338 (−0.26), L55, L0, L18 N695, L21 N968 (0.24).
- *The CLIP "Golden Gate" attribute is the wrong judge, twice over.* Ranked by it, the winners were horizontal
  ripple and stripe dials in d1/d2 (L3–L6; z up to 28 against the random controls) with no bridge in any image; after
  adding a texture penalty, the winners became fog-and-dusk dials in L65 (N413, N1298, N807, N622…), because CLIP's
  Golden Gate is "fog + orange + bridge" and fog alone trips it. Both are lens artefacts, not bridges.
- *Ranked by "a suspension bridge" instead, and checked by eye, SDXL-Turbo has a family of bridge-building neurons in
  d2.a1 (L17–L21), with a few in u0 (L42):* L17 N868 (+2) throws an **arch bridge** across the lake; L18 N695 (+2) a
  **steel girder bridge** over the boat; L21 N968 (+2) a **plank deck and pier** (and it is also the strongest
  reader, contrast d +2.8); L20 N4912 (−2) a **stone viaduct** with arches; L19 N1644 (−2) and N3841 (−2) **steel
  trusses**; L17 N4161 (+2) **pylons**; L20 N2203 (+2) a rope-and-plank deck; L42 N1946 (−2) wooden trusses; L36 N77
  (+2) a dock. Each writes a different *kind* of bridge; none writes suspension towers or International Orange, so none
  is the Golden Gate on its own. The region is the same mid-UNet block where the dog, Buzz and bird neurons live.
- Sheets: `sheets/hunt_susp_top.jpg` (the builders), `sheets/hunt_top.jpg` (the fog artefact), rankings in
  `runs/hunt_ranking*.json`, every candidate's rows in `runs/hunt-L*.jsonl`.

**Dose ladders and the two-neuron bridge (10:03–10:25 UTC).** On the twelve demo prompts:
- **L17 N868 at +2** is a *bridge neuron* in the Golden-Gate-Claude sense: every prompt gets a bridge (the castle
  becomes a bridge tower in fog, the dog runs under a big bridge, the portrait becomes a steel tower, the cat sits on
  a bridge railing); mean "suspension bridge" shift +4.9, cos to baseline 0.81; +3 and +4 are weaker, not stronger.
- **L18 N695 at +3** is a *suspension-bridge neuron*: the Tokyo street becomes a blue-lit suspension bridge over
  water, the castle and astronaut become suspension towers, the portrait a steel bridge; susp +4.5, cos 0.77.
- L21 N968 turns out to be posts, beams and scaffolding rather than bridges; L19 N3841, L42 N1946/N3219 are trusses
  that damage the scene (cos ≤ 0.65).
- **Together (`sweep/combo.py`, L17 N868 +2 and L18 N695 +2) they do what no single unit did: all twelve prompts
  become a suspension bridge**, towers and cables included, and the rowing boat sits under an *orange* one. The lens
  puts "the Golden Gate Bridge" at the top of the shift list on 7 of 12 prompts (astronaut +11.7, portrait +13.1,
  castle +8.5, street +7.3, library +7.6, forest +4.8) with cos 0.47–0.90. Adding the L68 red-tint dial (N2092 at
  +1) only adds glitch; the pair on its own is the result. Sheet: `sheets/combo_arch_susp.jpg`.

**Stage 4, the necessity test (261 candidates, 48 layers, 10:06–10:44 UTC; `sweep/hunt_ablate.sh`).** Pushed down on
three prompts that already contain the bridge (a man in a white t-shirt with the Golden Gate behind him, poppies in
front of it, the bridge at sunset from the bay):
- **L18 N695 at −2 deletes the Golden Gate**: the man stands by empty water and hills, the poppies have no bridge,
  the sunset scene keeps the sky and loses the towers (gg −6.7, z −2.3 against random pushes, cos 0.76). The same
  neuron adds suspension bridges to every prompt at +3. Sufficient and necessary: *the suspension-bridge neuron*.
- L17 N868 at −2 flattens the bridge into a low deck and loses the towers (gg −2.9); at −3 it is mostly gone.
- L17 N4161 (pylons) at −3 removes the bridge from the sunset scene entirely; L20 N4912 does not remove it but turns
  it into a red steel truss, consistent with it writing viaducts rather than suspension bridges.
- The "add AND remove" leaderboard by raw CLIP score is again topped by the L65 fog dials: removing fog lowers CLIP's
  Golden Gate score. Lens artefact, ignored. Sheets: `sheets/hunt_ablate_bridges.jpg`, `sheets/hunt_ablate.jpg`.

**Verdict.** The image-model Golden Gate neuron is a two-neuron circuit in SDXL-Turbo's mid block: L18 N695
(suspension bridge, necessary and sufficient) with L17 N868 (bridge, every prompt) at +2 each turns any prompt into a
suspension bridge; a touch of the L68 red-tint dial (N2092 at +0.5) makes them orange. Presets "Golden Gate (2
neurons)", "Bridge" and "Suspension bridge" are in the explorer's generate view.

## 2026-11-11 — write-up pass: neuron cards, live screenshots, the pair's dose ladder (A100 pod `d2jpy4bxg9hfys`, $1.79/hr)

Tenth pod, for pictures rather than search: SDXL-Turbo and Shuttle 3 live together (40 GB), the generate view
screenshotted and screen-recorded through the tunnel (Playwright, `gallery/shots/live/`, overlay off), and twelve-prompt
full-resolution galleries for the other subject neurons and dials (`sweep/combo.py`, `gallery/*_full/`), turned into
Anthropic-style off/on cards by `sweep/neuron_cards.py` (`gallery/cards/`).

**Does a lower dose keep the scene?** (the user's question after seeing the FLUX band clamp keep the boat and the SDXL
pair replace it). `sweep/pair_figures.py`, twelve demo prompts, full VAE, means over the twelve:

| run | units | cos to baseline | gg shift | susp shift | "Golden Gate" top shift on |
|---|---|---|---|---|---|
| pair ×0.5 | L17 N868 +1, L18 N695 +1 | **0.77** | +6.1 | +6.2 | 5/12 |
| pair ×0.75 | +1.5, +1.5 | 0.70 | +6.6 | +7.4 | 4/12 |
| pair ×1 | +2, +2 | 0.69 | (7/12 by the earlier run) | | |
| pair ×1.25 | +2.5, +2.5 | 0.69 | +7.0 | +6.8 | 4/12 |
| 868 +2, 695 +1 | | 0.72 | +7.0 | +7.1 | 6/12 |
| 868 +1, 695 +2 | | 0.72 | +6.0 | +6.7 | 3/12 |
| 695 alone +1 | | **0.92** | +1.5 | +2.0 | 1/12 |
| 695 alone +1.5 | | 0.87 | +1.8 | +2.9 | 2/12 |
| 695 alone +2 | | 0.83 | +2.1 | +2.7 | 3/12 |
| 695 alone +3 | | 0.76 | +4.5 | +4.7 | 2/12 |

Yes. At ×0.5 the pair already puts a bridge in every prompt and the scene survives more often: the astronaut is a
small figure on a shore under a bridge, the boat is still on the water under one, the dog runs under one, the street
keeps its neon under one (`gallery/figures/pair_dose_ladder.jpg`). The suspension neuron alone at +1 to +2 is the
FLUX-like setting: the boat, the dog, the Tokyo street and even the portrait keep their subject and gain a suspension
bridge behind it (`gallery/figures/susp_dose_ladder.jpg`); at +3 the subject gives way. Beyond ×1 the pair gains
nothing (the lens plateaus and similarity falls). Presets updated: "Golden Gate (2 neurons, gentle)" (+1, +1) and
"Suspension bridge" now defaults to +1.5. The bridge neuron (L17 N868) is what replaces the subject; the suspension
neuron adds to it.

**The assistant, exercised (same pod, key added by the user).** Pushed `runs/{L10,L18,L46,L68,focus-L17,focus-L18}` and the
L18/L68/focus thumbs (rsync crawls at ~50 files/s; a tar pipe over ssh ran at ~2 MB/s, 15 min for L18+L68), then asked
three Lyra-style questions and recorded the session (`gallery/shots/assistant/`): "find a dog neuron" → searched the
sweeps, steered L18 N2683 live on "a cozy living room with a sofa" at −1/−1.5/−2, opened it, listed runners-up;
"find a cat neuron" → tested five candidates live, none draws a cat, said so with a table; "find a neuron that makes
images black and white" → L46 N1987 (+), tested and named, with the day-one caveat. Pod `d2jpy4bxg9hfys` terminated
after ~2.5 h (~$4.50); list-pods shows only the user's two unrelated EXITED pods.

## 2026-11-11 — the saved data, mined for examples (no pod)

`sweep/more_figures.py`: tier sheets for L10/L18/L46/L68 (the top unshown neurons per layer, off | on from the sweep
thumbnails, labels from `runs/L*.units.jsonl`), twelve-prompt cards for every gallery that had none (dog, aerial both
ways, bridge neuron alone, susp +1.5 / +3, pair ×1 / ×0.5, pair + red dial, FLUX feature), the necessity panel
(off / 695 −2 / 695 −3 / pair −2 on ggshirt, ggflower, ggsunset), six-prompt pair grids for fourteen settings, single
pairs for the gentle pair and susp +1.5, and three-row highlight cards (off / −2 / +2).

**The next tier of L18, named from the thumbnails:** N568 wine barrels (−2; lens only says "a sunset −15, flowers −12",
no barrel word), N4168 lips/mouth (+2), N1480 airliner (−2), N508 helicopter (+2), N2298 bare torso (−2), N216
Renaissance portrait (+2), N3704 children's-book cartoon (+2), N1172 carved sign with the subject's name (+2), N1622
vintage advertisement page (+2), N2356 web page (−2), N3641 old-master portrait (−2), N938 Manhattan from above (−2).
L10: N206 comic halftone (−2), N4745 pixel cartoon (+2, cos 0.78). L46: N1498 isometric cubes (+2), N3126 pencil
sketch (−2), N1921 carved text (−2). L68: N526 acid green (+2), N953 magenta (+2), N1223 blue night (+2, cos 0.81).

Doc: ~96 new figures across "Neuron cards" (next tier, six-more-prompts grids, more dials, layers 10 and 46, the
bridge circuit's twelve-prompt cards, the necessity panel, the eleven runner-up focus sheets, three orange attempts),
"Dose", "The feature" (twelve-prompt card, the six honest failures, four ladders, six other features' maps, the
concept vector at two layer sets), "Moving to FLUX" (four style neurons), "Also tried: Klein" (three ladders); Details
tab: per-dose tables for the ten featured L18 neurons and six L68 dials, null table, necessity ranking, suspension
ranking; full account: Day 9.

**Charts and diagrams in the doc (same day).** Ten drawn widgets, numbers from the run files: circuit diagram (868 → 695 → optional 2092; FLUX lane with f13287), hunt funnel (332,800 → 10,906 → 13 → 2 → 1), per-layer hunt map (max susp z per layer; L17–23 cluster, L42 trusses, L65/67 fog), L18 dose-response (nine neurons vs control 16), L18 null histogram (1,861 of 5,120 above 16.0), hunt scatter (shift vs similarity, 695 ninth), pair/susp dose ladder with the FLUX 0.85–0.90 band, brightness plateau (N151/N294, −6…+6, 0.9/0.1 limits), f13287 selectivity (37.7/3.9/1.6), share above controls per layer. Widget ids in scratchpad blobs.txt.

**Editorial pass on the doc (same evening).** Story tab reordered into: intro + reading guide → one-minute story → question → findings → other things → background (image models, diffusion, DiT, what a sweep is) → method → SDXL results → FLUX → SAE → the feature → the bridge hunt (no longer an "Update" at the end) → Klein → the lesson (amendment rewritten) → neuron-card gallery → the tool → **Limitations** (9 points) → **Open questions** → **Related work** → **Reproducing this** → Acknowledgements → Glossary (+10 terms). Fixed: f13287 mean activation stated consistently (37.7/3.9/1.6 first SAE; 30.0/3.4/1.4 rebuilt), "all 332,800 checked" → "screened, 10,906 steered", assistant no longer "not yet exercised", dateline 4–11 Nov.

**Verification pass on the doc.** All three tabs read end to end. Fixed: the necessity z for L18 N695 at −2 is −6.5 against the 12 same-dose random pushes (recomputed from `runs/ablate-L18.jsonl`: shift −6.71, random mean −1.70, sd 0.77); the earlier "z −2.3" in this log and in two doc paragraphs was wrong and is corrected in the doc. Machine count is eleven incl. the 5-minute H100 (5090, A100c, RTX PRO 6000, H100, H100 5-min, A100 rebuild, A100 Klein 9.5 h $17, A100 demo, RTX PRO 6000 live $2.49/hr, RTX 4090 hunt, A100 write-up 2.5 h $4.50); recorded cost ≈ $60, estimated total $90–100; Details timeline table now has all eleven. Related-work citations verified on arXiv/WACV/ECCV/COLM listings (2410.22366, 2311.12092, 2402.17762, 2305.08891, transformer-circuits scaling-monosemanticity). Full account: "Day 9" → "Day 8, continued"; header count updated.

**Reader walkthroughs of the final story tab (rev 73, all 543 blocks read).**
*As a beginner, in the reading guide's order:* the one-minute story, question and findings read cleanly; the background sections arrive before any number; the first "z" a beginner meets (method list) lacked a gloss → added; "the lens by 16 points" → "a lens mass of at most 16"; the dog figure's alt text said "pushed up" where the text says −2 → fixed; the number of denoising passes was stated three ways (SDXL "four passes" in the background, "one step" in the harness) → now consistent: one pass in the sweeps, four in the live view, FLUX four, Klein sixteen; Klein was missing from "two image models were used" → added as the third, visited briefly.
*As a researcher, claim by claim:* findings 1–7 each have a figure and numbers that match the Details tab; f13287 selectivity 37.7/3.9/1.6 consistent in four places; necessity z −6.5 consistent after the earlier fix; the "every neuron" claim now distinguishes steered (FLUX d6, four SDXL layers) from screened (10,906 of 332,800); "neon in two (1496, 2237)" named a neuron not on the sheet → "neon in one (1496)"; the SAE layer count ("three layers") now says three single-stream layers, five in the first session; the dose-ladder table's blank cells for the +2,+2 row are a genuine gap in the run (gg/susp means were not recorded for that run) and are left blank rather than filled. Related work, limitations, open questions, reproduction and acknowledgements read back in full. Eleven edits made in one update.

**Correction (verification pass).** The "Golden Gate is CLIP's top shift on 7 of 12 prompts" claim for the pair at +2,+2 (10 Nov entry above) does not match the saved rows: in `runs/combo-arch_susp.jsonl` it is top on 5 of 12 (astronaut, castle, street, portrait, forest) and second on 4 more (kitchen, dog, library, boat); library's top is "a suspension bridge" (+8.6 vs +7.6). Means for that run: cos 0.68, gg +6.5, susp +7.7. The doc, the pair cards and the circuit diagram now say 5 of 12 (second on 4 more). Also: lens mass is the sum of the five largest |z| (rank.py), not the sum over the whole vocabulary; two captions and one Details sentence said the latter and are corrected.

**Details and Full account re-read in full after the edits (Details rev 5, 222 blocks; Full account rev 9, 193 blocks), checked claim by claim against the story.** Fixed: the blank full-strength row of the dose table in both tabs (now +6.5 / +7.7 / 5 of 12, with a note that these come from the 512-px run of the same setting); the "7 of 12" count in the Details bridge table, the Full account's Day 7 paragraph, the story's findings list, the story's dose table, the circuit diagram and both pair cards; the lens-mass definition in the Details per-dose intro, the story glossary and two chart labels (it is the sum of the five largest |z|, as `rank.py` computes it, not the sum over the vocabulary); "the one layer swept completely" now says the one FLUX layer; the SAE recording paragraph now mentions the two double-stream blocks of the first session; "a tenth pod" is now the eleventh machine (tenth that did work).

## Day 8, continued: review fixes (11 November 2026, evening)

Applied the beginner (38 items) and researcher (46 items) reviews to the write-up. Corrections of substance, all three tabs:

- Brightness dial direction was backwards in the SDXL table and the figure alt: pushed DOWN brightens (lum 0.88 at −2), pushed UP darkens (0.09 at +2); saturation moves with it, so "nothing else changes" was removed. Plateau: 0.82–0.87 / 0.09–0.14 beyond ±2.
- Orange: the layer 68 red-tint dial at +0.5 tints two or three of the twelve bridges orange and glitches the rest; "makes them all orange" removed everywhere.
- "Necessary and sufficient" removed. N695 at −2 removes the bridge from the three prompts that ask for it (one seed each, a push not an ablation). The pair was never tested against other suspension bridges; said so in the findings, the answer, and a new limitation. "Circuit" = the two pushes add; no interaction test.
- Block naming: layers 4–23 are the second and deepest down block (checked against runs/*.units.jsonl names: d1 = 4 layers, d2 = 20, m = 10, u0 = 30, u1 = 6). "Middle"/"mid block" wording for L17–L21 removed; "layers 14 to 23" (a slip during the fix) corrected to 4 to 23. FLUX: s5 = layer 24 of 57, counted from s0; "fifth single-stream" removed; a block-naming paragraph added to "How a DiT works".
- Readers: L23 N3153 is the top contrast-search reader (d 4.4); L21 N968 the strongest reader among bridge builders (d 2.8); L18 N3520 is a lens "mover", relabelled. Readers were pushed directly only in SDXL-Turbo; "in every model" claims narrowed.
- Numbers labelled by prompt set and decoder (Units and settings paragraph in Details): per-dose table = z on four sweep prompts; hunt = raw CLIP shifts on three horizon prompts; dose tables = twelve prompts; dog mass 81 (twelve) vs 71 (four); clamp +0.047 (first session, lake) vs +0.018 (rebuilt mean).
- Shortfall 94,836 vs 98,304 stated as unaccounted for; 60% vs 20.7% (48 vs 128 controls) reconciled; "eighty GPU-hours" replaced with the arithmetic (two million images ≈ 25 h on the 4090 at 22/s); throughput: 86/s on the 5090, ~22 on the 4090, FLUX 2.3/s on 5090/A100 and 4.5 on H100.
- Klein: the layer-20 ladder (best writer, half a generic bridge feature) is now shown; caption no longer mentions attention; "the other two models" → FLUX.1.
- Feature family: ~30 features separate GG perfectly at s5; the four strongest (13287, 182, 2244, 12491) were clamped; 12491/22975 sit on a massive channel. Layer-12: 11256 is selective, 14127 is the generic bridge detector (19 vs 31). 4392 is a halftone grid.
- Added: nine limitations (CLIP selects and judges; 12 controls in the hunt; tiny decoder; SAE labels are prompt text; Shuttle 3 stand-in; no prompt baseline; clamp means ≈ random control; pair specificity untested; experiments not run), an open-questions item listing the referee's experiments, seven related-work entries marked "cited from memory", twenty glossary rows, a block-naming paragraph. Limitations/Open questions/Related work moved before the gallery; reading guide updated.
- Widgets: circuit diagram title (deepest down block), evidence lines shortened to fit and "Necessary:" reworded; funnel stage label and GPU-hours note updated.

Final text of all three tabs dumped to scratch review2/ and sent to two fresh reviewers (beginner, researcher) for a walkthrough of the final version.

### Second review round on the final text (same evening)

Two fresh reviewers read the final dumps (beginner 17 items, researcher 25). Applied: headline softened everywhere to "the closest thing to the Golden Gate" with the untested-specificity caveat (one-minute story, short answer, gallery caption, full-account verdict); L21 N968 qualified as the strongest reader *among the bridge builders* (top reader overall L23 N3153) in all three tabs and the card alt; "uniform clamps never give a bridge" corrected (towers in the lake at 2–2.5×, lattice elsewhere; the family at 0.75× uniform is the exception); the four clamped family features are "chosen for where they fire", not "the four strongest" (9829 outranks 12491 and 2244); N695 at +3 on the twelve prompts = GG +4.5, susp +4.7, sim 0.76 (from combo-susp3_full.jsonl; the Details cell had the GG column labelled as suspension); N868 at −2 z −0.8, mostly gone at −3 with no z; the SAE rebuild described as a same-seed rerun (not an independent replication) with the unexplained 20% drop; spare-copies and schedule-ceiling claims marked as readings/hypotheses (tiny-decoder clipping named as the alternative); "one block from the pixels" replaced (convolutions and decoder follow L68); Klein = 5 mode blocks with image/text sub-blocks + 28 single; d6 = seventh double-stream block; concept-vector depth claim narrowed to layers 16–20 (L23 failed, mid block untried); band-clamp similarity by prompt (0.86 lake, 0.92–0.95 car/woman); glossary fixes (null percentile 1.0 in files; write column = column of W_out; seed added); new limitations: no control for the pair and the pair tuned on its test prompts; no family-wide ablation; no random control for concept-vector pushes; no best-prompt random figure for the clamps. Lin et al. wording corrected (final training timestep); Kwon et al. h-space = mid-block output.
Final revs: story 206, details 31, full account 46.

### Full re-read of the finished text (same night)

All three tabs re-read end to end from fresh full views (story rev 206→212, 563 blocks; details rev 31→33, 209 blocks; full account rev 46→47, 184 blocks) after the second review round. Found and fixed on that read: the bold finding heading still said "is a two-neuron circuit"; "five papers above" pointed the wrong way; the glossary Dose row used a different unit wording from the body; the glossary SNR row kept "last step"; "full resolution" was undefined (it means the full VAE decoder at 512 px, confirmed from gallery/*_full image sizes and combo.py's default --size 512); the full account still called the SAE rebuild "a property of the model"; the timeline row conflated the 1,024-px FLUX demo with the 512-px SDXL galleries; the Details dial/destroyer cutoff sentence implied a single threshold. Then two fresh readers (beginner, researcher) were given the finished dumps for a verdict.

### Final verdict readers (same night)

Beginner verdict: presentable; six items applied (FLUX "any scene" → "any open scene with a horizon"; layer 68 placement; who judged the pictures; the dose-table splice wording; "ablation" defined in the glossary; the findings list is numbered on the live page). Researcher verdict on the pre-fix dump: not presentable on three wording claims, all applied: "any scene" (same fix, plus a limitation that the FLUX targets and masks were tuned on the reported prompts and that six of twelve demo prompts only get lattice); "none of the four swept layers paints the bridge" (N695 was in the layer-18 sweep data at +2, mass 26; the text now says it was recognised only in the hunt); "the only one that writes anything CLIP calls a landmark" (CLIP's landmark score is also moved by fog, ripples and aerial maps; reworded to "builds a landmark-like structure when looked at"). Also from that pass: layer 68 has one more transformer layer after it (not "only convolutions"); "subjects are decided in the first step" → "appear"; the "does not wire together" claim now gives the random-cosine baseline (~0.02 in 3,072 dims); the two-neuron-Golden-Gate name kept as a label; repository not public and the 93 lens descriptions are in sd_explorer/metrics.py. Final revs: story 231, details 34, full account 48. Dumps in scratch final2/ sent back to the researcher for a re-verdict.
Re-verdict on the corrected text: researcher "presentable to a researcher as an informal write-up", no blocking items (three notes: "landmark-like" sentence loosened to "the bridge neurons" since the viaduct neuron also builds one; the clamp equation is in the Details tab as a LaTeX block that the text export skips; the decoder of the 10 November pair run is not recorded). Beginner "presentable to a beginner". Final revs: story 232, details 34, full account 48.

### Full-read verdicts on the exact final revision (closing)

Dumped the exact final revisions (story 232 → edits → 245, details 34 → 37, full account 48 → 49) and had two fresh readers read every line of all three tabs with stated line coverage. Beginner round 1: not presentable (six items: the lesson said single neurons are poor handles for subjects against the twenty-plus subject neurons shown; the gentle pair "subject stays" vs "sometimes replaced"; "two checks confirm" overstated; two low-similarity rules in Details; "all seven swap the subject" included the aerial view; one ungrammatical sentence) — all applied. Researcher round 1: presentable, no blocking items; three near items applied (FLUX single-neuron coverage stated as ~13,000 of 700,000 in finding 1, the method and the limitations; late-step pushes split by model; random control similarity "not tabulated separately"; Klein layers 13/17/20 = s8/s12/s15). Round 2, both readers re-read every line of the new dumps: beginner "presentable to a beginner", no blocking items; researcher "presentable to a researcher as an informal write-up", no blocking items. Two leftover timeline phrases and one lesson word ("place" → "landmark") fixed after the verdicts. Final revisions: story 246, details 39, full account 49.
Frozen final revision (story 246, details 39, full account 49): dumped, diffed against the previously read copies (three phrases changed, full account byte-identical), and read in full, every line, by both verdict readers. Beginner: "presentable to a beginner", no blocking items (one non-blocking wording note: "52 blocks were not touched" in the Details timeline refers to neuron sweeps; the contrast search and autoencoders did read other layers). Researcher: "presentable to a researcher as an informal write-up", no blocking items. No edits were made after these reads. The write-up is frozen at these revisions.

### Two sections added at the user's request (same night)

"Using sparse autoencoders for mechanistic interpretability" after The lesson (why neurons are not the unit; what an SAE is; a seven-step recipe for an image model; what it bought; what it cost; how it compares with Golden Gate Claude; when to reach for it) and "Hardware and models" after Reproducing (a models table with Hub ids: stabilityai/sdxl-turbo + taesdxl, shuttleai/shuttle-3-diffusion + taef1, black-forest-labs/flux-3-image-klein, openai/clip-vit-base-patch32, the TopK SAEs, Claude; a hardware table of the eleven Runpod GPUs by type; cost and throughput; the M3 Max 36 GB MacBook Pro for everything else). Reading guide updated. Story rev 250.
