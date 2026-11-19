#!/usr/bin/env bash
# Golden Gate demo grid on FLUX.1 (Shuttle 3) with the saved s2/s5/s7 SAEs: 1024 px, full VAE, twelve unrelated
# prompts plus car and lake. Variants: band clamp, feature family, late-step clamp, three layers. Prints _EXIT=<code>.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
until grep -q "_EXIT=" /workspace/setup.log 2>/dev/null; do sleep 20; done
grep -q "_EXIT=0" /workspace/setup.log || { echo "setup failed"; echo _EXIT=1; exit 1; }
source /workspace/env.sh; source /opt/venv/bin/activate
python - <<'PY'
import torch, os
for L in (21, 24, 26):
    src, dst = f"/data/sae/flux/sae_L{L}_fp16.pt", f"/data/sae/flux/sae_L{L}.pt"
    if not os.path.exists(dst):
        d = torch.load(src, map_location="cpu")
        torch.save({k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in d.items()}, dst)
        print("restored", dst)
PY
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
P="demo"; COMMON="--data sae/flux --prompts demo --size 1024 --full-vae --batch 6 --random-features 2 --thumb 256"
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287 --targets 1.5,2,2.5,3 --mask band:0.3:0.55 --run gg-demo-band --save-full full/gg-demo-band
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287+182+2244+12491 --targets 0.75,1,1.25,1.5 --run gg-demo-family --save-full full/gg-demo-family
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287 --targets 1.25,1.5,2,2.5 --from-step 1 --run gg-demo-late --save-full full/gg-demo-late
run python sweep/sae_steer.py $COMMON --layer 21,24,26 --features 18426:13287:13531 --targets 0.75,1,1.25,1.5 --from-step 1 --run gg-demo-3L-late --save-full full/gg-demo-3L-late
run python sweep/sae_steer.py $COMMON --layer 21,24,26 --features 18426:13287:13531 --targets 1.5,2,2.5,3 --mask band:0.3:0.55 --run gg-demo-3L-band --save-full full/gg-demo-3L-band
echo _EXIT=0
