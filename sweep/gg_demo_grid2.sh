#!/usr/bin/env bash
# Demo grid 2: clamp only the EARLY denoising steps (composition is decided there), then let the model render
# the rest unclamped; and more steps (8) so the clamp has room to integrate. Prints _EXIT=<code>.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
until grep -q "_EXIT=" /workspace/grid.log 2>/dev/null; do sleep 20; done
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
COMMON="--data sae/flux --prompts demo --size 1024 --full-vae --batch 6 --random-features 2 --thumb 256"
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287 --targets 2,3,4,6 --to-step 1 --run gg-demo-early1 --save-full full/gg-demo-early1
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287 --targets 2,3,4 --to-step 2 --run gg-demo-early2 --save-full full/gg-demo-early2
run python sweep/sae_steer.py $COMMON --layer 21,24,26 --features 18426:13287:13531 --targets 1,1.5,2,3 --to-step 1 --run gg-demo-3L-early1 --save-full full/gg-demo-3L-early1
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287 --targets 1.5,2,3 --steps 8 --to-step 3 --run gg-demo-8step-early3 --save-full full/gg-demo-8step-early3
run python sweep/sae_steer.py $COMMON --layer 24 --features 13287 --targets 1.5,2,2.5 --steps 8 --mask band:0.3:0.55 --run gg-demo-8step-band --save-full full/gg-demo-8step-band
echo _EXIT=0
