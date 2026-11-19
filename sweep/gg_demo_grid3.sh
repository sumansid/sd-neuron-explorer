#!/usr/bin/env bash
# Demo grid 3: the s5 feature family (the four strongest Golden Gate features at L24) at gentle targets, uniform,
# banded and early-only. The family at 0.75x already put a bridge on the beach horizon in grid 1.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
until grep -q "_EXIT=" /workspace/grid2.log 2>/dev/null; do sleep 20; done
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
COMMON="--data sae/flux --prompts demo --size 1024 --full-vae --batch 6 --random-features 2 --thumb 256"
FAM="13287+182+2244+12491"
run python sweep/sae_steer.py $COMMON --layer 24 --features $FAM --targets 0.4,0.5,0.6,0.75 --run gg-demo-family-low --save-full full/gg-demo-family-low
run python sweep/sae_steer.py $COMMON --layer 24 --features $FAM --targets 0.75,1,1.5,2 --mask band:0.3:0.55 --run gg-demo-family-band --save-full full/gg-demo-family-band
run python sweep/sae_steer.py $COMMON --layer 24 --features $FAM --targets 1,1.5,2,3 --to-step 1 --run gg-demo-family-early1 --save-full full/gg-demo-family-early1
echo _EXIT=0
