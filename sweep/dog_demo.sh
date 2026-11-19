#!/usr/bin/env bash
# SDXL-Turbo dog-neuron demo: L18 N2683 (dog), N300 (anime child), N976 (aerial view) on the twelve demo prompts,
# full VAE, 512 px native, images kept at full size. Waits for the SDXL weights to be present.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
source /workspace/env.sh; source /opt/venv/bin/activate
until grep -q "sdxl ok" /workspace/sdxl_download.log 2>/dev/null && grep -q "_EXIT=" /workspace/grid3.log 2>/dev/null; do sleep 20; done  # the FLUX grids fill the GPU; run after them
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 )) s"; }
run python sweep/steer_sweep.py --model sdxl --layer 18 --neurons 2683,300,976 --doses=-2,-1.5,-1,1,2 --prompts demo --random-controls 4 --batch 8 --thumb 512 --full-vae --run sdxl-demo-L18
run python sweep/rank.py sdxl-demo-L18 --show 8
echo _EXIT=0
