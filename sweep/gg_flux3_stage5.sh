#!/usr/bin/env bash
# FLUX 3 Klein, stage 5: push the late-layer feature (s15 / L20 f20164), the one whose clamp kept the scene and
# raised the Golden Gate lens, harder and in a band, and pair it with the s8 feature.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
run python sweep/sae_steer.py --data sae/flux3 --layer 20 --features 20164 --targets 2.5,3,4 --random-features 2 --batch 1 --run flux3-sae-L20-high
run python sweep/sae_steer.py --data sae/flux3 --layer 20 --features 20164 --targets 2,3,4,6 --mask band:0.3:0.55 --random-features 2 --batch 1 --run flux3-sae-L20-band
run python sweep/sae_steer.py --data sae/flux3 --layer 13,20 --features 23562:20164 --targets 1,1.5,2 --random-features 2 --batch 1 --run flux3-sae-L13+20-clamp
echo _EXIT=0
