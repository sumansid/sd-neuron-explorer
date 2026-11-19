#!/usr/bin/env bash
# FLUX 3 Klein, stage 4: finer clamp ladders between "nothing" (1x) and "lattice" (1.5x), per layer and banded.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
run python sweep/sae_steer.py --data sae/flux3 --layer 17 --features 13555 --targets 1.1,1.2,1.3,1.4 --random-features 3 --batch 1 --run flux3-sae-L17-fine
run python sweep/sae_steer.py --data sae/flux3 --layer 17 --features 13555 --targets 1.25,1.5,1.75 --mask band:0.3:0.55 --random-features 3 --batch 1 --run flux3-sae-L17-band-fine
run python sweep/sae_steer.py --data sae/flux3 --layer 13 --features 23562 --targets 1,1.25,1.5,2 --random-features 2 --batch 1 --run flux3-sae-L13-clamp
run python sweep/sae_steer.py --data sae/flux3 --layer 20 --features 20164 --targets 1,1.25,1.5,2 --random-features 2 --batch 1 --run flux3-sae-L20-clamp
run python sweep/sae_find.py --data sae/flux3 --layer 13 --phase maps --features 23562 --batch 1
run python sweep/sae_find.py --data sae/flux3 --layer 20 --phase maps --features 20164 --batch 1
echo _EXIT=0
