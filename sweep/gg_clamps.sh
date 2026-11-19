#!/usr/bin/env bash
# Stronger Golden Gate clamps: bash sweep/gg_clamps.sh <f21> <f24> <f26>   (feature ids from sae_find rank)
# Uniform clamp past the 1.75x knee, a horizon band and the top half pushed hard, and the three-layer clamp
# above 1x. Serial, one process at a time; prints _EXIT=<code>.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
F21=$1; F24=$2; F26=$3
source /workspace/env.sh; source /opt/venv/bin/activate
run() { echo "== $*"; python sweep/sae_steer.py --data sae/flux "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; }
run --layer 24 --features "$F24" --targets 1.5,1.75,2,2.25,2.5,3 --random-features 3 --batch 12 --run flux-sae2-L24-clamp-high
run --layer 24 --features "$F24" --targets 2,2.5,3,4,5 --mask band:0.3:0.55 --random-features 3 --batch 12 --run flux-sae2-L24-band-high
run --layer 24 --features "$F24" --targets 2,2.5,3,4 --mask top:0.5 --random-features 3 --batch 12 --run flux-sae2-L24-top-high
run --layer 21,24,26 --features "$F21:$F24:$F26" --targets 1,1.25,1.5,1.75,2 --random-features 3 --batch 12 --run flux-sae2-L21+24+26-high
run --layer 21,24,26 --features "$F21:$F24:$F26" --targets 1.5,2,2.5,3 --mask band:0.3:0.55 --random-features 3 --batch 12 --run flux-sae2-L21+24+26-band-high
echo _EXIT=0
