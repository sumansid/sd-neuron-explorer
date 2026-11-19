#!/usr/bin/env bash
# FLUX 3 Klein, stage 3: the Golden Gate autoencoder pass. The FLUX 3 adapter runs one prompt per pipeline call, so
# every script that reads hooks per batch row uses batches that hold one prompt: collect --batch 2 (its two seeds),
# score and clamps --batch 1. Prints _EXIT=<code>.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
rm -rf sae/flux3
run python sweep/sae_collect.py --model flux3 --layers 13,17,20 --tokens 48 --seeds 2 --batch 2 --out sae/flux3
for L in 13 17 20; do run python sweep/sae_train.py --data sae/flux3 --layer $L --features 24576 --k 32 --steps 3000; done
run python sweep/sae_find.py --data sae/flux3 --layer 17 --layers 13,17,20 --phase score --batch 1
for L in 13 17 20; do run python sweep/sae_find.py --data sae/flux3 --layer $L --phase rank --tag gg --top 30; done
F13=$(python sweep/pick_feature.py sae/flux3/find_L13_gg.json); F17=$(python sweep/pick_feature.py sae/flux3/find_L17_gg.json); F20=$(python sweep/pick_feature.py sae/flux3/find_L20_gg.json)
echo "features: L13 $F13  L17 $F17  L20 $F20"
run python sweep/sae_find.py --data sae/flux3 --layer 17 --phase maps --features $F17 --export saegg3 --batch 1
run python sweep/sae_steer.py --data sae/flux3 --layer 17 --features $F17 --targets 0,0.5,1,1.5,2,3 --random-features 3 --batch 1 --run flux3-sae-L17-clamp
run python sweep/sae_steer.py --data sae/flux3 --layer 17 --features $F17 --targets 1,2,3,4 --mask band:0.3:0.55 --random-features 3 --batch 1 --run flux3-sae-L17-band
run python sweep/sae_steer.py --data sae/flux3 --layer 13,17,20 --features $F13:$F17:$F20 --targets 0.5,0.75,1,1.5 --random-features 3 --batch 1 --run flux3-sae-L13+17+20-clamp
echo _EXIT=0
