#!/usr/bin/env bash
# FLUX 3 Klein, stage 2, unattended: a fine dose ladder, sample sweeps of four layers at the dose that keeps the
# image, the Golden Gate autoencoder on the three layers that read the bridge most (s8, s12, s15 = L13, L17, L20),
# feature ranking, and clamp sweeps of the top feature. Prints _EXIT=<code>.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
run python sweep/steer_sweep.py --model flux3 --layer 12 --neurons random:8 --doses=-4,-2,-1,1,2,4 --dose-unit absmax --random-controls 4 --batch 8 --run flux3-calib2-L12
python sweep/calib_summary.py flux3-calib2-L12
DOSE=$(python sweep/pick_dose.py flux3-calib2-L12 0.93); echo "sweep dose: +-$DOSE absmax"
for L in 3 8 17 30; do
  run python sweep/steer_sweep.py --model flux3 --layer $L --neurons random:128 --doses=-$DOSE,$DOSE --dose-unit absmax --random-controls 48 --batch 8 --run flux3-L$L
  run python sweep/rank.py flux3-L$L --show 15
done
run python sweep/sae_collect.py --model flux3 --layers 13,17,20 --tokens 48 --seeds 2 --out sae/flux3
for L in 13 17 20; do run python sweep/sae_train.py --data sae/flux3 --layer $L --features 24576 --k 32 --steps 3000; done
run python sweep/sae_find.py --data sae/flux3 --layer 17 --layers 13,17,20 --phase score
for L in 13 17 20; do run python sweep/sae_find.py --data sae/flux3 --layer $L --phase rank --tag gg --top 30; done
F13=$(python sweep/pick_feature.py sae/flux3/find_L13_gg.json); F17=$(python sweep/pick_feature.py sae/flux3/find_L17_gg.json); F20=$(python sweep/pick_feature.py sae/flux3/find_L20_gg.json)
echo "features: L13 $F13  L17 $F17  L20 $F20"
run python sweep/sae_find.py --data sae/flux3 --layer 17 --phase maps --features $F17 --export saegg3
run python sweep/sae_steer.py --data sae/flux3 --layer 17 --features $F17 --targets 0,0.5,1,1.5,2,3 --random-features 3 --batch 6 --run flux3-sae-L17-clamp
run python sweep/sae_steer.py --data sae/flux3 --layer 17 --features $F17 --targets 1,2,3,4 --mask band:0.3:0.55 --random-features 3 --batch 6 --run flux3-sae-L17-band
run python sweep/sae_steer.py --data sae/flux3 --layer 13,17,20 --features $F13:$F17:$F20 --targets 0.5,0.75,1,1.5 --random-features 3 --batch 6 --run flux3-sae-L13+17+20-clamp
echo _EXIT=0
