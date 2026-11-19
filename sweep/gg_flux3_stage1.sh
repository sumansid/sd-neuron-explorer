#!/usr/bin/env bash
# FLUX 3 Klein, stage 1: per-neuron stats (for absmax doses), the Golden Gate contrast search (which layers read
# the bridge), and a dose-calibration ladder on random neurons of one single-stream layer. Prints _EXIT=<code>.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
source /workspace/env.sh; source /opt/venv/bin/activate
t0=$(date +%s)
run() { echo "== $*"; "$@" || { echo "failed: $*"; echo _EXIT=1; exit 1; }; echo "   done $(( $(date +%s) - t0 ))s"; }
run python sweep/stats.py --model flux3 --n 128 --batch 8
run python sweep/contrast.py --model flux3 --concept goldengate --n 64 --batch 8 --top 40
run python sweep/contrast.py --model flux3 --concept bw --n 64 --batch 8 --top 40
run python sweep/steer_sweep.py --model flux3 --layer 12 --neurons random:12 --doses=-64,-32,-16,-8,8,16,32,64 --dose-unit absmax --random-controls 6 --batch 8 --run flux3-calib-L12
run python sweep/rank.py flux3-calib-L12 --show 12
echo _EXIT=0
