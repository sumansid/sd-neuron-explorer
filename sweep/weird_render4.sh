#!/usr/bin/env bash
# Fourth pass: the best of the third pass on the twelve demo scenes.
cd /workspace/sd-neuron-explorer || exit 2
source /workspace/env.sh; source .venv/bin/activate
set -e
R="python sweep/steer_sweep.py --model sdxl --random-controls 0 --batch 8 --thumb 512 --full-vae --prompts demo"
$R --layer 18 --neurons 3204,4782,517 --doses=-2,2 --run demo-new3-L18
$R --layer 46 --neurons 1245,2240,2063,4429,1620,1565,4324 --doses=-2,2 --run demo-new3-L46
echo RENDER4_DONE
