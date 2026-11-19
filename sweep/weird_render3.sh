#!/usr/bin/env bash
# Third pass: thirty more unpictured neurons with strong subject phrases, four sweep prompts, +-2.
cd /workspace/sd-neuron-explorer || exit 2
source /workspace/env.sh; source .venv/bin/activate
until grep -q 'RENDER2_DONE\|_EXIT=' /workspace/render2.log 2>/dev/null; do sleep 10; done
set -e
R="python sweep/steer_sweep.py --model sdxl --random-controls 0 --batch 8 --thumb 512 --full-vae"
$R --layer 18 --neurons 1429,4973,4359,517,4782,3204 --doses=-2,2 --run new3-L18
$R --layer 46 --neurons 4778,4324,185,2362,1245,2251,1387,856,1620,4029,3357,2404,1565,4429,2467,629,3535,445,2238,3532,4967,2063,2240 --doses=-2,2 --run new3-L46
echo RENDER3_DONE
