#!/usr/bin/env bash
# Second pass: the same odd neurons on the twelve demo scenes.
cd /workspace/sd-neuron-explorer || exit 2
source /workspace/env.sh; source .venv/bin/activate
until grep -q 'RENDER_DONE\|_EXIT=' /workspace/render.log 2>/dev/null; do sleep 10; done
set -e
R="python sweep/steer_sweep.py --model sdxl --random-controls 0 --batch 8 --thumb 512 --full-vae --prompts demo"
$R --layer 46 --neurons 5,948 --doses=-1,1 --run demo-weird-L46
$R --layer 18 --neurons 568,3826,2886,4168,2298 --doses=-2,2 --run demo-weird-L18
$R --layer 18 --neurons 1232,1446,69,3569,3255,2117 --doses=-2,2 --run demo-new-L18
$R --layer 46 --neurons 456,4630,4368,1996,223,3969 --doses=-2,-1,1,2 --run demo-new-L46
echo RENDER2_DONE
