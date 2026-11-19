#!/usr/bin/env bash
# Every neuron used in the clips, on the "famous" prompts (elon, zuck), 512 px full VAE.
cd /workspace/sd-neuron-explorer || exit 2
source /workspace/env.sh; source .venv/bin/activate
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
set -e
R="python sweep/steer_sweep.py --model sdxl --random-controls 0 --batch 8 --thumb 512 --full-vae --prompts elon,zuck"
$R --layer 18 --neurons 297,1505,1532,300,3462,5097,2683,976,2298,4168,3204,1232,4782,568,2886,3569,216,3641,938 --doses=-2,2 --run famous-L18
$R --layer 46 --neurons 1245,4630,2240,1620,456,2063,4429,5,3969,1565,2894,2174 --doses=-2,-1,1,2 --run famous-L46
echo RENDER_DONE
