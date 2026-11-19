#!/usr/bin/env bash
# Second batch of transformation neurons on the famous prompts (elon, zuck), 512 px full VAE.
cd /workspace/sd-neuron-explorer || exit 2
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
source /workspace/env.sh; source .venv/bin/activate
set -e
R="python sweep/steer_sweep.py --model sdxl --random-controls 0 --batch 8 --thumb 512 --full-vae --prompts elon,zuck"
$R --layer 18 --neurons 3771,4249,343,1567,3001,4034,4868,2987,3335,1226,3704,4096,2385,893,465,1650 --doses=-2,2 --run famous2-L18
$R --layer 46 --neurons 2384,888,3410,4422,3631,1691,417,3472,5017,4158,2258 --doses=-2,-1,1,2 --run famous2-L46
$R --layer 10 --neurons 4745,1419,3066 --doses=-2,2 --run famous2-L10
$R --layer 68 --neurons 953,66,91 --doses=-1,1 --run famous2-L68
echo RENDER_DONE
