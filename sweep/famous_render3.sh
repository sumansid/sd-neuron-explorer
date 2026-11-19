#!/usr/bin/env bash
# Third batch of transformation neurons on the famous prompts (elon, zuck), 512 px full VAE.
cd /workspace/sd-neuron-explorer || exit 2
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
source /workspace/env.sh; source .venv/bin/activate
set -e
R="python sweep/steer_sweep.py --model sdxl --random-controls 0 --batch 8 --thumb 512 --full-vae --prompts elon,zuck"
$R --layer 18 --neurons 1342,1172,767,1625,2333,4165,2486,4126,4584,1529,1480,1711,935,148,4403,4836,3392,4874,3048,4148,707 --doses=-2,2 --run famous3-L18
$R --layer 46 --neurons 4564,249,1921,1569,3506,4628,4071 --doses=-2,-1,1,2 --run famous3-L46
$R --layer 68 --neurons 524,2295 --doses=-2,2 --run famous3-L68
$R --layer 18 --neurons 2356,225,2332,1622,625,2934,324,3254,4597,801,3719,3767,3487,3743 --doses=-2,-1,1,2 --run famous3b-L18
$R --layer 46 --neurons 1383,4899,1992 --doses=-2,-1,1,2 --run famous3b-L46
echo RENDER_DONE
