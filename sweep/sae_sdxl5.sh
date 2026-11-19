#!/usr/bin/env bash
# Background-only clamp (the subject's tokens left alone) and the L68 red-tint dial on top of the Golden Gate features.
cd /workspace/sd-neuron-explorer || exit 2
source /workspace/env.sh; source .venv/bin/activate
set -e
mkdir -p stats sheets gallery
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R() { [ -f "runs/${@: -1}.meta.json" ] && { echo "skip ${@: -1}"; return; }; python sweep/sae_steer.py "$@"; }
S="R --data sae/sdxl --full-vae --thumb 512 --batch 6 --random-features 0"
ALL=car,woman,fruit,lake,cat,kitchen,astronaut,breakfast,castle,dog,street,library,desert,portrait,boat,forest,elon,zuck
$S --layer 16,18,20 --features 547:14420:14108 --targets 1,1.25,1.5,2 --prompts $ALL --mask auto:0.4 --save-full gallery/sdxl-sae-3L-automask --run sdxl-sae-3L-automask
$S --layer 16,18,20 --features 547:14420:14108 --targets 1,1.25,1.5,2 --prompts $ALL --mask auto:0.4 --push 68:2092:0.5 --save-full gallery/sdxl-sae-3L-automask-red --run sdxl-sae-3L-automask-red
$S --layer 18 --features 14420 --targets 2,2.5,3,4 --prompts $ALL --mask auto:0.4 --save-full gallery/sdxl-sae-L18-automask --run sdxl-sae-L18-automask
$S --layer 16,18,20 --features 547:14420:14108 --targets 0.75,1,1.25 --prompts $ALL --push 68:2092:0.5 --save-full gallery/sdxl-sae-3L-red --run sdxl-sae-3L-red
echo CLAMP_DONE
