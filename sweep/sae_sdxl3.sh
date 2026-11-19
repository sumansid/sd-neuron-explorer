#!/usr/bin/env bash
# Golden Gate feature on the famous prompts and a finer dose ladder (the SAEs and corpus maxima are pushed, not rebuilt).
cd /workspace/sd-neuron-explorer || exit 2
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
source /workspace/env.sh; source .venv/bin/activate
set -e
mkdir -p stats sheets gallery
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="python sweep/sae_steer.py --data sae/sdxl --full-vae --thumb 512 --batch 6 --random-features 0"
$S --layer 18 --features 14420 --targets 1.5,1.75,2,2.25,2.5,3 --prompts elon,zuck --save-full gallery/sdxl-sae-L18-famous --run sdxl-sae-L18-famous
$S --layer 16,18,20 --features 547:14420:14108 --targets 0.75,1,1.25 --prompts elon,zuck --save-full gallery/sdxl-sae-L16+18+20-famous --run sdxl-sae-L16+18+20-famous
$S --layer 18 --features 14420 --targets 1.25,1.5,1.75,2,2.25,2.5 --prompts car,woman,fruit,lake,cat,kitchen,astronaut,breakfast,castle,dog,street,library,desert,portrait,boat,forest --save-full gallery/sdxl-sae-L18-fine --run sdxl-sae-L18-fine
echo CLAMP_DONE
