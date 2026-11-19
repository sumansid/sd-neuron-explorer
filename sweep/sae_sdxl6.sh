#!/usr/bin/env bash
# The Golden Gate clamp on prompts that give the bridge room: office, window, desk, balcony variants of the famous prompts.
cd /workspace/sd-neuron-explorer || exit 2
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
source /workspace/env.sh; source .venv/bin/activate
set -e
mkdir -p stats sheets gallery
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="python sweep/sae_steer.py --data sae/sdxl --full-vae --thumb 512 --batch 6 --random-features 0"
P=elon,elonoffice,elonwindow,elondesk,elonbalcony,zuck,zuckoffice,zuckwindow
$S --layer 16,18,20 --features 547:14420:14108 --targets 1.5,2,2.5 --prompts $P --mask auto:0.4 --save-full gallery/sdxl-sae-3L-office-mask --run sdxl-sae-3L-office-mask
$S --layer 16,18,20 --features 547:14420:14108 --targets 0.75,1,1.25 --prompts $P --save-full gallery/sdxl-sae-3L-office --run sdxl-sae-3L-office
echo CLAMP_DONE
