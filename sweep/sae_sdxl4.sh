#!/usr/bin/env bash
# The three d2.a1 Golden Gate features clamped together on the twelve demo scenes, gentle targets, full size.
cd /workspace/sd-neuron-explorer || exit 2
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
source /workspace/env.sh; source .venv/bin/activate
set -e
mkdir -p stats sheets gallery
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sweep/sae_steer.py --data sae/sdxl --full-vae --thumb 512 --batch 6 --random-features 0 --layer 16,18,20 --features 547:14420:14108 --targets 0.75,1,1.25,1.5 --prompts demo --save-full gallery/sdxl-sae-L16+18+20-demo --run sdxl-sae-L16+18+20-demo
echo CLAMP_DONE
