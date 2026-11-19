#!/usr/bin/env bash
# Clamp the SDXL-Turbo Golden Gate features found by sae_sdxl.sh: maps of where they fire, dose ladders on the four
# sweep prompts (with random-feature controls), the twelve demo scenes, and the three d2.a1 layers together.
cd /workspace/sd-neuron-explorer || exit 2
source /workspace/env.sh; source .venv/bin/activate
set -e
mkdir -p stats sheets gallery
[ -f stats/concept_saegg_sdxl.npz ] || python sweep/sae_find.py --data sae/sdxl --layer 18 --phase maps --features 14420,8210,1313 --export saegg_sdxl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="python sweep/sae_steer.py --data sae/sdxl --full-vae --thumb 512 --batch 6"
$S --layer 18 --features 14420 --targets 0,0.5,1,1.5,2,3 --prompts car,woman,fruit,lake,ggcar,ggbridge --save-full gallery/sdxl-sae-L18 --run sdxl-sae-L18-clamp
$S --layer 18 --features 14420 --targets 1,1.5,2,3 --prompts demo --random-features 0 --save-full gallery/sdxl-sae-L18-demo --run sdxl-sae-L18-demo
$S --layer 20 --features 14108 --targets 1,2,3 --prompts car,woman,fruit,lake --random-features 3 --run sdxl-sae-L20-clamp
$S --layer 16 --features 547 --targets 1,2,3 --prompts car,woman,fruit,lake --random-features 3 --run sdxl-sae-L16-clamp
$S --layer 16,18,20 --features 547:14420:14108 --targets 0.5,1,1.5,2 --prompts car,woman,fruit,lake --random-features 3 --save-full gallery/sdxl-sae-L16+18+20 --run sdxl-sae-L16+18+20-clamp
echo CLAMP_DONE
