#!/usr/bin/env bash
# SAE pass on SDXL-Turbo: collect the residual stream leaving six 1280-dim blocks over the SAE corpus, train one TopK
# SAE per layer, score every feature on every corpus image, rank against the Golden Gate tag.
cd /workspace/sd-neuron-explorer || exit 2
until grep -q SETUP_DONE /workspace/setup.log 2>/dev/null; do sleep 10; done
source /workspace/env.sh; source .venv/bin/activate
set -e
L=16,18,20,23,41,46
python sweep/sae_collect.py --model sdxl --layers $L --tokens 256 --seeds 6 --batch 16 --out sae/sdxl
for l in ${L//,/ }; do python sweep/sae_train.py --data sae/sdxl --layer $l --features 16384 --k 32 --steps 3000; done
python sweep/sae_find.py --data sae/sdxl --layer 18 --layers $L --phase score --batch 16
for l in ${L//,/ }; do python sweep/sae_find.py --data sae/sdxl --layer $l --phase rank --tag gg --top 24; done
echo SAE_DONE
