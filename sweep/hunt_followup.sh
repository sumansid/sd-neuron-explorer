#!/usr/bin/env bash
# After the bridge hunt's stage 2/3: (a) a dose ladder for the bridge-building neurons on the twelve demo prompts,
# (b) the necessity test (sweep/hunt_ablate.sh) with those neurons added to the targets. Log: /workspace/followup.log
set -uo pipefail
cd /workspace/sd-neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh; source .venv/bin/activate
echo "RANKINGS $(date -u +%H:%M:%S)"
python sweep/hunt_summary.py --by score --top 16 --out runs/hunt_ranking.json --sheet sheets/hunt_top.jpg > runs/hunt_summary_gg.txt
python sweep/hunt_summary.py --by score --top 24 --attr "a suspension bridge" --out runs/hunt_ranking_susp.json --sheet sheets/hunt_susp_top.jpg > runs/hunt_summary_susp.txt
echo "FOCUS $(date -u +%H:%M:%S)"
# the bridge builders seen by eye in the stage-2 thumbnails (layer: neurons)
declare -A FOCUS=( [17]="868,2768,4161" [18]="695,709" [19]="3841" [20]="2203,4912" [21]="968,4763" [42]="3219,1946" [36]="77" )
for L in "${!FOCUS[@]}"; do
  python sweep/steer_sweep.py --layer "$L" --neurons "${FOCUS[$L]}" --prompts demo --doses " -4,-3,-2,2,3,4" --random-controls 8 --run "focus-L$L" --batch 24 --thumb 192 \
    && python sweep/rank.py "focus-L$L" --show 0 > /dev/null
  echo "FOCUS_DONE $L $(date -u +%H:%M:%S)"
done
echo "ABLATE $(date -u +%H:%M:%S)"
EXTRA_TARGETS='{"17":[868,2768,4161],"18":[695,709],"19":[3841],"20":[2203,4912],"21":[968,4763],"42":[3219,1946],"36":[77]}' TOP=120 READERS=40 bash sweep/hunt_ablate.sh > /workspace/ablate.log 2>&1
tail -40 /workspace/ablate.log
echo _EXIT=0
