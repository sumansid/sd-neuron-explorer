#!/usr/bin/env bash
# The bridge hunt, end to end, on a pod (run inside tmux; log in /workspace/hunt.log):
#   stage 1  shortlist ~170 candidate writers/readers per layer from the weights (sweep/bridge_hunt.py)
#   stage 2  steer every candidate in every one of the 70 layers at ±2 on three horizon scenes, 12 random-direction
#            controls per layer (sweep/steer_sweep.py), then rank each run for the explorer (sweep/rank.py)
#   stage 3  rank all candidates by Golden Gate lens z with the scene kept (sweep/hunt_summary.py) + contact sheet
# Resumable: steer_sweep skips rows already in runs/hunt-L<L>.jsonl.
set -uo pipefail
cd /workspace/sd-neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh; source /opt/venv/bin/activate 2>/dev/null || source .venv/bin/activate
MODEL="${MODEL:-sdxl}"
PROMPTS="${PROMPTS:-lake,boat,desert}"
if [ "$MODEL" = sdxl ]; then
  DOSES="${DOSES:-" -2,2"}"; UNIT=rel; LAYERS="${LAYERS:-$(seq 0 69)}"; SHORT=runs/hunt_shortlist.json; PREFIX=hunt-L; BATCH=24; SHEET=sheets/hunt_top.jpg
else
  DOSES="${DOSES:-" -32,32"}"; UNIT=absmax; LAYERS="${LAYERS:-$(seq 0 56)}"; SHORT=runs/hunt_${MODEL}_shortlist.json; PREFIX=hunt-${MODEL}-L; BATCH=8; SHEET=sheets/hunt_${MODEL}_top.jpg
fi
echo "STAGE1 $MODEL $(date -u +%H:%M:%S)"
[ -f "$SHORT" ] || python sweep/bridge_hunt.py --model "$MODEL" --top-write 120 --top-read 40 --out "$SHORT" || { echo _EXIT=1; exit 1; }
echo "STAGE2 $(date -u +%H:%M:%S)"
for L in $LAYERS; do
  N=$(python -c "import json; d=json.load(open('$SHORT')); print(','.join(map(str, d['layers'][$L]['neurons'])))")
  python sweep/steer_sweep.py --model "$MODEL" --layer "$L" --neurons "$N" --prompts "$PROMPTS" --doses "$DOSES" --dose-unit $UNIT --random-controls 12 --run "$PREFIX$L" --batch $BATCH --thumb 160 \
    && python sweep/rank.py "$PREFIX$L" --show 0 > /dev/null
  echo "LAYER_DONE $L $(date -u +%H:%M:%S)"
done
echo "STAGE3 $(date -u +%H:%M:%S)"
python sweep/hunt_summary.py --min-cos 0.75 --top 16 --prefix "$PREFIX" --shortlist "$SHORT" --sheet "$SHEET" | tee "runs/hunt_${MODEL}_summary.txt"
echo _EXIT=$?
