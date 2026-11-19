#!/usr/bin/env bash
# Sweep one layer of SDXL-Turbo: push every neuron up and down, keep every picture, rank what changed.
#   ./sweep.sh 18                      # layer 18, doses -2,-1,1,2, four prompts, 128 random controls
#   ./sweep.sh 18 --doses=-1,1         # fewer doses, half the pictures
#   ./sweep.sh 46 --neurons 0-499      # part of a layer
# Anything after the layer number is passed to sweep/steer_sweep.py. Layers are 0 to 69.
# Layer 18 (5,120 neurons) takes about 22 minutes on an RTX 5090 and writes runs/L18.* and thumbs/L18/.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "run ./setup.sh first"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
LAYER="${1:?usage: ./sweep.sh LAYER [steer_sweep options]}"; shift || true

if [ ! -f "stats/sdxl-turbo_L${LAYER}.npz" ]; then
  echo "== layer statistics (once per model, a few minutes)"
  python sweep/stats.py --n 128
fi

echo "== sweeping layer ${LAYER}"
python sweep/steer_sweep.py --layer "$LAYER" --neurons all --doses=-2,-1,1,2 --random-controls 128 --run "L${LAYER}" "$@"

echo "== ranking"
python sweep/rank.py "L${LAYER}" --show 40

echo
echo "Done. Open the tool with ./explore.sh and pick run L${LAYER}."
