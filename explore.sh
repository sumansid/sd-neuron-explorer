#!/usr/bin/env bash
# Open the browser tool on the sweeps in runs/ and thumbs/.
#   ./explore.sh                 # browse results            http://localhost:8791
#   ./explore.sh --live          # also load SDXL-Turbo to steer any neuron on your own prompt (GPU)
#   ./explore.sh --live flux     # Shuttle 3 (FLUX.1 architecture) instead; --live sdxl,flux loads both (32 GB of GPU memory)
# The assistant panel needs ANTHROPIC_API_KEY in the environment.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { echo "run ./setup.sh first"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
exec python explorer/server.py "$@"
