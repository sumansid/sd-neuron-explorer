#!/usr/bin/env bash
# On the pod: start (or restart) the live explorer with both models in a tmux session, logging to /workspace/live.log.
# Pass ANTHROPIC_API_KEY through /workspace/env.sh if the sidebar assistant should work there (the user adds it).
set -euo pipefail
cd /workspace/sd-neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source /opt/venv/bin/activate
export SD_FLUX_T5_GPU=1  # keep the T5 text encoder on the GPU: new prompts encode in ms instead of ~6 s
tmux kill-session -t live 2>/dev/null || true
tmux new-session -d -s live "python explorer/server.py --live sdxl,flux --port 8791 2>&1 | tee /workspace/live.log; echo _EXIT=\$? >> /workspace/live.log"
echo "started; tail -f /workspace/live.log"
