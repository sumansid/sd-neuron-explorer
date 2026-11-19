#!/usr/bin/env bash
# Faster pod setup for SDXL-Turbo work: reuse the container image's own torch (runpod/pytorch:*-torch280-cu1281 ships
# torch 2.8 + CUDA 12.8) instead of downloading 4 GB of wheels. A stdlib venv with --system-site-packages lets pip see
# the system torch, so only the small libraries are fetched. Run from /workspace/sd-neuron-explorer.
set -euo pipefail
cat > /workspace/env.sh <<'ENV'
export HF_HOME=/workspace/hf PATH=/workspace/bin:$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
ENV
grep -q /workspace/env.sh ~/.bashrc || echo "source /workspace/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source /workspace/env.sh
command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux rsync)
rm -rf .venv
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import torch; print('system torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install -q --disable-pip-version-check diffusers transformers accelerate safetensors pillow fastapi uvicorn python-multipart hf_transfer
python - <<'PY' > /workspace/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sdxl-turbo", allow_patterns=["*fp16.safetensors", "*.json", "*.txt"])
snapshot_download("madebyollin/taesdxl", allow_patterns=["*.safetensors", "*.json"])
snapshot_download("openai/clip-vit-base-patch32", allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("weights ok")
PY
tail -1 /workspace/download.log
python -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.get_device_name())"
echo SETUP_DONE
