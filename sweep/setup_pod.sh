#!/usr/bin/env bash
# Prepares a fresh Runpod PyTorch pod: uv, the project's dependencies, and the model weights (SDXL-Turbo, the tiny
# SDXL decoder, CLIP ViT-B/32). Everything lives under /workspace so a stopped pod restarts ready.
# Run from /workspace/sd-neuron-explorer.
set -euo pipefail
cat > /workspace/env.sh <<'ENV'
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/.uv-cache UV_PYTHON_INSTALL_DIR=/workspace/.uv-python
export UV_INSTALL_DIR=/workspace/bin PATH=/workspace/bin:$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
ENV
grep -q /workspace/env.sh ~/.bashrc || echo "source /workspace/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source /workspace/env.sh
command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux rsync)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install diffusers transformers accelerate safetensors numpy pillow fastapi uvicorn python-multipart hf_transfer
python - <<'PY' > /workspace/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sdxl-turbo", allow_patterns=["*fp16.safetensors", "*.json", "*.txt"])
snapshot_download("madebyollin/taesdxl", allow_patterns=["*.safetensors", "*.json"])
snapshot_download("openai/clip-vit-base-patch32", allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("weights ok")
PY
tail -1 /workspace/download.log
python -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.get_device_name())"
