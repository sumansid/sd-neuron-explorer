#!/usr/bin/env bash
# Fresh Runpod PyTorch pod for FLUX 3 Image Klein: uv, diffusers from git (modular pipeline), Qwen3-VL text
# encoder support, and the weights on the container disk. The repo is gated: /workspace/env.sh must export
# HF_TOKEN (the user adds it) before the download step; the script stops with WAITING_FOR_TOKEN otherwise.
# Run from /workspace/sd-neuron-explorer.
set -euo pipefail
touch /workspace/env.sh
grep -q HF_HOME /workspace/env.sh || cat >> /workspace/env.sh <<'ENV'
export HF_HOME=/flux-hf UV_CACHE_DIR=/opt/.uv-cache UV_PYTHON_INSTALL_DIR=/opt/.uv-python
export UV_INSTALL_DIR=/opt/bin PATH=/opt/bin:$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
ENV
grep -q /workspace/env.sh ~/.bashrc || echo "source /workspace/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source /workspace/env.sh
command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux rsync)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d /opt/venv ] || uv venv --python 3.12 /opt/venv; ln -sfn /opt/venv .venv; mkdir -p /data/sae; ln -sfn /data/sae sae
# shellcheck disable=SC1091
source /opt/venv/bin/activate
uv pip install torch torchvision --torch-backend=auto
uv pip install "git+https://github.com/huggingface/diffusers.git" "transformers>=4.57" accelerate safetensors numpy pillow fastapi uvicorn python-multipart hf_transfer sentencepiece protobuf
python -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.get_device_name())"
echo DEPS_DONE
if [ -z "${HF_TOKEN:-}" ]; then echo WAITING_FOR_TOKEN; exit 3; fi
mkdir -p /flux-hf
python - <<'PY' > /workspace/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/flux-3-image-klein", ignore_patterns=["variants/*", "*.gguf", "*.png", "*.jpg"])
snapshot_download("openai/clip-vit-base-patch32", allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("weights ok")
PY
tail -1 /workspace/download.log
du -sh /flux-hf
echo SETUP_DONE
