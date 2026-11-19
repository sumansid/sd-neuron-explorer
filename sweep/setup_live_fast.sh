#!/usr/bin/env bash
# Fast setup for the live explorer demo (SDXL-Turbo + FLUX/Shuttle 3 together): reuse the image's own torch (2.8+cu128)
# in a --system-site-packages venv, pip only the small libraries, weights on the container disk (/flux-hf).
# Run from /workspace/sd-neuron-explorer.  Log: /workspace/setup.log
set -euo pipefail
cat > /workspace/env.sh <<'ENV'
export HF_HOME=/flux-hf PATH=/workspace/bin:$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
ENV
grep -q /workspace/env.sh ~/.bashrc || echo "source /workspace/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source /workspace/env.sh
command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux rsync)
rm -rf .venv; python3 -m venv --system-site-packages .venv; ln -sfn "$PWD/.venv" /opt/venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install -q --disable-pip-version-check diffusers transformers accelerate safetensors pillow fastapi uvicorn python-multipart hf_transfer sentencepiece protobuf anthropic
mkdir -p /flux-hf
python - <<'PY' > /workspace/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sdxl-turbo", allow_patterns=["*fp16.safetensors", "*.json", "*.txt"])
snapshot_download("madebyollin/taesdxl", allow_patterns=["*.safetensors", "*.json"])
snapshot_download("openai/clip-vit-base-patch32", allow_patterns=["*.safetensors", "*.json", "*.txt"])
snapshot_download("madebyollin/taef1", allow_patterns=["*.safetensors", "*.json"])
snapshot_download("shuttleai/shuttle-3-diffusion", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"], ignore_patterns=["*.gguf", "*fp8*", "*-nf4*"])
print("weights ok")
PY
tail -1 /workspace/download.log; du -sh /flux-hf
python -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.get_device_name())"
echo SETUP_DONE
