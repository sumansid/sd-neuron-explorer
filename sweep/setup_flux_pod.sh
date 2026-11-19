#!/usr/bin/env bash
# Fresh Runpod PyTorch pod for the FLUX (Shuttle 3) runs: uv, deps, and the weights on the container disk
# (/flux-hf; ~35 GB, lost on restart). Run from /workspace/sd-neuron-explorer.
set -euo pipefail
cat > /workspace/env.sh <<'ENV'
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
uv pip install torch --torch-backend=auto
uv pip install diffusers transformers accelerate safetensors numpy pillow fastapi uvicorn python-multipart hf_transfer sentencepiece protobuf
mkdir -p /flux-hf
python - <<'PY' > /workspace/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("shuttleai/shuttle-3-diffusion", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
                  ignore_patterns=["*.gguf", "*fp8*", "*-nf4*"])
snapshot_download("madebyollin/taef1", allow_patterns=["*.safetensors", "*.json"])
snapshot_download("openai/clip-vit-base-patch32", allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("weights ok")
PY
tail -1 /workspace/download.log
du -sh /flux-hf
python -c "import torch, diffusers, transformers; print('torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.get_device_name())"
echo SETUP_DONE
