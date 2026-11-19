#!/usr/bin/env bash
# One-time setup: a virtual environment, the dependencies, and the model weights.
#   ./setup.sh
# Needs Python 3.10+ and, for sweeps and live steering, a GPU with 12 GB or more.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null; then
  [ -d .venv ] || uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install torch --torch-backend=auto
  uv pip install -e . hf_transfer
else
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -q --upgrade pip
  pip install -q torch
  pip install -q -e . hf_transfer
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
from huggingface_hub import snapshot_download
print("downloading SDXL-Turbo, the small SDXL decoder and CLIP ViT-B/32 (about 7 GB, once) ...")
snapshot_download("stabilityai/sdxl-turbo", allow_patterns=["*fp16.safetensors", "*.json", "*.txt"])
snapshot_download("madebyollin/taesdxl", allow_patterns=["*.safetensors", "*.json"])
snapshot_download("openai/clip-vit-base-patch32", allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("weights ok")
PY
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo
echo "Done. Next:  ./sweep.sh 18      (sweep one layer)     ./explore.sh      (open the browser tool)"
