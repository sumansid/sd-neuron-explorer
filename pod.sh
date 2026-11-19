#!/usr/bin/env bash
# Thin wrapper around the pod's direct SSH: ./pod.sh ssh [cmd], ./pod.sh push (code up), ./pod.sh pull (results down).
# Edit POD_HOST / POD_PORT after a pod restart (direct SSH changes on every stop/start).
POD_HOST="${POD_HOST:?set POD_HOST to the pod's SSH address}"
POD_PORT="${POD_PORT:-22}"
KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -i "$KEY" -p "$POD_PORT")
cd "$(dirname "$0")"
case "${1:-ssh}" in
  ssh)   shift; "${SSH[@]}" "root@$POD_HOST" "$@" ;;
  push)  rsync -rltz --no-o --no-g -e "${SSH[*]}" --exclude .venv --exclude runs --exclude thumbs --exclude stats --exclude __pycache__ --exclude sae --exclude gallery --exclude sheets --exclude "stats-*" --exclude "*.emb.npy" --exclude explorer/cache --exclude '*.pyc' ./ "root@$POD_HOST:/workspace/sd-neuron-explorer/" ;;
  pull)  for d in runs stats thumbs; do rsync -rltz --no-o --no-g -e "${SSH[*]}" "root@$POD_HOST:/workspace/sd-neuron-explorer/$d/" "$d/" 2>/dev/null || true; done ;;
  push-demo)  # what the live explorer needs besides the code: the FLUX L24 SAE, its corpus maxima, the connection cache, stats, SAE runs
         "${SSH[@]}" "root@$POD_HOST" "mkdir -p /workspace/sd-neuron-explorer/{sae/flux,sae/flux2,explorer/cache,stats,runs,thumbs/{L18,L68,L46,flux-sae2-L24-band-high,flux-L24}}"
         rsync -rltz --no-o --no-g -e "${SSH[*]}" sae/flux2/sae_L24_fp16.pt "root@$POD_HOST:/workspace/sd-neuron-explorer/sae/flux2/"
         rsync -rltz --no-o --no-g -e "${SSH[*]}" sae/flux/feat_L24_max.npy sae/flux/meta.json sae/flux/attr.npy "root@$POD_HOST:/workspace/sd-neuron-explorer/sae/flux/"
         rsync -rltz --no-o --no-g -e "${SSH[*]}" sae/flux2/find_L24_gg.json "root@$POD_HOST:/workspace/sd-neuron-explorer/sae/flux2/"
         rsync -rltz --no-o --no-g -e "${SSH[*]}" explorer/cache/ "root@$POD_HOST:/workspace/sd-neuron-explorer/explorer/cache/"
         rsync -rltz --no-o --no-g -e "${SSH[*]}" stats/ "root@$POD_HOST:/workspace/sd-neuron-explorer/stats/"
         rsync -rltz --no-o --no-g -e "${SSH[*]}" --include 'L18.*' --include 'L68.*' --include 'L46.*' --include 'flux-sae2-L24-*' --include 'flux-L24.*' --exclude '*' runs/ "root@$POD_HOST:/workspace/sd-neuron-explorer/runs/"
         for d in L18 L68 L46 flux-sae2-L24-band-high flux-L24; do rsync -rltz --no-o --no-g -e "${SSH[*]}" "thumbs/$d/" "root@$POD_HOST:/workspace/sd-neuron-explorer/thumbs/$d/"; done ;;
  tunnel) shift; "${SSH[@]}" -C -N -L "${1:-8792}:localhost:8791" "root@$POD_HOST" ;;
  *) echo "usage: $0 ssh [cmd] | push | pull | push-demo | tunnel [localport]"; exit 1 ;;
esac
