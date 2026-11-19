#!/usr/bin/env bash
# Rebuild the single-stream Golden Gate SAEs from scratch on a fresh pod: setup, collect block outputs for
# s2/s5/s7 (L21/L24/L26), train one TopK SAE per layer, score every feature on the corpus, rank for the "gg" tag.
# Run from /workspace/sd-neuron-explorer; prints _EXIT=<code> at the end so a waiter can poll the log.
cd /workspace/sd-neuron-explorer || { echo _EXIT=2; exit 2; }
t0=$(date +%s)
bash sweep/setup_flux_pod.sh > /workspace/setup.log 2>&1 || { echo "setup failed"; tail -5 /workspace/setup.log; echo _EXIT=1; exit 1; }
echo "setup done $(( $(date +%s) - t0 ))s"
source /workspace/env.sh; source /opt/venv/bin/activate
python sweep/sae_collect.py --model flux --layers 21,24,26 --tokens 192 --seeds 2 --out sae/flux > /workspace/collect.log 2>&1 || { echo "collect failed"; tail -5 /workspace/collect.log; echo _EXIT=1; exit 1; }
echo "collect done $(( $(date +%s) - t0 ))s"
for L in 21 24 26; do
  python sweep/sae_train.py --data sae/flux --layer $L --features 24576 --k 32 --steps 3000 > /workspace/train_L$L.log 2>&1 || { echo "train L$L failed"; tail -5 /workspace/train_L$L.log; echo _EXIT=1; exit 1; }
  tail -1 /workspace/train_L$L.log; echo "train L$L done $(( $(date +%s) - t0 ))s"
done
python sweep/sae_find.py --data sae/flux --layer 24 --layers 21,24,26 --phase score > /workspace/score.log 2>&1 || { echo "score failed"; tail -5 /workspace/score.log; echo _EXIT=1; exit 1; }
echo "score done $(( $(date +%s) - t0 ))s"
for L in 21 24 26; do
  python sweep/sae_find.py --data sae/flux --layer $L --phase rank --tag gg --top 30 > /workspace/rank_L$L.log 2>&1 || { echo "rank L$L failed"; echo _EXIT=1; exit 1; }
done
echo "rank done $(( $(date +%s) - t0 ))s"
echo _EXIT=0
