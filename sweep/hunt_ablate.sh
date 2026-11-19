#!/usr/bin/env bash
# Stage 4 of the bridge hunt, the necessity test: take the best candidates from stage 3 (runs/hunt_ranking.json) and
# the strongest readers from the contrast search, push each one DOWN on three prompts that already contain the
# bridge (sweep/prompts.py GG_PROMPTS), and see whether the bridge disappears. A real "Golden Gate neuron" should
# both add the bridge when pushed up (stage 2) and remove it when pushed down (here).
#   TOP=150 READERS=50 bash sweep/hunt_ablate.sh      (log: /workspace/ablate.log)
set -uo pipefail
cd /workspace/sd-neuron-explorer
# shellcheck disable=SC1091
source /workspace/env.sh; source /opt/venv/bin/activate 2>/dev/null || source .venv/bin/activate
TOP="${TOP:-150}"; READERS="${READERS:-50}"
python - "$TOP" "$READERS" <<'PY'
import json, sys, collections, os
top, readers = int(sys.argv[1]), int(sys.argv[2])
by = collections.defaultdict(set)
for f in ("runs/hunt_ranking.json", "runs/hunt_ranking_susp.json"):
    if os.path.exists(f):
        for c in json.load(open(f))["ranked"][:top]:
            by[c["layer"]].add(c["neuron"])
for L, ns in json.loads(os.environ.get("EXTRA_TARGETS", "{}")).items():  # bridge builders seen by eye
    by[int(L)].update(ns)
for r in json.load(open("stats/contrast_goldengate.json"))["top"][:readers]:
    by[r["layer"]].add(r["neuron"])
json.dump({str(k): sorted(v) for k, v in sorted(by.items())}, open("runs/ablate_targets.json", "w"))
print("layers", len(by), "neurons", sum(len(v) for v in by.values()))
PY
echo "ABLATE $(date -u +%H:%M:%S)"
for L in $(python -c "import json; print(' '.join(json.load(open('runs/ablate_targets.json'))))"); do
  N=$(python -c "import json; print(','.join(map(str, json.load(open('runs/ablate_targets.json'))['$L'])))")
  python sweep/steer_sweep.py --layer "$L" --neurons "$N" --prompts ggshirt,ggflower,ggsunset --doses " -3,-2,2" --random-controls 12 --run "ablate-L$L" --batch 24 --thumb 160 \
    && python sweep/rank.py "ablate-L$L" --show 0 > /dev/null
  echo "LAYER_DONE $L $(date -u +%H:%M:%S)"
done
python sweep/hunt_ablate_summary.py --top 16 | tee runs/ablate_summary.txt
echo _EXIT=$?
