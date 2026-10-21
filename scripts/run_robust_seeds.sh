#!/usr/bin/env bash
# ============================================================
# Robustness under sign-flip (model-poisoning) attack, multi-seed.
# Reproduces Fig.6 / terminal-F1 table with the ACTUAL implemented attack
# (robust_experiment.py --attack signflip, scale=-4, amplified negation of
# the model update; labels are NOT modified).
#
# Configs per mode (IID / non-IID), matching the manuscript:
#   FedAvg p=0.0 clean ; FedAvg p=0.1 ; FedAvg p=0.2
#   Trimmed p=0.1 beta=0.1 ; Trimmed p=0.2 beta=0.2
# Common budget R=60, le=1. Seeds 0 1 2.
#
# Each (mode, config) is an independent job group that loops over seeds;
# the 10 groups are distributed round-robin across GPUs.
# Output: revision_id/unified/robust_curves/{mode}_{agg}_p{p}_signflip_b{b}_s{seed}.csv
#
# Usage:
#   bash run_robust_seeds.sh 0 1 2 3
#   bash run_robust_seeds.sh 0 1
#   SEEDS="0 1 2" bash run_robust_seeds.sh 0 1 2 3
# ============================================================
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1 2 3)
NG=${#GPUS[@]}
SEEDS=${SEEDS:-"0 1 2"}
MODES=(iid noniid)
# rows: agg poison beta
CONFIGS=(
  "fedavg 0.0 0.1"
  "fedavg 0.1 0.1"
  "fedavg 0.2 0.1"
  "trimmed 0.1 0.1"
  "trimmed 0.2 0.2"
)
mkdir -p logs_unified

echo "=== sign-flip robustness, seeds=[$SEEDS], GPUs=${GPUS[*]} ==="
g=0
for MODE in "${MODES[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    set -- $CFG
    AGG=$1; P=$2; B=$3
    GPU=${GPUS[$(( g % NG ))]}; g=$((g+1))
    (
      for S in $SEEDS; do
        echo "[$MODE $AGG p=$P b=$B signflip] seed=$S -> cuda:$GPU"
        python robust_experiment.py --mode "$MODE" --agg "$AGG" \
          --poison "$P" --beta "$B" --attack signflip \
          --rounds 60 --le 1 --seed "$S" --device "cuda:$GPU"
      done
    ) > logs_unified/robust_${MODE}_${AGG}_p${P}_b${B}.log 2>&1 &
  done
done
wait
echo "=== robustness done -> revision_id/unified/robust_curves/ (3 seeds each) ==="
echo "monitor: tail -f logs_unified/robust_*.log"
