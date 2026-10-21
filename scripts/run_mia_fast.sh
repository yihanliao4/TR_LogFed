#!/usr/bin/env bash
# ============================================================
# Fast fair-MIA launcher: a flat, shuffled work QUEUE fed to all
# GPUs dynamically (no GPU sits idle). Resumable (matrix_k100_fair
# skips rows already present in the target CSVs).
#
# Uses CUDA_VISIBLE_DEVICES, so every worker passes --device cuda:0
# (the dispatcher maps it to the physical GPU).
#
#   bash run_mia_fast.sh 0 1 2 3
#   SKIP_L3=1 bash run_mia_fast.sh 0 1 2 3      # skip pooled 300 (bound)
#   DRY=1 bash run_mia_fast.sh 0 1 2 3         # list jobs, don't run
# ============================================================
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
GPUS=("$@"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0 1 2 3)
NG=${#GPUS[@]}
SEEDS=${SEEDS:-"0 1 2"}
DS_LIST=(BGL HDFS Thunderbird)
MODELS=(deeplog loganomaly neurallog onelog)
PY="python matrix_k100_fair.py"
FED="--mode mia --rounds 50 --iid-le 3 --cross-le 3 --full-detector"
mkdir -p logs_fast
echo "=== fast fair MIA r=3 R=50 | $NG GPUs (${GPUS[*]}) SKIP_L3=${SKIP_L3:-0} ==="

# ---- build shuffled job lists (one job per line) ----
build_lists(){
  : > /tmp/mia_a.$$
  for ds in "${DS_LIST[@]}"; do
    for m in "${MODELS[@]}"; do
      for s in $SEEDS; do
        # L0: per-dataset central300 + fed_iid(full detector)
        echo "A0 $ds $m $s" >> /tmp/mia_a.$$
        # L1: per-dataset central-matched 75
        echo "A1 $ds $m $s" >> /tmp/mia_a.$$
      done
    done
  done
  : > /tmp/mia_b.$$
  for m in "${MODELS[@]}"; do
    for s in $SEEDS; do
      echo "B0 $m $s" >> /tmp/mia_b.$$     # cross fed_noniid
      echo "B2 $m $s" >> /tmp/mia_b.$$     # pooled central 150
      [ -z "$SKIP_L3" ] && echo "B3 $m $s" >> /tmp/mia_b.$$  # pooled 300
    done
  done
  shuf /tmp/mia_a.$$ > /tmp/mia_a.$$.s; mv /tmp/mia_a.$$.s /tmp/mia_a.$$
  shuf /tmp/mia_b.$$ > /tmp/mia_b.$$.s; mv /tmp/mia_b.$$.s /tmp/mia_b.$$
}

run_one(){
  # $1 physical gpu, rest = job fields
  local gpu=$1; local tag=$2
  shift 2
  local kind=$1
  local log cmd
  if [ "$kind" = "A0" ]; then
    local ds=$2 m=$3 s=$4
    log="logs_fast/A0_${ds}_${m}_s${s}.log"
    cmd="$PY $FED --ds $ds --model $m --seed $s --device cuda:0"
  elif [ "$kind" = "A1" ]; then
    local ds=$2 m=$3 s=$4
    log="logs_fast/A1_${ds}_${m}_s${s}.log"
    cmd="$PY --mode mia --only-central --central-ep 75 --ds $ds --model $m --seed $s --device cuda:0"
  elif [ "$kind" = "B0" ]; then
    local m=$2 s=$3
    log="logs_fast/B0_cross_${m}_s${s}.log"
    cmd="$PY $FED --cross --model $m --seed $s --device cuda:0"
  elif [ "$kind" = "B2" ]; then
    local m=$2 s=$3
    log="logs_fast/B2_pool150_${m}_s${s}.log"
    cmd="$PY --mode mia --only-central --cross --central-ep 150 --model $m --seed $s --device cuda:0"
  elif [ "$kind" = "B3" ]; then
    local m=$2 s=$3
    log="logs_fast/B3_pool300_${m}_s${s}.log"
    cmd="$PY --mode mia --only-central --cross --central-ep 300 --model $m --seed $s --device cuda:0"
  fi
  echo "[$tag gpu$gpu] $cmd"
  [ -n "$DRY" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$log" 2>&1
}

# worker pool: GPU-ID tokens in a FIFO; a job grabs a free GPU, runs,
# then returns the token so the next job takes the genuinely freed GPU.
run_pool(){
  local list=$1
  local fifo=/tmp/mia_gpu.$$
  mkfifo "$fifo"; exec 9<>"$fifo"
  for g in "${GPUS[@]}"; do echo "$g" >&9; done
  while IFS=' ' read -r -a J; do
    local gpu; read -r gpu <&9
    (
      run_one "$gpu" "${J[0]}" "${J[@]}"
      echo "$gpu" >&9
    ) &
  done < "$list"
  wait
  exec 9>&-; rm -f "$fifo"
}

build_lists
echo "Phase A jobs: $(wc -l < /tmp/mia_a.$$)  | Phase B jobs: $(wc -l < /tmp/mia_b.$$)"
if [ -n "$DRY" ]; then
  echo "--- DRY: Phase A (shuffled) ---"; cat /tmp/mia_a.$$
  echo "--- DRY: Phase B (shuffled) ---"; cat /tmp/mia_b.$$
  rm -f /tmp/mia_a.$$ /tmp/mia_b.$$
  exit 0
fi
run_pool /tmp/mia_a.$$
echo "--- Phase A done ---"
run_pool /tmp/mia_b.$$
echo "--- Phase B done ---"
rm -f /tmp/mia_a.$$ /tmp/mia_b.$$
echo "=== all done. CSVs in revision_id/unified/main_k100/ ==="
