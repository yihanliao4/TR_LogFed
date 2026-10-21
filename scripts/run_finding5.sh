#!/usr/bin/env bash
# ============================================================
# Finding 5 / Table VII / Table VIII re-run (average timing).
#
# Step 1: efficiency_profile.py (profiling is light, so it runs all four
#         backbones SERIALLY on ONE GPU; this avoids concurrent jobs
#         overwriting the same eff_timing.csv / eff_params.csv).
#         It already averages wall-clock over timed rounds (1 warm-up;
#         IID 20 rounds, cross 3 rounds) with CUDA events and records
#         params / FP32 size / analytic RTT at 1/5/20 Mbps.
# Step 2: finding5_e2e.py (CPU-only) reads the two CSVs and writes the
#         average full end-to-end run time for R in {30,50,100}:
#         avg_e2e = R * (avg fed_cross_round compute + analytic RTT).
#
# Usage:
#   bash run_finding5.sh [GPU_ID]        # default cuda:0
#   bash run_finding5.sh 3               # pin to a spare card
# Output: revision_id/unified/efficiency/
#           eff_timing.csv, eff_params.csv, finding5_e2e.csv
# ============================================================
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

GPU=${1:-0}
mkdir -p logs_unified
EFF_DIR="revision_id/unified/efficiency"

echo "=== Finding 5 average profiling on cuda:$GPU ==="
python efficiency_profile.py --device "cuda:$GPU" 2>&1 | tee logs_unified/finding5.log
python finding5_e2e.py | tee logs_unified/finding5_e2e.log
echo "=== Finding 5 done -> $EFF_DIR/{eff_timing,eff_params,finding5_e2e}.csv ==="
