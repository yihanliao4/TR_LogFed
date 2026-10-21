#!/usr/bin/env bash
# Minimal end-to-end verification: trains tiny configs and prints one metric
# line per stage. These are NOT paper numbers. Run from the repo root after
# data preparation:
#   bash scripts/smoke_test.sh cuda:0     # (or cpu)
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV=${1:-cuda:0}
echo "=== LogFed smoke test on $DEV ==="

echo "[1/3] central MIA, 2 epochs (BGL / DeepLog) ..."
python matrix_k100_fair.py --mode mia --only-central --central-ep 2 \
    --ds BGL --model deeplog --seed 0 --device "$DEV"

echo "[2/3] sign-flip robustness, 3 rounds (IID, FedAvg, p=0.2) ..."
python robust_experiment.py --mode iid --agg fedavg --poison 0.2 \
    --attack signflip --scale -4.0 --rounds 3 --le 1 --seed 0 --device "$DEV"

echo "[3/3] post-training Gaussian on the detector, 2 rounds (BGL/DeepLog) ..."
python dp_detection.py --ds BGL --model deeplog --rounds 2 --le 1 --seed 0 \
    --device "$DEV"

echo "=== smoke test finished; cleaning smoke-only outputs ==="
rm -f revision_id/unified/main_k100/mia_k100_cep2_centralonly.csv
rm -f robust/iid_fedavg_p0.2_signflip_b0.1_s0.csv
echo "done. (If each stage printed a metric row, the environment is working.)"
