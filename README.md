# LogFed: Privacy-Preserving Federated Log Anomaly Detection

Official code for the paper **"Enhancing Reliability of Log Anomaly Detection via Privacy-Preserving Federated Learning"**. LogFed is
a unified next-event-prediction framework for log anomaly detection (LAD)
under federated learning (FL), with four backbones (DeepLog, LogAnomaly,
NeuralLog, OneLog), a black-box membership-inference-attack (MIA) privacy
study, empirical and formal differential privacy (DP), Byzantine-robust
aggregation, and an efficiency/communication profile.

This repository reproduces every experiment, table, and figure in the paper.
It is CPU-compatible for small smoke tests but the full matrix is intended for
multi-GPU machines.

> **Run everything from the repository root.** The Python modules use flat
> imports (`from models_next import build_model`), so they are laid out at the
> root rather than in a package.

---

## 1. Repository structure

```
logfed/
├── README.md  requirements.txt  LICENSE  .gitignore
│
├── # ---- core framework (flat modules at root) ----
├── standard_detection.py     # data split, fused sequence/count detector, central training
├── models_next.py            # all four backbones (LSTM / attention / Transformer / char-CNN)
├── models_pl.py              # soft-prompt variants (ablation)
├── fed_core.py               # FedAvg, client selection, Trimmed Mean, Gaussian/DP-SGD noise, run_fed
│
├── # ---- experiment entry points ----
├── matrix_k100.py            # MAIN grid: detection & MIA (K=100, IID / cross non-IID)
├── matrix_k100_fair.py       # matched-budget MIA (central 75/150/300, full-detector federated)
├── mia_matrix.py  mia_on_detection.py  lira.py  mechanism_memorization.py
├── dp_detection.py  dp_sgd.py  mia_dp.py
├── robust_experiment.py  efficiency_profile.py
├── pl_ablation.py  pl_ttest.py  aggregate_seeds.py  make_figures.py
│
├── # ---- data preparation (raw -> revision_id/unified/*.pkl) ----
├── build_all_data.py  prepare_data.py
│
├── scripts/
│   ├── smoke_test.sh          # quick verification (CPU/1 GPU)
│   ├── run_mia_fast.sh        # matched-budget MIA matrix, dynamic multi-GPU queue
│   ├── run_robust_seeds.sh    # sign-flip robustness over 3 seeds, multi-GPU
│   ├── run_finding5.sh        # efficiency / end-to-end timing
│   └── finding5_e2e.py        # CPU post-processing for the efficiency table
|
└── results/revision/          # small result CSVs shipped for figure/table reproduction
```

---

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

Tested with Python 3.10/3.11 and PyTorch >= 2.1 on CUDA.

---

## 3. Data preparation

Experiments read three preprocessed pickles from
`revision_id/unified/{BGL,HDFS,Thunderbird}.pkl` (the path is set by `DATA` in
`standard_detection.py`). The raw BGL / HDFS / Thunderbird logs are public
benchmarks and are **not** bundled here; download them from their official
sources and place them under `datasets/` as expected by `build_all_data.py`,
then run:

```bash
python build_all_data.py
```

The pickles hold learned event-embedding sequences (and character sequences
for OneLog), next-event labels, block/window grouping, and the HDFS
event-count profile. Raw logs and `*.pkl` are excluded by `.gitignore`.

---

## 4. Quick verification (smoke test)

After data preparation, a few-minute reduced configuration verifies the
environment end-to-end (these numbers are **not** paper results):

```bash
bash scripts/smoke_test.sh cuda:0      # use 'cpu' if no GPU
```

It runs a 2-epoch central MIA, a 3-round sign-flip run, and a 2-round Gaussian
sweep, printing one metric line per stage, then removes its temporary outputs.

---

## 5. Reproduce the paper

All training scripts are **resumable**: completed
`(dataset, model, seed, setting)` cells are skipped on restart, so any command
can be interrupted and re-launched safely.

### 5.1 Detection (Tables III–IV)

```bash
python matrix_k100.py --mode detection --rounds 50 --iid-le 3 --cross-le 3
python pl_ablation.py && python pl_ttest.py
```

### 5.2 MIA privacy study (Table V) and the matched-budget comparison

```bash
# dynamic 4-GPU queue: L0 deployed federated + central 300 bound,
# L1 central matched 75 (IID), L2/L3 pooled matched 150 / converged 300
bash scripts/run_mia_fast.sh 0 1 2 3
# add SKIP_L3=1 to omit the expensive pooled-300 upper bound
```

### 5.3 Differential privacy (Fig. 3, Table VI)

```bash
python dp_detection.py --rounds 50 --le 3
python dp_sgd.py
python mia_dp.py
```

The Gaussian mechanism is RMS-relative, `W <- W + sigma RMS(W) N`, with
`sigma in {0.0,...,0.4}`.

### 5.4 Sign-flip robustness (Fig. 6)

```bash
bash scripts/run_robust_seeds.sh 0 1 2 3
```

The attack is a Byzantine **update sign-flip with amplification**
`w_tilde = w_t + gamma (w^k - w_t)`, `gamma = -4` (no labels are changed);
malicious fraction `p in {0,0.1,0.2}`, Trimmed Mean uses `beta = p`.

### 5.5 Efficiency (Tables VII–VIII)

```bash
bash scripts/run_finding5.sh
```

### 5.6 Aggregation and figures

```bash
python aggregate_seeds.py
python make_figures.py
```

---

## 6. Outputs

- Training/evaluation metrics: `revision_id/unified/main_k100/*.csv` and
  `robust/*.csv`.
- Figures: `figs/` (or the `--out` path of the plot scripts).
- Each result row is a `(dataset, model, seed, setting)` cell, so downstream
  aggregation is a group-by; the shipped `results/revision/*.csv` let you
  regenerate the figures/tables without re-running training.

---

## 7. Notes

- Main results use seeds 0,1,2; the soft-prompt ablation uses five seeds with
  paired t-tests.
- The balanced-test F1 in the robustness experiment (about 0.73) is
  intentionally not comparable to the fused imbalanced F1 of the detection
  tables; only the relative FedAvg vs. Trimmed Mean behavior is meaningful.

