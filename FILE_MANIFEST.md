# LogFed release file manifest

## A. INCLUDE (curated core — what the paper claims depend on)

**Framework / detection**
- `standard_detection.py` — split, fused sequence/count detector, central training/eval
- `models_next.py` — all 4 backbones (DeepLog, LogAnomaly, NeuralLog, OneLog)
- `models_pl.py` — soft-prompt variants (only used by the prompt ablation)
- `fed_core.py` — partition, FedAvg, client selection, Trimmed Mean, Gaussian/DP-SGD noise, `run_fed`

**Main experiment grids**
- `matrix_k100.py` — canonical detection + MIA grid (K=100)
- `matrix_k100_fair.py` — matched-budget MIA (central 75/150/300, full-detector federated)

**Attacks / privacy**
- `mia_matrix.py`, `mia_on_detection.py`, `lira.py`, `mechanism_memorization.py`

**Differential privacy**
- `dp_detection.py` (Gaussian, Fig. 3), `dp_sgd.py` (formal), `mia_dp.py`

**Robustness / efficiency / ablation**
- `robust_experiment.py` (sign-flip, Fig. 6), `efficiency_profile.py`
- `pl_ablation.py`, `pl_ttest.py`

**Aggregation / figures**
- `aggregate_seeds.py`, `make_figures.py`
- `plot_dp_fig3.py`, `plot_fig6_robust.py`

**Data prep**
- `build_all_data.py`, `prepare_data.py` (add any builder these import)

**Launchers / verification**
- `run_mia_fast.sh`, `run_robust_seeds.sh`, `run_finding5.sh`, `finding5_e2e.py`, `smoke_test.sh`

**Results (tiny CSVs, ship for figure/table reproduction)**
- `results/revision/*.csv` (detection_k100_final, dp_detection, mia_k100_final,
  eff_timing, eff_params, mechanism_memorization, mia_dp_table_*)

## B. EXCLUDE (diagnostics / exploratory / superseded pipelines)

- Diagnostics: `*_diagnose.py`, `*_diag.py`, `hdfs_*diag*`, `inspect_clients.py`,
  `check_data.py`, `probe_bgl_signal.py`, `next_event_gap.py`, `longtail_gap.py`,
  `diagnose_bgl_gap.py`, `bgl_diagnose.py`, `hdfs_count_diag.py`, `hdfs_model_diag.py`,
  `verify_test.py`
- Exploration / superseded: `scan_fed.py`, `search_fed_advantage.py`,
  `small_data_fed.py`, `extreme_overfit.py`, `fingerprint_test.py`,
  `fed_longtail.py`, `fed_mia.py`, `mia_lad.py`, `iid_noniid.py`,
  `pl_iid_noniid.py`, `final_pipeline.py`, `onelog.py` (OneLog now lives in
  `models_next.py`; keep only if an old script still imports it),
  `attack_mlp.py`, `attack_transfer.py` (keep only if `lira.py`/`matrix*` import them)
- Ad-hoc run scripts: `run_all.py`, `run_bgl_*.py`, `run_hdfs_deeplog.py`,
  `run_experiments.sh`, `run_iid_r3.sh`, `run_noniid_r3.sh`, `run_noniid_parallel.sh`,
  `run_k100.sh`, `rerun_onelog_r30c05.sh`, `resplit_bgl.py`, `build_bgl.py`
- Table builders (`build_table3.py`, `build_big_table.py`, `build_ablation.py`) — optional;
  keep only if you want reviewers to regenerate the formatted tables directly.

## C. NEVER commit (large / generated / sensitive)
- Raw BGL/HDFS/Thunderbird logs and `revision_id/**/*.pkl` (rebuild via `build_all_data.py`)
- `*.pt`, `*.ckpt`, `logs*/`, `__pycache__/`, `robust/` (regenerated), `*.out`
- Any absolute path / private directory references.

## D. Before pushing — sanity checklist
1. `grep -RInE "/Users/|/home/|C:\\\\" src/` → no machine-specific paths (paths are read from `DATA`).
2. Fresh clone → `pip install -r requirements.txt` → `bash scripts/smoke_test.sh cuda:0`.
3. Confirm the README command for one full table/figure reproduces from a shipped CSV.
4. Add the final paper BibTeX to README §8 and set the LICENSE year.
