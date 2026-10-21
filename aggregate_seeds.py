#!/usr/bin/env python3
# aggregate_seeds.py
# Aggregate multi-seed Soft Prompt ablation results and test whether the
# prompt-learning gain is statistically significant (Reviewer 5).
#
# Reads the NonIID CSVs written by pl_iid_noniid.py for prompt lengths
# {0,5,10} across several seeds, extracts the F1 for one fixed (P, r)
# configuration, reports mean +/- std, and runs paired t-tests against the
# prompt_len=0 baseline.

import os
import argparse
import numpy as np
import pandas as pd


def load_f1(results_dir, pl, K, S, T, seed, P, r):
    path = os.path.join(
        results_dir,
        f"NonIID_PL{pl}_K{K}_S{S}_T{T}_seed{seed}.csv",
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # tolerate float formatting of participant
    row = df[(df["participant"].round(3) == round(P, 3)) & (df["r"] == r)]
    if len(row) == 0:
        return None
    return float(row["f1"].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="revision_results/pl_ablation")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--S", type=int, default=10000)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 42, 123, 7, 2025])
    ap.add_argument("--pls", type=int, nargs="+", default=[0, 5, 10])
    ap.add_argument("--out_csv", default="revision_results/pl_ablation_summary.csv")
    args = ap.parse_args()

    # collect per-seed F1 for each prompt length
    data = {}
    for pl in args.pls:
        vals = []
        for seed in args.seeds:
            f1 = load_f1(args.results_dir, pl, args.K, args.S, args.T, seed, args.P, args.r)
            vals.append(f1)
        data[pl] = vals

    per_seed = pd.DataFrame(data, index=[f"seed{s}" for s in args.seeds])
    per_seed.index.name = "seed"
    print("=== Per-seed Non-IID F1 ===")
    print(per_seed.to_string())
    print()

    # summary
    summary_rows = []
    base = np.array([v for v in data.get(0, []) if v is not None], dtype=float)

    from scipy import stats

    for pl in args.pls:
        arr = np.array([v for v in data[pl] if v is not None], dtype=float)
        row = {
            "prompt_len": pl,
            "n_seeds": len(arr),
            "f1_mean": float(arr.mean()) if len(arr) else np.nan,
            "f1_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "f1_min": float(arr.min()) if len(arr) else np.nan,
            "f1_max": float(arr.max()) if len(arr) else np.nan,
        }
        if pl != 0 and len(base) == len(arr) and len(arr) > 1:
            t_stat, p_val = stats.ttest_rel(arr, base)
            row["paired_t"] = float(t_stat)
            row["paired_p"] = float(p_val)
            row["mean_delta_vs_base"] = float(arr.mean() - base.mean())
            row["significant_0.05"] = bool(p_val < 0.05)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    print("=== Summary (mean +/- std) ===")
    print(summary.to_string(index=False))
    print()

    # plain-language verdict for the response letter
    if "paired_p" in summary.columns:
        for _, r0 in summary[summary["prompt_len"] != 0].iterrows():
            verdict = "SIGNIFICANT" if bool(r0.get("significant_0.05", False)) else "NOT significant"
            print(f"PL-{int(r0['prompt_len'])} vs baseline: "
                  f"delta={r0.get('mean_delta_vs_base', float('nan')):+.6f}, "
                  f"p={r0.get('paired_p', float('nan')):.4f} -> {verdict} at alpha=0.05")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    per_seed.to_csv(args.out_csv.replace(".csv", "_perseed.csv"))
    print(f"\n[Saved] {args.out_csv}")


if __name__ == "__main__":
    main()
