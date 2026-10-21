"""Regenerate Fig. 6 (sign-flip robustness) as the MEAN of 3 seeds with a
+/-1 std shaded band, from the robust CSVs.

Expected csv filenames (per mode iid/noniid, aggregator, p, beta, seed):
  {mode}_{agg}_p{p}_signflip_b{b}_s{seed}.csv
with columns round,P,R,F1.

Usage:
  python plot_fig6_robust.py --csvdir <dir_with_csvs> --out fig6_robust_corrected
"""
import os, argparse, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csvdir", default=".")
ap.add_argument("--out", default="fig6_robust_corrected")
# parse_known_args so it also runs inside Jupyter (which injects -f kernel.json)
a, _ = ap.parse_known_args()

# (legend, aggregator, p, beta, linestyle, color)
SERIES = [
    ("FedAvg, p=0 (clean)", "fedavg", "0.0", "0.1", "-",  "black"),
    ("FedAvg, p=0.1",       "fedavg", "0.1", "0.1", "--", "tab:red"),
    ("FedAvg, p=0.2",       "fedavg", "0.2", "0.1", "--", "tab:orange"),
    ("Trimmed Mean, p=0.1", "trimmed","0.1", "0.1", "-",  "tab:blue"),
    ("Trimmed Mean, p=0.2", "trimmed","0.2", "0.2", "-",  "tab:green"),
]
SEEDS = [0, 1, 2]

def load(mode, agg, p, b):
    frames = []
    for s in SEEDS:
        fp = os.path.join(a.csvdir, f"{mode}_{agg}_p{p}_signflip_b{b}_s{s}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        d = pd.read_csv(fp)
        frames.append(d.sort_values("round")["F1"].values)
    L = min(len(f) for f in frames)
    arr = np.stack([f[:L] for f in frames], axis=0)
    return np.arange(L), arr.mean(0), arr.std(0, ddof=1)

COL2 = 7.0
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True)
for ax, mode, title in [(axes[0], "iid", "(a) IID"), (axes[1], "noniid", "(b) Non-IID")]:
    for label, agg, p, b, ls, col in SERIES:
        r, m, sd = load(mode, agg, p, b)
        ax.plot(r, m, ls=ls, color=col, lw=2.0, label=label)
        ax.fill_between(r, m - sd, m + sd, color=col, alpha=0.12, lw=0)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.65, 0.785)
axes[0].set_ylabel("F1 (balanced test)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, 0.00), fontsize=8, frameon=False,
           columnspacing=0.8, handlelength=1.7, handletextpad=0.3)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(a.out + ".pdf", bbox_inches="tight")
fig.savefig(a.out + ".png", bbox_inches="tight")
print("wrote", a.out + ".pdf", a.out + ".png")
