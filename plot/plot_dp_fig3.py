"""Regenerate the post-training Gaussian utility phase-transition figure
(the paper's Fig. 3) from the CORRECT (RMS-relative) dp_detection.csv.

Noise model: W <- W + sigma * RMS(W) * N. The correct grid is
sigma in 0.00..0.40 step 0.05 and the safe points are
BGL 0.10 / HDFS 0.20 / Thunderbird 0.02.

IMPORTANT: an older experiment used an ABSOLUTE std (W <- W + sigma*N),
whose csv spans sigma 0.000-0.030 and collapses near 0.01. Do NOT use that
file. This script sanity-checks the csv before plotting and aborts if it
looks like the old absolute-std file.

Usage:
  python plot_dp_fig3.py --csv dp_detection.csv
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="dp_detection.csv")
ap.add_argument("--out", default="fig3_dp_phase_corrected")
# parse_known_args so it also runs inside Jupyter (which injects -f kernel.json)
a, _ = ap.parse_known_args()

df = pd.read_csv(a.csv)
g = df.groupby(["dataset", "sigma"])["fused_F1"].mean().reset_index()
piv = g.pivot_table(index="sigma", columns="dataset", values="fused_F1")

# ---- sanity check: must be the RMS-relative grid (0..0.4) ----
sigmax = float(piv.index.max())
assert sigmax >= 0.35, (
    f"[ABORT] {a.csv} max sigma={sigmax} < 0.35 -- this looks like the OLD "
    "absolute-std file (0.000-0.030). Point --csv at the RMS-relative "
    "dp_detection.csv (grid 0.00-0.40) that matches the text/Table VI.")
bgl01 = piv.loc[0.10, "BGL"] if 0.10 in piv.index else np.nan
tb01 = piv.loc[0.10, "Thunderbird"] if 0.10 in piv.index else np.nan
assert bgl01 > 0.97, f"[ABORT] BGL F1(0.10)={bgl01:.3f}, expected ~0.986 (wrong file)"
assert 0.45 < tb01 < 0.62, f"[ABORT] TB F1(0.10)={tb01:.3f}, expected ~0.536 (wrong file)"

print("verified RMS-relative data (sigma up to", sigmax, "):")
print(piv.round(4).to_string())

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(3.6, 2.8))
for ds, mk, col in [("BGL", "o", "tab:blue"),
                    ("HDFS", "s", "tab:green"),
                    ("Thunderbird", "^", "tab:red")]:
    sub = g[g["dataset"] == ds].sort_values("sigma")
    ax.plot(sub["sigma"], sub["fused_F1"], marker=mk, ms=4,
            color=col, lw=1.6, label=ds)

safe = {"BGL": (0.10, 0.986), "HDFS": (0.20, 0.980), "Thunderbird": (0.02, 0.956)}
for ds, (sx, sy) in safe.items():
    ax.scatter([sx], [sy], facecolors="none", edgecolors="black",
               s=70, zorder=5)

ax.set_xlabel(r"Post-training noise multiplier $\sigma$")
ax.set_ylabel("Fused F1")
ax.set_xlim(0.0, 0.40)
ax.set_ylim(0.0, 1.02)
ax.set_xticks(np.arange(0.0, 0.41, 0.05))
ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(a.out + ".pdf", bbox_inches="tight")
fig.savefig(a.out + ".png", bbox_inches="tight")
print("\nwrote", a.out + ".pdf", a.out + ".png")
