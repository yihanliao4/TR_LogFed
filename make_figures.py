"""
LogFed paper figures (self-contained, column-name robust).
Reads:
  dp_detection.csv                 -> Fig. 4 DP utility phase transition
  mechanism_memorization.csv      -> Fig. 5 long-tail mechanism (2 subplots)
  robust/*.csv (10 files)         -> Fig. 6 sign-flip robustness (2 panels)
Writes:
  figs/fig4_dp_phase.pdf
  figs/fig5_mechanism.pdf
  figs/fig6_robust.pdf
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths (edit if your files live elsewhere) ----
DP_DET   = "dp_detection.csv"
MECH     = "mechanism_memorization.csv"
ROB      = "robust"
OUTDIR   = "figs"
os.makedirs(OUTDIR, exist_ok=True)

# IEEE single-column ~3.4in; two-panel plots use ~7.0in
COL1, COL2 = 3.4, 7.4
# Times New Roman style: serif text and STIX math for a Times-like look.
plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8, "figure.dpi": 150, "savefig.dpi": 300})


def pick(df, *cands):
    """Return the first present column among cands (case/underscore-insensitive)."""
    for c in cands:
        if c in df.columns:
            return c
    norm = {c.lower().replace("_", ""): c for c in df.columns}
    for c in cands:
        k = c.lower().replace("_", "")
        if k in norm:
            return norm[k]
    raise KeyError(f"none of {cands}; cols={list(df.columns)}")


# =================================================================
# Fig. 4 — DP utility phase transition: fused F1 vs sigma
# =================================================================
df = pd.read_csv(DP_DET)
DS  = pick(df, "dataset", "ds")
SIG = pick(df, "sigma", "noise_sigma", "dp_sigma")
F1  = pick(df, "fused_F1", "F1", "f1")
g = df.groupby([DS, SIG])[F1].mean().reset_index()

fig, ax = plt.subplots(figsize=(COL1, 2.6))
for ds, mk, col in [("BGL", "o", "tab:blue"),
                    ("HDFS", "s", "tab:green"),
                    ("Thunderbird", "^", "tab:red")]:
    sub = g[g[DS] == ds].sort_values(SIG)
    ax.plot(sub[SIG], sub[F1], marker=mk, ms=4, color=col, lw=1.6, label=ds)
# mark safe operating points
for ds, sx, sy in [("BGL", 0.10, 0.986), ("HDFS", 0.20, 0.980),
                   ("Thunderbird", 0.02, 0.956)]:
    ax.annotate("", xy=(sx, sy), xytext=(sx, sy - 0.10),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))
ax.set_xlabel(r"Post-training noise multiplier $\sigma$")
ax.set_ylabel("Fused F1")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig4_dp_phase.pdf", bbox_inches="tight")
plt.close(fig)


# =================================================================
# Fig. 5 — Long-tail mechanism: (a) member NLL, (b) per-bucket AUC
# =================================================================
d = pd.read_csv(MECH)
PAR = pick(d, "paradigm", "setting")
BUC = pick(d, "bucket")
NLL = pick(d, "member_nll", "memberNLL", "mem_nll")
AUC = pick(d, "auc", "AUC")
BUCKET_ORDER = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", ">50"]
d[BUC] = pd.Categorical(d[BUC], categories=BUCKET_ORDER, ordered=True)
x = list(range(len(BUCKET_ORDER)))

agg = d.groupby([PAR, BUC], observed=True).agg(
    nll=(NLL, "mean"), auc=(AUC, "mean")).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.7))
for par, ls, col in [("central", "-", "tab:red"), ("fed", "--", "tab:blue")]:
    sub = agg[agg[PAR] == par].sort_values(BUC)
    axes[0].plot(x, sub["nll"], ls=ls, color=col, lw=1.8,
                 label="Centralized" if par == "central" else "Federated")
    axes[1].plot(x, sub["auc"], ls=ls, color=col, lw=1.8,
                 label="Centralized" if par == "central" else "Federated")
axes[0].set_ylabel("Member NLL")
axes[1].set_ylabel("Membership AUC")
axes[1].axhline(0.5, color="gray", lw=0.8, ls=":")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Training-frequency bucket $f$")
    ax.grid(alpha=0.3)
axes[1].set_ylim(0.45, 1.02)
axes[0].legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig5_mechanism.pdf", bbox_inches="tight")
plt.close(fig)


# =================================================================
# Fig. 6 — Sign-flip robustness: F1 vs round, IID / Non-IID
# =================================================================
# (panel, legend, aggregator, malicious fraction p, attack, b, linestyle, color)
SERIES = [
    ("Clean",              "fedavg", "0.0", "none",     "0.1", "-",  "black"),
    ("FedAvg, p=0.1",      "fedavg", "0.1", "signflip", "0.1", "--", "tab:red"),
    ("FedAvg, p=0.2",      "fedavg", "0.2", "signflip", "0.1", "--", "tab:orange"),
    ("Trimmed Mean, p=0.1", "trimmed", "0.1", "signflip", "0.1", "-", "tab:blue"),
    ("Trimmed Mean, p=0.2", "trimmed", "0.2", "signflip", "0.2", "-", "tab:green"),
]


def rfile(setting, agg, p, attack, b):
    return os.path.join(ROB, f"{setting}_{agg}_p{p}_{attack}_b{b}_s0.csv")


fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True)
for ax, setting, title in [(axes[0], "iid", "IID"),
                           (axes[1], "noniid", "Non-IID")]:
    for label, agg, p, attack, b, ls, col in SERIES:
        fp = rfile(setting, agg, p, attack, b)
        if not os.path.exists(fp):
            print("MISSING:", fp)
            continue
        d = pd.read_csv(fp)
        RND = pick(d, "round", "rnd")
        FF  = pick(d, "F1", "f1")
        ax.plot(d[RND], d[FF], ls=ls, color=col, lw=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.65, 0.785)
axes[0].set_ylabel("F1 (balanced test)")

# Compact single-row, frameless legend (5 entries) to minimize vertical space.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, 0.00),
           fontsize=8, frameon=False,
           columnspacing=0.8,
           handlelength=1.7,
           handletextpad=0.3)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f"{OUTDIR}/fig6_robust.pdf", bbox_inches="tight")
plt.close(fig)

print("Done. Figures written to", OUTDIR)
