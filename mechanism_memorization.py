"""Long-tail n-gram memorization-dilution mechanism (core novelty, contribution 2).

Direct, quantitative evidence that FedAvg DILUTES memorization of long-tail
n-grams instead of hiding it behind DP noise.

For a victim trained to a worst-case memorization state (NeuralLog, N_FIX=80000,
central 300 epochs vs federated K=100 / R=60 / le=2 / C=0.5 -- identical to
mia_dp / matrix_k100 so the trained models cross-check):

  1. bucket every TRAIN n-gram by its in-training frequency f:
       1, 2, 3, 4-5, 6-10, 11-20, 21-50, >50
  2. for each bucket draw up to N_PER (seen) member windows;
  3. draw a fixed pool of genuinely-novel test windows (training frequency 0);
  4. under each paradigm compute
       member_nll  = mean NLL on the bucket's members,
       non_nll     = mean NLL on the novel pool,
       gap         = non_nll - member_nll   (positive => members are remembered)
       auc         = per-bucket MIA AUC (members vs novel, score = -NLL).

Expected mechanism signature (this is what Fig. "memorization" should show):
  * central: singleton (f=1) member_nll -> 0, gap large, per-bucket AUC high
    (the centralized model memorizes rare n-grams);
  * federated: common-gram member_nll stays low (still learned), but singleton
    member_nll stays HIGH and per-bucket AUC -> 0.5 (the rare-gram update is
    diluted by averaging over clients that never saw it).

Output: revision_id/unified/mechanism_memorization.csv
Usage:
  python mechanism_memorization.py --device cuda:0 --ds BGL --model neurallog
  python mechanism_memorization.py --device cuda:0 --ds all --model neurallog
"""
import os, csv, argparse
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import standard_detection as sd
from fed_core import run_fed

N_FIX = 80000
CENTRAL_EP = 300
K, ROUNDS, LE, C = 100, 60, 2, 0.5
N_PER = 500
N_NOVEL = 500
# (lo, hi_inclusive_or_None, label)
BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"),
           (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"),
           (51, None, ">50")]
OUT_CSV = f"{sd.DATA}/mechanism_memorization.csv"
DATASETS = ["BGL", "HDFS", "Thunderbird"]


def build_pools(D, ds, seed, model):
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, seed, False, ds)
    if len(tr_idx) > N_FIX:
        r0 = np.random.RandomState(seed)
        tr_idx = tr_idx[r0.permutation(len(tr_idx))[:N_FIX]]
    group, glabel = D["group"], D["group_label"]
    X = D["X"]
    Xcol = D["Xchar"] if model == "onelog" else X
    ynext = D["ynext"]

    keys = [tuple(x) for x in X[tr_idx]]
    tcount = Counter(keys)
    fv = np.array([tcount[tuple(x)] for x in X[tr_idx]])

    normal_test_g = te_groups[:n_norm]
    norm_te_idx = te_idx[np.isin(group[te_idx], normal_test_g)]
    train_keys = set(tcount.keys())
    novel = np.array([tuple(x) not in train_keys for x in X[norm_te_idx]])
    non_pool = norm_te_idx[novel]

    rng = np.random.RandomState(seed)
    if len(non_pool) > N_NOVEL:
        non_idx = non_pool[rng.permutation(len(non_pool))[:N_NOVEL]]
    else:
        non_idx = non_pool

    bucket_idx = []
    for lo, hi, lab in BUCKETS:
        if hi is None:
            sel = np.where(fv >= lo)[0]
        else:
            sel = np.where((fv >= lo) & (fv <= hi))[0]
        if len(sel) > N_PER:
            sel = sel[rng.permutation(len(sel))[:N_PER]]
        bucket_idx.append((lo, hi, lab, tr_idx[sel]))
    return Xcol, ynext, tr_idx, non_idx, bucket_idx


def bucket_measures(m, Xcol, ynext, mem_idx, non_idx):
    lm = sd.per_sample_loss(m, Xcol[mem_idx], ynext[mem_idx])
    ln = sd.per_sample_loss(m, Xcol[non_idx], ynext[non_idx])
    yt = np.r_[np.ones(len(lm)), np.zeros(len(ln))].astype(int)
    score = -np.r_[lm, ln]
    auc = roc_auc_score(yt, score)
    return float(np.mean(lm)), float(np.mean(ln)), float(np.mean(ln) - np.mean(lm)), auc


def run_cell(ds, model, seed, device):
    D = sd.pickle.load(open(f"{sd.DATA}/{ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    Xcol, ynext, tr_idx, non_idx, bucket_idx = build_pools(D, ds, seed, model)
    Xtr = Xcol[tr_idx]
    ytr = ynext[tr_idx]

    sd.CENTRAL_EP = CENTRAL_EP
    sd.set_seed(seed)
    mc = sd.train_central(model, n, nc, mlen, Xtr, ytr)
    sd.set_seed(seed + 1000)
    mf = run_fed(model, n, nc, mlen, Xtr, ytr, K,
                 ROUNDS, LE, C, dp_sigma=0.0, seed=seed, device=device)

    rows = []
    for paradigm, m in (("central", mc), ("fed", mf)):
        non_nll = None
        for lo, hi, lab, mem_idx in bucket_idx:
            mn, nn, gap, auc = bucket_measures(m, Xcol, ynext, mem_idx, non_idx)
            non_nll = nn
            rows.append((ds, model, seed, paradigm, lab, lo,
                         hi if hi is not None else "", len(mem_idx),
                         f"{mn:.5f}", f"{nn:.5f}", f"{gap:.5f}", f"{auc:.4f}"))
        print(f"  [{ds} {model} s{seed} {paradigm}] "
              f"f=1 mem_nll={rows[-len(BUCKETS)][8]:>8} non_nll={non_nll:.4f}")
    return rows


def read_existing():
    """Returns dict keyed by (ds,model,seed,paradigm,bucket) -> row tuple."""
    rows = {}
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV)):
            key = (r["dataset"], r["model"], r["seed"], r["paradigm"], r["bucket"])
            rows[key] = r
    return rows


def cell_done(rows, ds, model, seed):
    need = set()
    for paradigm in ("central", "fed"):
        for _, _, lab in BUCKETS:
            need.add((ds, model, str(seed), paradigm, lab))
    return need.issubset(rows.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ds", default="BGL")
    ap.add_argument("--model", default="neurallog")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]
    ds_list = DATASETS if a.ds == "all" else [a.ds]

    rows = read_existing()
    header = ["dataset", "model", "seed", "paradigm", "bucket",
              "flo", "fhi", "n_member", "member_nll", "non_nll", "gap", "auc"]
    for ds in ds_list:
        for seed in seeds:
            if cell_done(rows, ds, a.model, seed):
                print(f"[skip] {ds} {a.model} s{seed} complete")
                continue
            for r in run_cell(ds, a.model, seed, a.device):
                # r is a tuple in the same column order as header
                key = (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]))
                rows[key] = dict(zip(header, [str(x) for x in r]))

    # stable ordering: dataset, model, seed, paradigm (central,fed), bucket order
    border = {lab: i for i, (_, _, lab) in enumerate(BUCKETS)}
    porder = {"central": 0, "fed": 1}
    out = sorted(rows.values(),
                 key=lambda r: (r["dataset"], r["model"], int(r["seed"]),
                                porder[r["paradigm"]], border[r["bucket"]]))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        for r in out:
            wr.writerow(r)
    print(f"\n[done] -> {OUT_CSV} ({len(out)} bucket rows)")


if __name__ == "__main__":
    main()
