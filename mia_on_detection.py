"""Quick MIA diagnostic on the SAME model used for detection (standard training
recipe in standard_detection.py), not an artificially over-fit model.

Black-box loss-based MIA (Yeom et al. 2018): query the victim for the
per-example NLL of the true next event.
  member : normal training windows whose input n-gram is rare in the actual
           training set (frequency 1..2) -- seen by the model.
  nonmem : normal test windows whose input n-gram NEVER occurs in the training
           set (genuinely novel) -- unseen.
Members and nonmembers are balanced. All metrics derive from one confusion
matrix. Long-tail membership is defined on event-id n-grams (D['X']).
"""
import argparse
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

import standard_detection as sd


def prf_from_cm(tn, fp, fn, tp):
    P = tp/(tp+fp) if tp+fp else 0
    R = tp/(tp+fn) if tp+fn else 0
    FPR = fp/(fp+tn) if fp+tn else 0
    F1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return P, R, FPR, F1


def attack(m, Xcol, member_idx, non_idx, ynext):
    lm = sd.per_sample_loss(m, Xcol[member_idx], ynext[member_idx])
    ln = sd.per_sample_loss(m, Xcol[non_idx], ynext[non_idx])
    # member expected to have LOWER loss; attack score = -loss
    yt = np.r_[np.ones(len(lm)), np.zeros(len(ln))].astype(int)
    score = -np.r_[lm, ln]
    auc = roc_auc_score(yt, score)
    # Youden threshold -> single confusion matrix
    best_j, bt = -1.0, 0.0
    for t in np.quantile(score, np.linspace(0, 1, 400)):
        pred = (score > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        j = tp/(tp+fn) - fp/(fp+tn)
        if j > best_j:
            best_j, bt = j, t
    pred = (score > bt).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    P, R, FPR, F1 = prf_from_cm(tn, fp, fn, tp)
    print(f"  MIA AUC={auc:.4f} P={P:.4f} R={R:.4f} FPR={FPR:.4f} F1={F1:.4f} "
          f"| member_loss med={np.median(lm):.4f} non_loss med={np.median(ln):.4f}")
    return auc, P, R, FPR, F1


def run(ds, model, seed, quick, epochs, rounds):
    # train the SAME architecture/data as the detector, but to a worst-case
    # memorization state (more epochs); train_central/train_fed read these globals
    sd.CENTRAL_EP = epochs
    sd.ROUNDS = rounds
    D = sd.pickle.load(open(f"{sd.DATA}/{ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, seed, quick, ds)
    group, glabel = D["group"], D["group_label"]
    X = D["X"]
    Xcol = D["Xchar"] if model == "onelog" else X
    ynext = D["ynext"]

    # event-id n-gram counts over the ACTUAL training windows
    keys = [tuple(x) for x in X[tr_idx]]
    tcount = Counter(keys)
    fv = np.array([tcount[tuple(x)] for x in X[tr_idx]])
    member_pool = tr_idx[(fv >= 1) & (fv <= 2)]

    # normal test windows (first n_norm groups are held-out normal)
    normal_test_g = te_groups[:n_norm]
    te_is_norm = np.isin(group[te_idx], normal_test_g)
    norm_te_idx = te_idx[te_is_norm]
    train_keys = set(tcount.keys())
    novel = np.array([tuple(x) not in train_keys for x in X[norm_te_idx]])
    non_pool = norm_te_idx[novel]

    rng = np.random.RandomState(seed)
    m_n = min(len(member_pool), len(non_pool))
    member_idx = member_pool[rng.permutation(len(member_pool))[:m_n]]
    non_idx = non_pool[rng.permutation(len(non_pool))[:m_n]]
    print(f"[{ds} | {model} | s{seed}] member={len(member_idx)} non={len(non_idx)} "
          f"(pool member={len(member_pool)} non={len(non_pool)})")

    Xtr, ytr = Xcol[tr_idx], ynext[tr_idx]
    sd.set_seed(seed)
    print(" central:")
    attack(sd.train_central(model, n, nc, mlen, Xtr, ytr), Xcol, member_idx, non_idx, ynext)
    sd.set_seed(seed + 1000)
    print(" fed_iid:")
    attack(sd.train_fed(model, n, nc, mlen, Xtr, ytr, True), Xcol, member_idx, non_idx, ynext)
    sd.set_seed(seed + 2000)
    print(" fed_noniid:")
    attack(sd.train_fed(model, n, nc, mlen, Xtr, ytr, False), Xcol, member_idx, non_idx, ynext)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ds", default="BGL")
    ap.add_argument("--model", default="deeplog")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=150, help="central epochs (worst-case)")
    ap.add_argument("--rounds", type=int, default=60, help="federated rounds (worst-case)")
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    run(a.ds, a.model, a.seed, a.quick, a.epochs, a.rounds)


if __name__ == "__main__":
    main()
