"""Full MIA matrix on the detection architecture trained to a worst-case
memorization state (black-box loss-based MIA, Yeom et al. 2018).

For each (dataset, model, seed):
  member : normal training windows whose input event-id n-gram is rare in the
           actual training set (frequency 1..2) -- the model has seen them.
  nonmem : normal test windows whose input n-gram NEVER occurs in training
           (genuinely novel) -- unseen. Balanced.
The victim has the SAME architecture and training data as the detector, but is
trained to a memorization state (more epochs/rounds). Every metric is derived
from a single confusion matrix; per-example prediction logs are written.
No differential privacy is used; the central-vs-federated gap comes from the
long-tail n-gram memorization being diluted by model averaging.
"""
import os, csv, argparse
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

import standard_detection as sd

DATA = sd.DATA
PRED = f"{DATA}/mia_predlogs"
RESULTS_CSV = f"{DATA}/mia_all.csv"
SEEDS = [0, 1, 2]
DATASETS = ["BGL", "HDFS", "Thunderbird"]
MODELS = ["deeplog", "loganomaly", "neurallog", "onelog"]
# MIA victim uses a fixed-size subset from the detection training distribution,
# trained to a worst-case memorization state (keeps the matrix tractable and
# makes comparisons fair; the variable under test is the training paradigm).
N_FIX = 80000


def prf_from_cm(tn, fp, fn, tp):
    P = tp/(tp+fp) if tp+fp else 0
    R = tp/(tp+fn) if tp+fn else 0
    FPR = fp/(fp+tn) if fp+tn else 0
    F1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return P, R, FPR, F1


def attack(m, Xcol, member_idx, non_idx, ynext, tag):
    lm = sd.per_sample_loss(m, Xcol[member_idx], ynext[member_idx])
    ln = sd.per_sample_loss(m, Xcol[non_idx], ynext[non_idx])
    yt = np.r_[np.ones(len(lm)), np.zeros(len(ln))].astype(int)
    score = -np.r_[lm, ln]
    auc = roc_auc_score(yt, score)
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

    os.makedirs(PRED, exist_ok=True)
    with open(f"{PRED}/{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["RowId", "true", "loss_score", "pred"])
        for i in range(len(yt)):
            w.writerow([i, int(yt[i]), f"{score[i]:.6f}", int(pred[i])])
    return {"auc": auc, "P": P, "R": R, "FPR": FPR, "F1": F1,
            "lm": float(np.median(lm)), "ln": float(np.median(ln))}


def run_cell(ds, model, seed, quick, epochs, rounds):
    sd.CENTRAL_EP = epochs
    sd.ROUNDS = rounds
    D = sd.pickle.load(open(f"{DATA}/{ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, seed, quick, ds)
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
    member_pool = tr_idx[(fv >= 1) & (fv <= 2)]

    normal_test_g = te_groups[:n_norm]
    norm_te_idx = te_idx[np.isin(group[te_idx], normal_test_g)]
    train_keys = set(tcount.keys())
    novel = np.array([tuple(x) not in train_keys for x in X[norm_te_idx]])
    non_pool = norm_te_idx[novel]

    rng = np.random.RandomState(seed)
    m_n = min(len(member_pool), len(non_pool))
    member_idx = member_pool[rng.permutation(len(member_pool))[:m_n]]
    non_idx = non_pool[rng.permutation(len(non_pool))[:m_n]]
    print(f"[{ds} | {model} | s{seed}] member={len(member_idx)} non={len(non_idx)}")

    Xtr, ytr = Xcol[tr_idx], ynext[tr_idx]
    out = {}
    sd.set_seed(seed)
    cm = sd.train_central(model, n, nc, mlen, Xtr, ytr)
    out["central"] = attack(cm, Xcol, member_idx, non_idx, ynext,
                            f"{ds}_{model}_s{seed}_central")
    sd.set_seed(seed + 1000)
    out["fed_iid"] = attack(sd.train_fed(model, n, nc, mlen, Xtr, ytr, True),
                            Xcol, member_idx, non_idx, ynext,
                            f"{ds}_{model}_s{seed}_fed_iid")
    sd.set_seed(seed + 2000)
    out["fed_noniid"] = attack(sd.train_fed(model, n, nc, mlen, Xtr, ytr, False),
                               Xcol, member_idx, non_idx, ynext,
                               f"{ds}_{model}_s{seed}_fed_noniid")
    return out


def load_done():
    done = set()
    if os.path.exists(RESULTS_CSV):
        for r in csv.DictReader(open(RESULTS_CSV)):
            done.add((r["dataset"], r["model"], r["seed"], r["setting"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ds", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=60)
    a = ap.parse_args()
    sd.device = torch.device(a.device)

    ds_list = [a.ds] if a.ds else DATASETS
    model_list = [a.model] if a.model else MODELS
    seed_list = [a.seed] if a.seed is not None else SEEDS

    done = load_done()
    new_file = not os.path.exists(RESULTS_CSV)
    fout = open(RESULTS_CSV, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "model", "seed", "setting",
                    "mia_AUC", "mia_P", "mia_R", "mia_FPR", "mia_F1",
                    "member_loss", "non_loss"])
    for ds in ds_list:
        for model in model_list:
            for seed in seed_list:
                need = [s for s in ["central", "fed_iid", "fed_noniid"]
                        if (ds, model, str(seed), s) not in done]
                if not need:
                    continue
                res = run_cell(ds, model, seed, a.quick, a.epochs, a.rounds)
                for setting in need:
                    r = res[setting]
                    print(f"  {setting:11s} AUC={r['auc']:.4f} P={r['P']:.4f} "
                          f"R={r['R']:.4f} FPR={r['FPR']:.4f} F1={r['F1']:.4f}")
                    w.writerow([ds, model, seed, setting,
                                f"{r['auc']:.4f}", f"{r['P']:.4f}",
                                f"{r['R']:.4f}", f"{r['FPR']:.4f}", f"{r['F1']:.4f}",
                                f"{r['lm']:.4f}", f"{r['ln']:.4f}"])
                fout.flush()
    fout.close()
    print(f"\n[done] -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
