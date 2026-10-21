"""R1 / Table VI: disentangle FL and DP, and locate the privacy sweet spot.
For each dataset, three/five victims share the same architecture/data split and
K=100 federation:
  (1) centralized, trained to a worst-case memorization state,
  (2) federated IID (FedAvg), no DP,
  (3) federated IID + a Gaussian mechanism on the final parameters, at several
      sigma levels chosen inside the detection-safe operating region.
For each victim we report the leak-free fused LAD F1 and the loss-based MIA
(AUC/P/R/FPR/F1) on the same long-tail member/non-member sets as mia_matrix.
All metrics come from a single confusion matrix; prediction logs are written.
"""
import os, csv, argparse, copy
from collections import Counter
import numpy as np
import torch

import standard_detection as sd
from mia_matrix import attack
import mia_matrix as mm
from fed_core import run_fed, add_model_noise

DATA = sd.DATA
PRED = f"{DATA}/mia_dp_predlogs"
OUT_CSV = f"{DATA}/mia_dp_table.csv"
SEEDS = [0, 1, 2]
N_FIX = 80000
CENTRAL_EP = 300
FED_ROUNDS, FED_LE, C = 60, 2, 0.5
# Gaussian-mechanism levels inside each dataset's detection-safe region.
DP_SIGMAS = {
    "BGL": [0.05, 0.1, 0.15],
    "HDFS": [0.1, 0.2, 0.4],
    "Thunderbird": [0.02, 0.04, 0.05],
}


def det_fused(m, D, tr_idx, te_idx, te_groups, train_g, n_norm):
    group, glabel = D["group"], D["group_label"]
    nll = sd.group_nll_scores(m, D["X"][te_idx], D["ynext"][te_idx],
                              group, te_idx, te_groups, glabel)
    gcounts = D.get("group_counts")
    cnt = None
    if gcounts is not None:
        Ctr = gcounts[train_g].astype(np.float64)
        cnt = sd.count_z_scores(gcounts, te_groups, Ctr.mean(0), Ctr.std(0))
    return sd.fuse_scores(nll["nll_max"], cnt, n_norm)


def load_done(path):
    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add((r["dataset"], r["seed"], r["setting"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ds", default="BGL")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    seed_list = [a.seed] if a.seed is not None else SEEDS
    ds = a.ds
    out_csv = f"{DATA}/mia_dp_table_{ds}.csv"
    pred_dir = f"{DATA}/mia_dp_predlogs_{ds}"

    done = load_done(out_csv)
    new_file = not os.path.exists(out_csv)
    fout = open(out_csv, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "model", "seed", "setting", "LAD_F1",
                    "MIA_AUC", "MIA_P", "MIA_R", "MIA_FPR", "MIA_F1"])

    D = sd.pickle.load(open(f"{DATA}/{ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    group, glabel, X, ynext = D["group"], D["group_label"], D["X"], D["ynext"]
    for seed in seed_list:
        tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(
            D, seed, False, ds)
        if len(tr_idx) > N_FIX:
            r0 = np.random.RandomState(seed)
            tr_idx = tr_idx[r0.permutation(len(tr_idx))[:N_FIX]]
        Xtr, ytr = X[tr_idx], ynext[tr_idx]

        # long-tail member / novel nonmember pools (same as mia_matrix)
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

        victims = {"central": None, "fed": None}
        sd.set_seed(seed)
        sd.CENTRAL_EP = CENTRAL_EP
        victims["central"] = sd.train_central("neurallog", n, nc, mlen, Xtr, ytr)
        sd.set_seed(seed + 1000)
        m_fed = run_fed("neurallog", n, nc, mlen, Xtr, ytr,
                        100, FED_ROUNDS, FED_LE, C,
                        dp_sigma=0.0, seed=seed, device=sd.device)
        victims["fed"] = m_fed
        # Gaussian mechanism on the same federated model at several safe levels
        for si, dsg in enumerate(DP_SIGMAS[ds]):
            tag = f"fed_dp{int(round(dsg * 100)):02d}"
            mdp = copy.deepcopy(m_fed)
            add_model_noise(mdp, dsg, seed=seed * 1000 + si * 7 + 3)
            victims[tag] = mdp

        for tag, victim in victims.items():
            if (ds, str(seed), tag) in done:
                continue
            dauc, dP, dR, dF1 = det_fused(
                victim, D, tr_idx, te_idx, te_groups, train_g, n_norm)
            mm.PRED = pred_dir
            r = attack(victim, X, member_idx, non_idx, ynext,
                       f"{ds}_neurallog_s{seed}_{tag}")
            print(f"[{ds} s{seed} {tag}] LAD_F1={dF1:.4f} "
                  f"MIA_AUC={r['auc']:.4f} MIA_F1={r['F1']:.4f} FPR={r['FPR']:.4f}")
            w.writerow([ds, "neurallog", seed, tag, f"{dF1:.4f}",
                        f"{r['auc']:.4f}", f"{r['P']:.4f}", f"{r['R']:.4f}",
                        f"{r['FPR']:.4f}", f"{r['F1']:.4f}"])
            fout.flush()
    fout.close()
    print(f"\n[done] -> {out_csv}")


if __name__ == "__main__":
    main()
