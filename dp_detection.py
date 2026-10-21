"""R1/DP experiment: detection vs privacy noise on the next-event models.
Per-dataset IID federation, K=100. A clean federated model is trained once per
(dataset, seed), then a Gaussian mechanism is applied to its final parameters
at increasing sigma (W <- W + sigma*RMS(W)*N), so every noise level shares the
exact same trained model and the only variable is the injected noise. We report
two leak-free results: the sequence-only detector (nll_max) and the fused
detector that also uses the HDFS event-count profile. Thresholds are calibrated
on held-out normal groups only.
"""
import os, csv, argparse, copy
import numpy as np
import torch

import standard_detection as sd
from fed_core import run_fed, add_model_noise

DATA = sd.DATA
OUT_CSV = f"{DATA}/dp_detection.csv"
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
N_REP = 5     # independent Gaussian noise realizations per sigma
SEEDS = [0, 1, 2]
DATASETS = ["BGL", "HDFS", "Thunderbird"]
TRAIN_CAP = 80000


def eval_methods(m, D, te_idx, te_groups, train_g, n_norm):
    group, glabel = D["group"], D["group_label"]
    nll = sd.group_nll_scores(m, D["X"][te_idx], D["ynext"][te_idx],
                              group, te_idx, te_groups, glabel)
    seq = nll["nll_max"]
    s_res = sd.fuse_scores(seq, None, n_norm)          # sequence-only
    cnt = None
    gcounts = D.get("group_counts")
    if gcounts is not None:
        Ctr = gcounts[train_g].astype(np.float64)
        cnt = sd.count_z_scores(gcounts, te_groups, Ctr.mean(0), Ctr.std(0))
    f_res = sd.fuse_scores(seq, cnt, n_norm)           # fused with count
    return s_res, f_res


def load_done():
    done = set()
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV)):
            done.add((r["dataset"], r["model"], r["seed"], r["sigma"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ds", default=None)
    ap.add_argument("--model", default="neurallog")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--le", type=int, default=1)
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    device = torch.device(a.device)
    sd.device = device

    ds_list = [a.ds] if a.ds else DATASETS
    seed_list = [a.seed] if a.seed is not None else SEEDS
    done = load_done()
    new_file = not os.path.exists(OUT_CSV)
    fout = open(OUT_CSV, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "model", "seed", "sigma", "K", "rounds",
                    "seq_AUC", "seq_P", "seq_R", "seq_F1",
                    "fused_AUC", "fused_P", "fused_R", "fused_F1",
                    "seq_F1_std", "fused_F1_std"])

    for ds in ds_list:
        D = sd.pickle.load(open(f"{DATA}/{ds}.pkl", "rb"))
        n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
        for seed in seed_list:
            tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(
                D, seed, False, ds)
            Xtr, ytr = D["X"][tr_idx], D["ynext"][tr_idx]
            if len(Xtr) > TRAIN_CAP:
                c0 = np.random.RandomState(seed)
                csel = c0.permutation(len(Xtr))[:TRAIN_CAP]
                Xtr, ytr = Xtr[csel], ytr[csel]
            sd.set_seed(seed)
            m_clean = run_fed(a.model, n, nc, mlen, Xtr, ytr,
                              a.K, a.rounds, a.le, a.C, dp_sigma=0.0,
                              seed=seed, device=device)
            for si, sigma in enumerate(SIGMAS):
                if (ds, a.model, str(seed), f"{sigma}") in done:
                    continue
                s_acc, f_acc = [], []
                for rep in range(N_REP):
                    m = copy.deepcopy(m_clean)
                    add_model_noise(m, sigma, seed=seed * 10000 + si * 100 + rep)
                    s_res, f_res = eval_methods(
                        m, D, te_idx, te_groups, train_g, n_norm)
                    s_acc.append(s_res); f_acc.append(f_res)
                s_acc = np.array(s_acc); f_acc = np.array(f_acc)
                sm, fm = s_acc.mean(0), f_acc.mean(0)
                print(f"[{ds} s{seed} sigma={sigma}] seqF1={sm[3]:.4f}"
                      f"+-{s_acc[:,3].std():.4f}  fusedF1={fm[3]:.4f}"
                      f"+-{f_acc[:,3].std():.4f}")
                w.writerow([ds, a.model, seed, sigma, a.K, a.rounds]
                           + [f"{v:.4f}" for v in sm]
                           + [f"{v:.4f}" for v in fm]
                           + [f"{s_acc[:,3].std():.4f}",
                              f"{f_acc[:,3].std():.4f}"])
                fout.flush()
    fout.close()
    print(f"\n[done] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
