"""R5: Soft Prompt (prefix tuning) ablation on the NeuralLog Transformer,
which is the next-event counterpart of the original LogFed Transformer.

Same framework, only one variable -- prompt_len in {0, 5, 10}. Detection uses
the identical leak-free fused detector as standard_detection. We run 5 seeds
for a paired t-test (prompt>0 vs prompt=0 on the same data split/seed).
All datasets use a fixed 80k training cap so the only varying factor is the
prompt; the fused detector for HDFS is count-dominated (negative control).
"""
import os, csv, argparse
import numpy as np
import torch

import standard_detection as sd
from models_pl import build_model_pl

DATA = sd.DATA
RESULTS_CSV = f"{DATA}/pl_ablation.csv"
SEEDS = [2026, 42, 123, 7, 2025]
PROMPTS = [0, 5, 10]
SETTINGS = ["central", "fed_iid", "fed_noniid"]
DATASETS = ["BGL", "HDFS", "Thunderbird"]
CAP = 80000


def eval_fused(m, D, tr_idx, te_idx, te_groups, train_g, n_norm):
    group, glabel = D["group"], D["group_label"]
    Xcol = D["X"]  # NeuralLog uses event ids
    ynext_te = D["ynext"][te_idx]
    ytrue_g = glabel[te_groups]
    nll = sd.group_nll_scores(m, Xcol[te_idx], ynext_te, group, te_idx,
                              te_groups, glabel)
    cnt = None
    gcounts = D.get("group_counts")
    if gcounts is not None:
        Ctr = gcounts[train_g].astype(np.float64)
        cnt = sd.count_z_scores(gcounts, te_groups, Ctr.mean(0), Ctr.std(0))
    return sd.fuse_scores(nll["nll_max"], cnt, n_norm)


def load_done():
    done = set()
    if os.path.exists(RESULTS_CSV):
        for r in csv.DictReader(open(RESULTS_CSV)):
            done.add((r["dataset"], r["seed"], r["setting"], r["prompt_len"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ds", default=None)
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    # fixed, equal training scale across datasets for a fair prompt comparison
    sd.TRAIN_CAP = CAP
    sd.TRAIN_CAP_DS = {}

    ds_list = [a.ds] if a.ds else DATASETS
    done = load_done()
    new_file = not os.path.exists(RESULTS_CSV)
    fout = open(RESULTS_CSV, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "model", "seed", "setting", "prompt_len",
                    "AUC", "P", "R", "F1"])

    for ds in ds_list:
        D = sd.pickle.load(open(f"{DATA}/{ds}.pkl", "rb"))
        n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
        Xcol = D["X"]
        ynext = D["ynext"]
        for seed in SEEDS:
            tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(
                D, seed, False, ds)
            Xtr, ytr = Xcol[tr_idx], ynext[tr_idx]
            for P in PROMPTS:
                sd.build_model = lambda name, n_, nc_, m_, pl=P: \
                    build_model_pl(name, n_, nc_, m_, prompt_len=pl)
                models = {}
                if "central" in SETTINGS:
                    sd.set_seed(seed)
                    models["central"] = sd.train_central("neurallog", n, nc, mlen, Xtr, ytr)
                if "fed_iid" in SETTINGS:
                    sd.set_seed(seed + 1000)
                    models["fed_iid"] = sd.train_fed("neurallog", n, nc, mlen, Xtr, ytr, True)
                if "fed_noniid" in SETTINGS:
                    sd.set_seed(seed + 2000)
                    models["fed_noniid"] = sd.train_fed("neurallog", n, nc, mlen, Xtr, ytr, False)
                for setting in SETTINGS:
                    if (ds, str(seed), setting, str(P)) in done:
                        continue
                    auc, P_, R, F1 = eval_fused(
                        models[setting], D, tr_idx, te_idx, te_groups,
                        train_g, n_norm)
                    print(f"[{ds} s{seed} P{P} {setting}] F1={F1:.4f}")
                    w.writerow([ds, "neurallog", seed, setting, P,
                                f"{auc:.4f}", f"{P_:.4f}", f"{R:.4f}", f"{F1:.4f}"])
                fout.flush()
    fout.close()
    print(f"\n[done] -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
