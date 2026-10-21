"""R1/Contribution-3 (manuscript Fig.6): robust aggregation under data
poisoning. One config per run (parallelize on GPUs).

Merges the three datasets into a unified event-id space:
  BGL ids 0..(n_BGL-1), HDFS shifted by n_BGL, Thunderbird shifted by
  n_BGL+n_HDFS. Each dataset is split identically to standard_detection.
  - mode=iid    : 100 clients receive a random partition of the merged pool.
  - mode=noniid : 100 clients split into 34/33/33 pools, each seeing only one
                  dataset (cross-dataset skew, as in the manuscript).
Each round, a fraction of clients are malicious (label corruption of the
next-event target); aggregation is FedAvg or coordinate-wise Trimmed Mean.
The per-round F1 on a fixed normal+anomaly test set is logged.
"""
import os, csv, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import standard_detection as sd
from models_next import build_model
from fed_core import fed_avg, trimmed_mean

DATASETS = ["BGL", "HDFS", "Thunderbird"]
K = 100
POOL_COUNTS = {"BGL": 34, "HDFS": 33, "Thunderbird": 33}
OUT_DIR = f"{sd.DATA}/robust_curves"
TRAIN_CAP = 50000
N_TEST = 8000


def load_merged(seed):
    Ds = {ds: pickle.load(open(f"{sd.DATA}/{ds}.pkl", "rb")) for ds in DATASETS}
    n_ev = {ds: Ds[ds]["num_event_ids"] for ds in DATASETS}
    offsets, off = {}, 0
    for ds in DATASETS:
        offsets[ds] = off
        off += n_ev[ds]
    total_n = off

    trX, trXc, trY, trD = [], [], [], []
    teX, teXc, teY, teD = [], [], [], []
    for ds in DATASETS:
        D = Ds[ds]
        tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(
            D, seed, False, ds)
        rng = np.random.RandomState(seed)
        tcap = min(TRAIN_CAP, len(tr_idx))
        tr_sel = tr_idx[rng.permutation(len(tr_idx))[:tcap]]
        # stratified test: normal + anomaly
        ylab = D["ylab"]
        norm_te = te_idx[ylab[te_idx] == 0]
        anom_te = te_idx[ylab[te_idx] == 1]
        n_an = min(N_TEST // 2, len(anom_te))
        n_no = min(N_TEST - n_an, len(norm_te))
        sel_norm = norm_te[rng.permutation(len(norm_te))[:n_no]]
        sel_anom = anom_te[rng.permutation(len(anom_te))[:n_an]]
        te_sel = np.concatenate([sel_norm, sel_anom])
        o = offsets[ds]
        trX.append(D["X"][tr_sel] + o); trXc.append(D["Xchar"][tr_sel])
        trY.append(D["ynext"][tr_sel] + o); trD.append(D["ylab"][tr_sel])
        teX.append(D["X"][te_sel] + o); teXc.append(D["Xchar"][te_sel])
        teY.append(D["ynext"][te_sel] + o); teD.append(D["ylab"][te_sel])
    return (total_n,
            np.concatenate(trX), np.concatenate(trXc),
            np.concatenate(trY), np.concatenate(trD),
            np.concatenate(teX), np.concatenate(teXc),
            np.concatenate(teY), np.concatenate(teD),
            Ds, offsets)


def client_pools(mode, N, seed, offsets):
    rng = np.random.RandomState(seed)
    if mode == "iid":
        return np.array_split(rng.permutation(N), K)
    # cross-dataset: need to know each dataset's rows in the merged train set.
    # Rebuilt by the caller via offsets; passed via global pool slices below.
    return None


def top1_f1(model, Xte, ynext_te, ylab_te, device):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 4096):
            xb = torch.from_numpy(Xte[i:i+4096]).long().to(device)
            yb = torch.from_numpy(ynext_te[i:i+4096]).long().to(device)
            p = model(xb).argmax(1)
            preds.append((p != yb).cpu().numpy())
    pred = np.concatenate(preds).astype(int)
    return sd.prf(ylab_te, pred)  # returns P, R, F1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="noniid", choices=["iid", "noniid"])
    ap.add_argument("--agg", default="fedavg", choices=["fedavg", "trimmed"])
    ap.add_argument("--poison", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--attack", default="signflip",
                    choices=["none", "labelflip", "signflip"])
    ap.add_argument("--scale", type=float, default=-4.0)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--le", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{a.mode}_{a.agg}_p{a.poison}_{a.attack}_b{a.beta}_s{a.seed}"
    out_csv = f"{OUT_DIR}/{tag}.csv"

    (total_n, trX, trXc, trY, trD, teX, teXc, teY, teD,
     Ds, offsets) = load_merged(a.seed)
    n, nc, mlen = total_n, 130, 40

    # build client pools
    rng = np.random.RandomState(a.seed)
    if a.mode == "iid":
        chunks = list(np.array_split(rng.permutation(len(trX)), K))
    else:
        chunks = []
        for ds in DATASETS:
            o = offsets[ds]
            rows = np.where((trX.min(1) >= o) &
                            (trX.min(1) < o + Ds[ds]["num_event_ids"]))[0]
            n_pool = POOL_COUNTS[ds]
            chunks.extend(np.array_split(rng.permutation(rows), n_pool))

    m = build_model("neurallog", n, nc, mlen).to(device)
    # Threat model: a FIXED set of Byzantine clients.
    #  - labelflip: corrupt clients apply one shared permutation of the target.
    #  - signflip (default): corrupt clients send a sign-flipped, amplified
    #    update  global + scale*(local-global)  (scale<0), a standard strong
    #    Byzantine/model-poisoning attack that FedAvg cannot average out.
    arng = np.random.RandomState(a.seed + 999)
    if a.attack != "none" and a.poison > 0:
        n_mal = max(1, int(round(a.poison * K)))
        malicious = set(arng.choice(K, n_mal, replace=False))
        flip_pi = arng.permutation(n) if a.attack == "labelflip" else None
    else:
        malicious, flip_pi = set(), None

    fout = open(out_csv, "w", newline="")
    w = csv.writer(fout); w.writerow(["round", "P", "R", "F1"])
    for r in range(a.rounds):
        global_state = m.state_dict()
        states, sizes = [], []
        for ci, c in enumerate(chunks):
            Xc, yc = trX[c], trY[c]
            if ci in malicious and a.attack == "labelflip":
                yc = flip_pi[yc.copy()]
            local = build_model("neurallog", n, nc, mlen).to(device)
            local.load_state_dict(global_state)
            opt = torch.optim.Adam(local.parameters(), 1e-3)
            dl = DataLoader(TensorDataset(torch.from_numpy(Xc).long(),
                                          torch.from_numpy(yc).long()),
                            512, shuffle=True)
            for _ in range(a.le):
                local.train()
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); F.cross_entropy(local(xb), yb).backward()
                    opt.step()
            if ci in malicious and a.attack == "signflip":
                lstate = local.state_dict()
                adv = {}
                for k in global_state:
                    adv[k] = global_state[k] + a.scale * (lstate[k] - global_state[k])
                local.load_state_dict(adv)
            states.append(local.state_dict()); sizes.append(len(c))
        if a.agg == "trimmed":
            m.load_state_dict(trimmed_mean(states, sizes, a.beta))
        else:
            m.load_state_dict(fed_avg(states, sizes))
        P, R, F1 = top1_f1(m, teX, teY, teD, device)
        w.writerow([r, f"{P:.4f}", f"{R:.4f}", f"{F1:.4f}"])
        fout.flush()
        print(f"[{tag}] r{r} F1={F1:.4f}")
    fout.close()
    print(f"\n[done] -> {out_csv}")


if __name__ == "__main__":
    main()
