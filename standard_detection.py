import os, csv, pickle, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd

from models_next import build_model

DATA = "revision_id/unified"
OUT_CSV = f"{DATA}/detection_all.csv"
SEEDS = [0, 1, 2]
DATASETS = ["BGL", "HDFS", "Thunderbird"]
MODELS = ["deeplog", "loganomaly", "neurallog", "onelog"]
SETTINGS = ["central", "fed_iid", "fed_noniid"]
GS = [1, 3]

K, ROUNDS, E = 10, 25, 2
CENTRAL_EP = 15
TRAIN_FRAC = 0.8
TRAIN_CAP = 80000
# per-dataset training cap (more data needed to learn normal top-1 transitions)
TRAIN_CAP_DS = {"HDFS": 600000, "Thunderbird": 250000}
BS = 512


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    torch.cuda.manual_seed_all(s)


def fed_avg(states, sizes):
    tot = sum(sizes); avg = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, n in zip(states, sizes):
            acc += s[k].float() * (n / tot)
        avg[k] = acc.to(states[0][k].dtype)
    return avg


def client_partition(y, K, iid):
    N = len(y)
    if iid:
        return np.array_split(np.random.permutation(N), K)
    order = np.argsort(y, kind="stable")
    shards = np.array_split(order, 2 * K)
    perm = np.random.permutation(2 * K)
    return [np.concatenate([shards[perm[2*c]], shards[perm[2*c+1]]])
            for c in range(K)]


def train_central(name, n, nc, mlen, X, y):
    m = build_model(name, n, nc, mlen).to(device)
    dl = DataLoader(TensorDataset(torch.from_numpy(X).long(),
                                  torch.from_numpy(y).long()), BS, shuffle=True,
                    pin_memory=True)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in tqdm(range(CENTRAL_EP), desc="central"):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
    return m


def train_fed(name, n, nc, mlen, X, y, iid):
    m = build_model(name, n, nc, mlen).to(device)
    chunks = client_partition(y, K, iid)
    for _ in tqdm(range(ROUNDS), desc=f"fed-{iid}"):
        states, sizes = [], []
        for c in chunks:
            local = build_model(name, n, nc, mlen).to(device)
            local.load_state_dict(m.state_dict())
            dl = DataLoader(TensorDataset(torch.from_numpy(X[c]).long(),
                                          torch.from_numpy(y[c]).long()),
                            BS, shuffle=True, pin_memory=True)
            opt = torch.optim.Adam(local.parameters(), 1e-3)
            for _ in range(E):
                local.train()
                for xb, yb in dl:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    opt.zero_grad()
                    F.cross_entropy(local(xb), yb).backward(); opt.step()
            states.append(local.state_dict()); sizes.append(len(c))
        m.load_state_dict(fed_avg(states, sizes))
    return m


@torch.no_grad()
def topg_miss(m, X, ynext, g, bs=4096):
    """Per-example anomaly flag: 1 if true next event is outside top-g."""
    m.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).long().to(device)
        yb = torch.from_numpy(ynext[i:i+bs]).long().to(device)
        topg = m(xb).topk(g, dim=1).indices
        hit = (topg == yb.unsqueeze(1)).any(1)
        out.append((~hit).cpu().numpy())
    return np.concatenate(out).astype(int)


@torch.no_grad()
def per_sample_loss(m, X, ynext, bs=4096):
    """Per-example cross-entropy (negative log-likelihood) of the true next event."""
    m.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).long().to(device)
        yb = torch.from_numpy(ynext[i:i+bs]).long().to(device)
        out.append(F.cross_entropy(m(xb), yb, reduction="none").cpu())
    return torch.cat(out).numpy()


def score_prf(ytrue, score):
    """From a continuous anomaly score: AUC and the F1-optimal threshold."""
    auc = roc_auc_score(ytrue, score)
    best_f1, bp, br, bt = -1, 0, 0, 0
    for t in np.quantile(score, np.linspace(0, 1, 400)):
        pred = (score > t).astype(int)
        P, R, F1 = prf(ytrue, pred)
        if F1 > best_f1:
            best_f1, bp, br, bt = F1, P, R, t
    return auc, bp, br, best_f1


def group_nll_scores(m, Xte, ynext_te, group, te_idx, te_groups, glabel):
    """Block-level detection from per-step NLL. A Fail block's few abnormal
    steps get diluted by the mean, so we also report max and 90th-percentile
    aggregations to surface the most abnormal step.
    Returns dict method -> block score array (aligned to te_groups)."""
    nll = per_sample_loss(m, Xte, ynext_te)
    g_of = group[te_idx]
    d = pd.DataFrame({"g": g_of, "l": nll})
    return {
        "nll_mean": d.groupby("g").l.mean().reindex(te_groups).fillna(0).values,
        "nll_max": d.groupby("g").l.max().reindex(te_groups).fillna(0).values,
        "nll_p90": d.groupby("g").l.quantile(0.9).reindex(te_groups).fillna(0).values,
    }


def prf(ytrue, ypred):
    tn, fp, fn, tp = confusion_matrix(ytrue, ypred, labels=[0, 1]).ravel()
    P = tp/(tp+fp) if tp+fp else 0
    R = tp/(tp+fn) if tp+fn else 0
    F1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    return P, R, F1


def split_data(D, seed, quick, ds):
    """Split by detection groups; return train/test sample indices + test groups."""
    group, glabel = D["group"], D["group_label"]
    rng = np.random.RandomState(seed)
    ng = np.where(glabel == 0)[0]
    ag = np.where(glabel == 1)[0]
    ng = ng[rng.permutation(len(ng))]
    ag = ag[rng.permutation(len(ag))]
    if quick:
        ng, ag = ng[:7000], ag[:4000]
    cut = int(len(ng) * TRAIN_FRAC)
    train_g_full, test_normal_g = ng[:cut], ng[cut:]
    train_g = train_g_full
    if not quick:
        cap = TRAIN_CAP_DS.get(ds, TRAIN_CAP)
        sizes = np.bincount(group, minlength=len(glabel))
        cum, keep = 0, []
        for g in train_g:
            if cum + sizes[g] > cap:
                break
            keep.append(g); cum += int(sizes[g])
        train_g = np.asarray(keep)
    te_groups = np.concatenate([test_normal_g, ag])
    n_norm = len(test_normal_g)

    def samples_of(gs):
        return np.where(np.isin(group, gs))[0]
    # DL training uses capped groups; count profile uses ALL normal train groups
    return samples_of(train_g), samples_of(te_groups), te_groups, train_g_full, n_norm


def count_z_scores(gcounts, groups, mean, std, BIG=1e6):
    """Block-level event-frequency anomaly (max per-event z-score).
    Events never seen in normal training yield BIG when present."""
    C = gcounts[groups].astype(np.float64)
    rare = mean < 1e-9
    z = np.where(rare[None, :],
                 np.where(C > 0, BIG, -np.inf),
                 (C - mean[None, :]) / (std[None, :] + 1e-9))
    z[C == 0] = -np.inf
    return z.max(axis=1)


def fuse_scores(seq_score, cnt_score, n_norm):
    """Final adaptive detector, leak-free thresholds calibrated only on the
    held-out NORMAL groups (first n_norm rows; anomalous labels are never used).
    - Session logs with a frequency profile (HDFS blocks): the event-count
      z-score is the primary signal (repetition/count anomalies), thresholded
      at the normal 99th pct; the sequence model only supplements at a very
      conservative 99.99th pct so its tail never dilutes count precision.
    - Continuous-flow logs (BGL/Thunderbird, no profile): the per-window NLL
      max is the primary signal at the normal 99th pct."""
    ytrue = np.r_[np.zeros(n_norm), np.ones(len(seq_score) - n_norm)]

    def z(s):
        mu, sd = s[:n_norm].mean(), s[:n_norm].std() + 1e-9
        return (s - mu) / sd

    seq_thr = np.quantile(seq_score[:n_norm], 0.99)
    if cnt_score is not None:
        cnt_thr = np.quantile(cnt_score[:n_norm], 0.99)
        seq_hi = np.quantile(seq_score[:n_norm], 0.9999)
        pred = (cnt_score > cnt_thr) | (seq_score > seq_hi)
        score = np.maximum(z(seq_score), z(cnt_score))
    else:
        pred = seq_score > seq_thr
        score = z(seq_score)
    auc = roc_auc_score(ytrue, score)
    P, R, F1 = prf(ytrue, pred.astype(int))
    return auc, P, R, F1


def run_one(ds, model, seed, need, quick):
    D = pickle.load(open(f"{DATA}/{ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    tr_idx, te_idx, te_groups, train_g, n_norm = split_data(D, seed, quick, ds)
    group, glabel = D["group"], D["group_label"]

    # block-level event-frequency profile (only HDFS ships group_counts);
    # linear summary statistics aggregate losslessly under federation.
    gcounts = D.get("group_counts")
    profile = None
    if gcounts is not None:
        Ctr = gcounts[train_g].astype(np.float64)
        profile = (Ctr.mean(0), Ctr.std(0))

    Xcol = D["Xchar"] if model == "onelog" else D["X"]
    Xtr, Xte = Xcol[tr_idx], Xcol[te_idx]
    ytr = D["ynext"][tr_idx]
    ynext_te = D["ynext"][te_idx]
    ytrue_g = glabel[te_groups]

    def eval_setting(m):
        out = {}
        for g in GS:
            miss = topg_miss(m, Xte, ynext_te, g)
            d = pd.DataFrame({"g": group[te_idx], "m": miss}).groupby("g").m.max()
            pred = d.reindex(te_groups).fillna(0).values.astype(int)
            P, R, F1 = prf(ytrue_g, pred)
            out[f"topk{g}"] = (None, P, R, F1)
        nll_scores = group_nll_scores(m, Xte, ynext_te, group, te_idx, te_groups, glabel)
        for nm, s in nll_scores.items():
            out[nm] = score_prf(ytrue_g, s)
        cnt = None
        if profile is not None:
            cnt = count_z_scores(gcounts, te_groups, *profile)
            out["count_z"] = score_prf(ytrue_g, cnt)
        # final fused detector, leak-free normal-calibration threshold
        out["fused"] = fuse_scores(nll_scores["nll_max"], cnt, n_norm)
        return out

    res = {}
    if "central" in need:
        set_seed(seed)
        res["central"] = eval_setting(train_central(model, n, nc, mlen, Xtr, ytr))
    if "fed_iid" in need:
        set_seed(seed + 1000)
        res["fed_iid"] = eval_setting(train_fed(model, n, nc, mlen, Xtr, ytr, True))
    if "fed_noniid" in need:
        set_seed(seed + 2000)
        res["fed_noniid"] = eval_setting(train_fed(model, n, nc, mlen, Xtr, ytr, False))
    return res


OUT_METHODS = ["topk1", "topk3", "nll_mean", "nll_max", "nll_p90",
               "count_z", "fused"]


def load_done():
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV) as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["model"], r["seed"],
                          r["setting"], r["method"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ds", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    global device, OUT_CSV
    device = torch.device(a.device)
    if a.quick:
        OUT_CSV = f"{DATA}/detection_quick.csv"

    done = load_done()
    new_file = not os.path.exists(OUT_CSV)
    fout = open(OUT_CSV, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "model", "seed", "setting", "method", "AUC", "P", "R", "F1"])

    ds_list = [a.ds] if a.ds else DATASETS
    model_list = [a.model] if a.model else MODELS
    seed_list = [a.seed] if a.seed is not None else SEEDS
    for ds in ds_list:
        for model in model_list:
            for seed in seed_list:
                need = [s for s in SETTINGS if any(
                    (ds, model, str(seed), s, m) not in done for m in OUT_METHODS)]
                if not need:
                    continue
                print(f"[{ds} | {model} | s{seed}] need={need}")
                res = run_one(ds, model, seed, need, a.quick)
                for setting in need:
                    r = res[setting]
                    for m in OUT_METHODS:
                        if m not in r:
                            continue
                        tup = r[m]
                        if tup[0] is None:
                            auc_s, P, R, F1 = "", tup[1], tup[2], tup[3]
                        else:
                            auc_s, P, R, F1 = f"{tup[0]:.4f}", tup[1], tup[2], tup[3]
                        w.writerow([ds, model, seed, setting, m, auc_s,
                                    f"{P:.4f}", f"{R:.4f}", f"{F1:.4f}"])
                fout.flush()
    fout.close()
    print(f"\n[done] detection -> {OUT_CSV}")


if __name__ == "__main__":
    main()
