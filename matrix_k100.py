"""Unified K=100 main matrices for the final manuscript.

This replaces the K=10, per-dataset label-skew federation used in the older
mia_matrix/run_all main tables, so that ALL reported federated numbers share the
same protocol as the new DP / TrimmedMean experiments and Table II:

  * IID      : per-dataset random partition, K=100 clients, participation C=0.5
               (fed_core.run_fed, fed_avg).
  * non-IID  : cross-dataset skew, exactly as in the manuscript and in
               robust_experiment. The three datasets are merged into a single
               event-id space (BGL 0..349, HDFS +350, Thunderbird +379, total
               3715) and split into 34 / 33 / 33 client pools, each client
               seeing only ONE dataset. Full participation C=1.0 (robust
               protocol) so every system contributes every round.

  --mode detection : detection training caps (standard_detection split_data);
                     central 15 epochs, IID R=50/le=1, cross R=60/le=3; reports
                     the leak-free fused detector F1/P/R/AUC per dataset.
  --mode mia       : fixed N_FIX=80000 victims trained to a worst-case
                     memorization state; central 300 epochs, federated
                     R=60/le=2; reports loss-based MIA AUC/P/R/FPR/F1 on the
                     rare-n-gram member vs genuinely-novel non-member pools.

Four models x three datasets x three seeds. Central training is per-dataset and
identical regardless of the federation protocol, so it is reused. The cross-
dataset model is trained once per (model, seed) and evaluated on all three
datasets. Fully resumable (CSV done-set); run different pieces on GPUs:

  python matrix_k100.py --mode detection --ds BGL --device cuda:0
  python matrix_k100.py --mode detection --ds HDFS --device cuda:1
  python matrix_k100.py --mode detection --ds Thunderbird --device cuda:2
  python matrix_k100.py --mode detection --cross --model neurallog --seed 0 --device cuda:0
"""
import os, csv, copy, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import standard_detection as sd
from models_next import build_model
from fed_core import run_fed, fed_avg, select_clients
import mia_matrix as mm

DATASETS = ["BGL", "HDFS", "Thunderbird"]
MODELS = ["deeplog", "loganomaly", "neurallog", "onelog"]
SEEDS = [0, 1, 2]
K = 100
POOL_COUNTS = {"BGL": 34, "HDFS": 33, "Thunderbird": 33}
N_FIX = 80000
# cross-dataset (non-IID) uses a balanced per-dataset cap equal to the
# smallest per-dataset detection cap (BGL=80000), so the three pools stay
# equal-sized (~2400 samples/client) while BGL's rare normal n-grams are
# covered as fully as in the per-dataset IID setting (50000 left a few
# normal windows under-fit on OneLog, whose high NLL tail lifted the fused
# normal-p99 threshold and collapsed recall).
CROSS_DET_CAP = 80000

# per-mode federation hyperparameters
CFG = {
    "detection": {"iid_rounds": 50, "iid_le": 1, "iid_C": 0.5,
                  "cross_rounds": 60, "cross_le": 3, "cross_C": 1.0,
                  "central_ep": 15},
    "mia":       {"iid_rounds": 60, "iid_le": 2, "iid_C": 0.5,
                  "cross_rounds": 60, "cross_le": 3, "cross_C": 1.0,
                  "central_ep": 300},
}


def out_paths(mode, tag=""):
    d = f"{sd.DATA}/main_k100"
    os.makedirs(d, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return (f"{d}/{mode}_k100{suffix}.csv",
            f"{d}/predlogs_{mode}{suffix}")


def load_ds(ds):
    D = sd.pickle.load(open(f"{sd.DATA}/{ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    return D, n, nc, mlen


def use_char(model):
    return model == "onelog"


def mia_pools(X, ynext, tr_idx, D, split, rng):
    """Rare n-gram members (frequency 1..2) vs genuinely-novel non-members,
    exactly the mia_matrix construction. Returns balanced index arrays."""
    group, glabel = D["group"], D["group_label"]
    te_idx, te_groups, train_g, n_norm = split
    keys = [tuple(x) for x in X[tr_idx]]
    tcount = Counter(keys)
    fv = np.array([tcount[tuple(x)] for x in X[tr_idx]])
    member_pool = tr_idx[(fv >= 1) & (fv <= 2)]
    normal_test_g = te_groups[:n_norm]
    norm_te_idx = te_idx[np.isin(group[te_idx], normal_test_g)]
    train_keys = set(tcount.keys())
    novel = np.array([tuple(x) not in train_keys for x in X[norm_te_idx]])
    non_pool = norm_te_idx[novel]
    m_n = min(len(member_pool), len(non_pool))
    member_idx = member_pool[rng.permutation(len(member_pool))[:m_n]]
    non_idx = non_pool[rng.permutation(len(non_pool))[:m_n]]
    return member_idx, non_idx


def det_fused(m, D, split, offset, char):
    """Leak-free fused detector F1 for one dataset; event ids shifted by offset
    (0 for per-dataset models)."""
    te_idx, te_groups, train_g, n_norm = split
    group, glabel = D["group"], D["group_label"]
    Xte = D["Xchar"][te_idx] if char else D["X"][te_idx] + offset
    ynext_te = D["ynext"][te_idx] + offset
    nll = sd.group_nll_scores(m, Xte, ynext_te, group, te_idx, te_groups, glabel)
    gcounts = D.get("group_counts")
    cnt = None
    if gcounts is not None:
        Ctr = gcounts[train_g].astype(np.float64)
        cnt = sd.count_z_scores(gcounts, te_groups, Ctr.mean(0), Ctr.std(0))
    return sd.fuse_scores(nll["nll_max"], cnt, n_norm)


# ----------------------------------------------------------------------------
# per-dataset central + fed IID
# ----------------------------------------------------------------------------

def run_per_dataset(mode, ds, model, seed, device, pred_dir):
    cfg = CFG[mode]
    D, n, nc, mlen = load_ds(ds)
    char = use_char(model)
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, seed, False, ds)
    split = (te_idx, te_groups, train_g, n_norm)
    # MIA victims use a fixed 80k subset to a memorization state; detection uses
    # the full (per-dataset capped) split.
    if mode == "mia" and len(tr_idx) > N_FIX:
        r0 = np.random.RandomState(seed)
        train_tr = tr_idx[r0.permutation(len(tr_idx))[:N_FIX]]
    else:
        train_tr = tr_idx
    Xtr = D["Xchar"][train_tr] if char else D["X"][train_tr]
    ytr = D["ynext"][train_tr]

    rows = []
    # central victim
    sd.set_seed(seed)
    sd.CENTRAL_EP = cfg["central_ep"]
    mc = sd.train_central(model, n, nc, mlen, Xtr, ytr)
    if mode == "detection":
        auc, P, R, F1 = det_fused(mc, D, split, 0, char)
        rows.append((ds, model, seed, "central", auc, P, R, F1))
    else:
        rng = np.random.RandomState(seed)
        mi, ni = mia_pools(D["X"], D["ynext"], train_tr, D, split, rng)
        mm.PRED = pred_dir
        r = mm.attack(mc, D["Xchar"] if char else D["X"], mi, ni,
                      D["ynext"], f"{ds}_{model}_s{seed}_central")
        rows.append((ds, model, seed, "central", r["auc"], r["P"], r["R"],
                     r["FPR"], r["F1"]))

    # federated IID (K=100, C=0.5)
    sd.set_seed(seed + 1000)
    mf = run_fed(model, n, nc, mlen, Xtr, ytr, K,
                 cfg["iid_rounds"], cfg["iid_le"], cfg["iid_C"],
                 dp_sigma=0.0, seed=seed, device=device)
    if mode == "detection":
        auc, P, R, F1 = det_fused(mf, D, split, 0, char)
        rows.append((ds, model, seed, "fed_iid", auc, P, R, F1))
    else:
        rng = np.random.RandomState(seed)
        mi, ni = mia_pools(D["X"], D["ynext"], train_tr, D, split, rng)
        mm.PRED = pred_dir
        r = mm.attack(mf, D["Xchar"] if char else D["X"], mi, ni,
                      D["ynext"], f"{ds}_{model}_s{seed}_fed_iid")
        rows.append((ds, model, seed, "fed_iid", r["auc"], r["P"], r["R"],
                     r["FPR"], r["F1"]))
    return rows


# ----------------------------------------------------------------------------
# cross-dataset non-IID (34/33/33 merged event space), train once per model/seed
# ----------------------------------------------------------------------------

def build_cross_data(mode, seed, model):
    """Returns (total_n, chunks, per-dataset eval bundles)."""
    char = use_char(model)
    Ds, splits = {}, {}
    offsets, off = {}, 0
    tr_store = {}
    for ds in DATASETS:
        D, n, nc, mlen = load_ds(ds)
        Ds[ds] = D
        offsets[ds] = off
        tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, seed, False, ds)
        splits[ds] = (te_idx, te_groups, train_g, n_norm)
        cap = N_FIX if mode == "mia" else CROSS_DET_CAP
        if len(tr_idx) > cap:
            r0 = np.random.RandomState(seed)
            tr_idx = tr_idx[r0.permutation(len(tr_idx))[:cap]]
        tr_store[ds] = tr_idx
        off += n

    rng = np.random.RandomState(seed)
    chunks = []  # (X_client, y_client)
    for ds in DATASETS:
        D = Ds[ds]
        o = offsets[ds]
        tr_idx = tr_store[ds]
        n_pool = POOL_COUNTS[ds]
        parts = np.array_split(rng.permutation(len(tr_idx)), n_pool)
        for p in parts:
            idx = tr_idx[p]
            Xc = D["Xchar"][idx] if char else D["X"][idx] + o
            yc = D["ynext"][idx] + o
            chunks.append((Xc, yc))
    total_n = off
    return total_n, Ds, offsets, splits, chunks, rng


def train_cross(model, seed, device, cfg):
    char = use_char(model)
    total_n, Ds, offsets, splits, chunks, rng = build_cross_data(
        cfg["_mode"], seed, model)
    nc, mlen = 130, 40
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = build_model(model, total_n, nc, mlen).to(device)
    for rnd in range(cfg["cross_rounds"]):
        active = select_clients(K, cfg["cross_C"], rng)
        old = m.state_dict()
        states, sizes = [], []
        for ci in active:
            Xc, yc = chunks[ci]
            local = build_model(model, total_n, nc, mlen).to(device)
            local.load_state_dict(old)
            opt = torch.optim.Adam(local.parameters(), 1e-3)
            dl = DataLoader(TensorDataset(torch.from_numpy(Xc).long(),
                                          torch.from_numpy(yc).long()),
                            512, shuffle=True)
            for _ in range(cfg["cross_le"]):
                local.train()
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); F.cross_entropy(local(xb), yb).backward(); opt.step()
            states.append(local.state_dict()); sizes.append(len(yc))
        m.load_state_dict(fed_avg(states, sizes))
        if (rnd + 1) % 10 == 0:
            print(f"[cross {model} s{seed}] round {rnd+1}/{cfg['cross_rounds']}")
    return m, Ds, offsets, splits, char


def eval_cross(mode, m, model, seed, Ds, offsets, splits, char, pred_dir):
    rows = []
    for ds in DATASETS:
        D = Ds[ds]
        o = offsets[ds]
        split = splits[ds]
        if mode == "detection":
            auc, P, R, F1 = det_fused(m, D, split, o, char)
            rows.append((ds, model, seed, "fed_noniid", auc, P, R, F1))
        else:
            # reconstruct the same capped training indices and rare-n-gram pools
            tr_idx = sd.split_data(D, seed, False, ds)[0]
            if len(tr_idx) > N_FIX:
                r0 = np.random.RandomState(seed)
                tr_idx = tr_idx[r0.permutation(len(tr_idx))[:N_FIX]]
            rng = np.random.RandomState(seed)
            mi, ni = mia_pools(D["X"], D["ynext"], tr_idx, D, split, rng)
            # shifted global arrays for the merged-space model
            Xeval = D["Xchar"] if char else D["X"] + o
            yeval = D["ynext"] + o
            mm.PRED = pred_dir
            r = mm.attack(m, Xeval, mi, ni, yeval,
                          f"{ds}_{model}_s{seed}_fed_noniid")
            rows.append((ds, model, seed, "fed_noniid", r["auc"], r["P"], r["R"],
                         r["FPR"], r["F1"]))
    return rows


# ----------------------------------------------------------------------------
# CSV / resume
# ----------------------------------------------------------------------------

def load_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            done.add((r["dataset"], r["model"], r["seed"], r["setting"]))
    return done


def write_rows(csv_path, pred_dir, mode, rows):
    new = not os.path.exists(csv_path)
    os.makedirs(pred_dir, exist_ok=True)
    fout = open(csv_path, "a", newline="")
    w = csv.writer(fout)
    if new:
        if mode == "detection":
            w.writerow(["dataset", "model", "seed", "setting",
                        "fused_AUC", "fused_P", "fused_R", "fused_F1"])
        else:
            w.writerow(["dataset", "model", "seed", "setting",
                        "MIA_AUC", "MIA_P", "MIA_R", "MIA_FPR", "MIA_F1"])
    for row in rows:
        w.writerow([row[0], row[1], row[2], row[3]] +
                   [f"{v:.4f}" for v in row[4:]])
    fout.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["detection", "mia"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ds", default=None, choices=DATASETS)
    ap.add_argument("--model", default=None, choices=MODELS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cross", action="store_true",
                    help="run only the cross-dataset non-IID piece")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override IID and cross rounds (Table III grid)")
    ap.add_argument("--C", type=float, default=None,
                    help="override IID participation ratio (Table III grid)")
    ap.add_argument("--cross-C", type=float, default=None,
                    help="override cross participation (default 1.0)")
    ap.add_argument("--cross-le", type=int, default=None,
                    help="override cross local epochs (use 1 to match the IID grid)")
    ap.add_argument("--iid-le", type=int, default=None,
                    help="override IID local epochs (Table III grid; e.g. 3)")
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    cfg = dict(CFG[a.mode]); cfg["_mode"] = a.mode
    # Table III grid overrides (detection only). Non-default R/C writes a
    # separate CSV so the main matrix and grid never collide.
    tag = ""
    if a.rounds is not None:
        cfg["iid_rounds"] = a.rounds; cfg["cross_rounds"] = a.rounds
        tag = f"R{a.rounds}"
    if a.C is not None:
        cfg["iid_C"] = a.C
        if a.cross_C is None:
            cfg["cross_C"] = a.C
        tag = (tag + f"_C{a.C}") if tag else f"C{a.C}"
    if a.cross_C is not None:
        cfg["cross_C"] = a.cross_C
    if a.iid_le is not None:
        cfg["iid_le"] = a.iid_le
        tag = (tag + f"_iidle{a.iid_le}") if tag else f"iidle{a.iid_le}"
    if a.cross_le is not None:
        cfg["cross_le"] = a.cross_le
        tag = (tag + f"_le{a.cross_le}") if tag else f"le{a.cross_le}"
    csv_path, pred_dir = out_paths(a.mode, tag)
    done = load_done(csv_path)

    models = [a.model] if a.model else MODELS
    seeds = [a.seed] if a.seed is not None else SEEDS

    if a.cross:
        for model in models:
            for seed in seeds:
                m, Ds, offsets, splits, char = train_cross(
                    model, seed, sd.device, cfg)
                rows = eval_cross(a.mode, m, model, seed, Ds, offsets,
                                  splits, char, pred_dir)
                rows = [r for r in rows
                        if (r[0], r[1], str(r[2]), r[3]) not in done]
                for r in rows:
                    print(f"[{a.mode}] {r[0]} {r[1]} s{r[2]} {r[3]} "
                          + ("F1=%.4f" % r[-1] if a.mode == "detection"
                             else "AUC=%.4f F1=%.4f" % (r[4], r[8])))
                if rows:
                    write_rows(csv_path, pred_dir, a.mode, rows)
        return

    dss = [a.ds] if a.ds else DATASETS
    for model in models:
        for ds in dss:
            for seed in seeds:
                want = [(ds, model, str(seed), "central"),
                        (ds, model, str(seed), "fed_iid")]
                if all(w in done for w in want):
                    continue
                rows = run_per_dataset(a.mode, ds, model, seed, sd.device,
                                       pred_dir)
                rows = [r for r in rows
                        if (r[0], r[1], str(r[2]), r[3]) not in done]
                for r in rows:
                    print(f"[{a.mode}] {r[0]} {r[1]} s{r[2]} {r[3]} "
                          + ("AUC=%.4f P=%.4f R=%.4f F1=%.4f" % (r[4], r[5], r[6], r[7])
                             if a.mode == "detection"
                             else "AUC=%.4f FPR=%.4f F1=%.4f" % (r[4], r[7], r[8])))
                if rows:
                    write_rows(csv_path, pred_dir, a.mode, rows)


if __name__ == "__main__":
    main()
