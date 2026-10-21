"""Fair-budget MIA matrix for LogFed (drop-in extension of matrix_k100.py).

Run this INSTEAD of matrix_k100.py for the MIA table; it does not change
the detection path. New flags implement the controlled comparison:

  * --full-detector : in mia mode, the FEDERATED IID victim is trained on the
                      exact detector split (BGL 80k / HDFS 600k / Thunderbird
                      250k), i.e. the deployed detector, instead of being
                      re-subsampled to N_FIX=80k. The central victim still
                      uses the fixed 80k pool. (Cross non-IID already uses
                      80k/dataset, identical to the detection cross cap.)
  * --central-ep N  : override central epochs. Use 75 for the IID-matched
                      central (rCR=3*0.5*50), 150 for the non-IID-matched
                      pooled central (rCR=3*1.0*50), and 300 for the
                      converged worst-case upper bound.
  * --only-central  : skip federated training. Per-dataset: report only the
                      centralized row. With --cross: train a single POOLED
                      merged-space centralized model over the same three
                      dataset pool the cross federated model sees (80k each),
                      evaluated on all three datasets, setting=central_noniid.

Suggested three-layer comparison (all at r=3, R=50):
  federated deployed   --mode mia --full-detector --rounds 50 --iid-le 3 --cross-le 3
  central IID-matched  --mode mia --only-central --central-ep 75
  central non-IID-match--mode mia --only-central --cross --central-ep 150
  central converged    --mode mia --only-central --central-ep 300
  central non-IID conv --mode mia --only-central --cross --central-ep 300

Each override writes its own tagged CSV (resumable, never overwrites others).
"""
import os, csv, argparse
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
CROSS_DET_CAP = 80000

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


def cap_indices(tr_idx, cap, seed):
    if len(tr_idx) > cap:
        r0 = np.random.RandomState(seed)
        return tr_idx[r0.permutation(len(tr_idx))[:cap]]
    return tr_idx


def mia_pools(X, ynext, tr_idx, D, split, rng):
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


def mia_row(mode, m, D, char, offset, X_pool_tr, split, ds, model,
            seed, setting, pred_dir):
    rng = np.random.RandomState(seed)
    mi, ni = mia_pools(D["X"], D["ynext"], X_pool_tr, D, split, rng)
    Xeval = D["Xchar"] if char else D["X"] + offset
    yeval = D["ynext"] + offset
    mm.PRED = pred_dir
    r = mm.attack(m, Xeval, mi, ni, yeval,
                  f"{ds}_{model}_s{seed}_{setting}")
    return (ds, model, seed, setting, r["auc"], r["P"], r["R"],
            r["FPR"], r["F1"])


# ---------------- per-dataset central + fed IID ----------------

def run_per_dataset(mode, ds, model, seed, device, pred_dir,
                    only_central=False, full_detector=False):
    cfg = CFG[mode]
    D, n, nc, mlen = load_ds(ds)
    char = use_char(model)
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, seed, False, ds)
    split = (te_idx, te_groups, train_g, n_norm)

    # Central MIA victim: fixed 80k pool (worst-case memorization). Detection
    # central: full (per-dataset capped) split.
    central_tr = cap_indices(tr_idx, N_FIX, seed) if mode == "mia" else tr_idx
    # Federated MIA victim: with --full-detector it is the exact deployed
    # detector split; otherwise the same fixed 80k pool as before.
    fed_tr = tr_idx if (mode == "mia" and full_detector) else central_tr

    rows = []
    # ---- centralized ----
    sd.set_seed(seed)
    sd.CENTRAL_EP = cfg["central_ep"]
    Xc = D["Xchar"][central_tr] if char else D["X"][central_tr]
    yc = D["ynext"][central_tr]
    mc = sd.train_central(model, n, nc, mlen, Xc, yc)
    if mode == "detection":
        auc, P, R, F1 = det_fused(mc, D, split, 0, char)
        rows.append((ds, model, seed, "central", auc, P, R, F1))
    else:
        rows.append(mia_row(mode, mc, D, char, 0, central_tr, split,
                            ds, model, seed, "central", pred_dir))
    if only_central:
        return rows

    # ---- federated IID (K=100, C=0.5) ----
    sd.set_seed(seed + 1000)
    Xf = D["Xchar"][fed_tr] if char else D["X"][fed_tr]
    yf = D["ynext"][fed_tr]
    mf = run_fed(model, n, nc, mlen, Xf, yf, K,
                 cfg["iid_rounds"], cfg["iid_le"], cfg["iid_C"],
                 dp_sigma=0.0, seed=seed, device=device)
    if mode == "detection":
        auc, P, R, F1 = det_fused(mf, D, split, 0, char)
        rows.append((ds, model, seed, "fed_iid", auc, P, R, F1))
    else:
        rows.append(mia_row(mode, mf, D, char, 0, fed_tr, split,
                            ds, model, seed, "fed_iid", pred_dir))
    return rows


# ---------------- cross-dataset non-IID ----------------

def build_cross_data(mode, seed, model):
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
        tr_store[ds] = cap_indices(tr_idx, cap, seed)
        off += n
    rng = np.random.RandomState(seed)
    chunks = []
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


def train_cross_central(model, seed, device, cfg):
    """Pooled centralized model on the SAME merged three-dataset pool the
    cross federated model sees (80k/dataset), for cfg central epochs.
    This is the fair central counterpart to the non-IID federated model."""
    char = use_char(model)
    total_n, Ds, offsets, splits, chunks, rng = build_cross_data(
        cfg["_mode"], seed, model)
    nc, mlen = 130, 40
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    Xall = np.concatenate([c[0] for c in chunks])
    yall = np.concatenate([c[1] for c in chunks])
    m = build_model(model, total_n, nc, mlen).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xall).long(),
                                  torch.from_numpy(yall).long()),
                    512, shuffle=True)
    for ep in range(cfg["central_ep"]):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
        if (ep + 1) % 25 == 0:
            print(f"[cross-central {model} s{seed}] epoch {ep+1}/{cfg['central_ep']}")
    return m, Ds, offsets, splits, char


def eval_cross(mode, m, model, seed, Ds, offsets, splits, char, pred_dir,
               setting="fed_noniid"):
    rows = []
    for ds in DATASETS:
        D = Ds[ds]
        o = offsets[ds]
        split = splits[ds]
        if mode == "detection":
            auc, P, R, F1 = det_fused(m, D, split, o, char)
            rows.append((ds, model, seed, setting, auc, P, R, F1))
        else:
            tr_idx = cap_indices(sd.split_data(D, seed, False, ds)[0],
                                 N_FIX, seed)
            rows.append(mia_row(mode, m, D, char, o, tr_idx, split,
                                ds, model, seed, setting, pred_dir))
    return rows


# ---------------- CSV / resume ----------------

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
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--cross-C", type=float, default=None)
    ap.add_argument("--cross-le", type=int, default=None)
    ap.add_argument("--iid-le", type=int, default=None)
    # ---- fair-budget additions ----
    ap.add_argument("--central-ep", type=int, default=None,
                    help="central epochs override (75 IID-matched / 150 "
                         "non-IID-matched / 300 converged)")
    ap.add_argument("--full-detector", action="store_true",
                    help="mia: federated IID victim uses the exact detector "
                         "split (no 80k subsample)")
    ap.add_argument("--only-central", action="store_true",
                    help="report only centralized row(s); with --cross train a "
                         "pooled merged-space central model")
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    cfg = dict(CFG[a.mode]); cfg["_mode"] = a.mode

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
    if a.central_ep is not None:
        cfg["central_ep"] = a.central_ep
        tag = (tag + f"_cep{a.central_ep}") if tag else f"cep{a.central_ep}"
    if a.full_detector:
        tag = (tag + "_fulldet") if tag else "fulldet"
    if a.only_central:
        tag = (tag + "_centralonly") if tag else "centralonly"
    csv_path, pred_dir = out_paths(a.mode, tag)
    done = load_done(csv_path)

    models = [a.model] if a.model else MODELS
    seeds = [a.seed] if a.seed is not None else SEEDS

    if a.cross:
        for model in models:
            for seed in seeds:
                if a.only_central:
                    m, Ds, offsets, splits, char = train_cross_central(
                        model, seed, sd.device, cfg)
                    rows = eval_cross(a.mode, m, model, seed, Ds, offsets,
                                      splits, char, pred_dir,
                                      setting="central_noniid")
                else:
                    m, Ds, offsets, splits, char = train_cross(
                        model, seed, sd.device, cfg)
                    rows = eval_cross(a.mode, m, model, seed, Ds, offsets,
                                      splits, char, pred_dir,
                                      setting="fed_noniid")
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
                want = [(ds, model, str(seed),
                         "central" if a.only_central else "fed_iid")]
                if not a.only_central:
                    want = [(ds, model, str(seed), "central"),
                            (ds, model, str(seed), "fed_iid")]
                if all(w in done for w in want):
                    continue
                rows = run_per_dataset(a.mode, ds, model, seed, sd.device,
                                       pred_dir, only_central=a.only_central,
                                       full_detector=a.full_detector)
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
