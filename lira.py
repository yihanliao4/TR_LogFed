"""LiRA (likelihood-ratio attack, Carlini et al., S&P 2022) against the same
K=100 LogFed victims reported in Table V.

We implement the standard online/population LiRA with a fixed-variance
Gaussian (the tractable, widely-used variant):

  * A fixed target set of M rare-n-gram members (training frequency 1-2) and
    M genuinely-novel non-members is drawn once (the same pools as Table V).
  * S shadow models are trained independently. Each shadow trains on a Bernoulli
    p_in subsample of the training pool, so every target is present in roughly
    S*p_in shadows and absent in the rest; shadow init / sampling differ too.
  * Per-example NLL of every target is recorded under every shadow.
  * For each target we form in/out loss distributions; a single global
    variance is used (fixed-variance Gaussian), per-target means are shrunk
    toward the global mean when too few observations exist.
  * The victim (trained once on the full pool) is scored with the Gaussian
    log-likelihood ratio  log p_in(l) - log p_out(l); we report the AUC
    (and an F1-optimal threshold F1) separating members from non-members.

Scope (strong-attack sanity check, kept tractable): BGL + NeuralLog, the
strongest-memorization setting.

Smoke test (tiny, does NOT write the CSV):
  python lira.py --paradigm central --quick --device cuda:0
Full (one paradigm per GPU):
  python lira.py --paradigm central    --device cuda:0 --shadows 32
  python lira.py --paradigm fed_iid    --device cuda:1 --shadows 32
  python lira.py --paradigm fed_noniid --device cuda:2 --shadows 16

Outputs: revision_id/unified/lira/lira.csv (resumable per paradigm).
"""
import os, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import standard_detection as sd
import matrix_k100 as mk
from models_next import build_model
from fed_core import run_fed, fed_avg, select_clients

DS = "BGL"
MODEL = "neurallog"
SEED = 0
P_IN = 0.5
N_TARGETS = 300
OUT_DIR = f"{sd.DATA}/lira"
OUT_CSV = f"{OUT_DIR}/lira.csv"

# victim protocols (identical to Table V / matrix_k100 mia cfg)
CENTRAL_EP = 300
IID_ROUNDS, IID_LE, IID_C = 60, 2, 0.5
CROSS_ROUNDS, CROSS_LE, CROSS_C = 60, 3, 1.0


# ---------------------------------------------------------------------------
# data / target pools
# ---------------------------------------------------------------------------

def base_pool(n_targets):
    """Fixed 80k victim training pool and the balanced member/non-member target
    pools (exactly the mia_pools construction)."""
    D, n, nc, mlen = mk.load_ds(DS)
    char = mk.use_char(MODEL)
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(D, SEED, False, DS)
    r0 = np.random.RandomState(SEED)
    tr_idx = tr_idx[r0.permutation(len(tr_idx))[:mk.N_FIX]]
    split = (te_idx, te_groups, train_g, n_norm)
    rng = np.random.RandomState(SEED)
    mi, ni = mk.mia_pools(D["X"], D["ynext"], tr_idx, D, split, rng)
    mi = mi[:n_targets]
    ni = ni[:n_targets]
    Xcol = D["Xchar"] if char else D["X"]
    return D, n, nc, mlen, char, tr_idx, Xcol, mi, ni


def target_loss(m, Xcol, ynext, mi, ni):
    lm = sd.per_sample_loss(m, Xcol[mi], ynext[mi])
    ln = sd.per_sample_loss(m, Xcol[ni], ynext[ni])
    return np.r_[lm, ln]  # (2M,)


# ---------------------------------------------------------------------------
# per-dataset (central / fed_iid) models
# ---------------------------------------------------------------------------

def train_central(Xtr, ytr, n, nc, mlen, seed, ep):
    sd.set_seed(seed)
    sd.CENTRAL_EP = ep
    return sd.train_central(MODEL, n, nc, mlen, Xtr, ytr)


def train_iid(Xtr, ytr, n, nc, mlen, seed, rounds, le):
    sd.set_seed(seed)
    return run_fed(MODEL, n, nc, mlen, Xtr, ytr, mk.K,
                   rounds, le, IID_C, seed=seed, device=sd.device,
                   verbose=False)


# ---------------------------------------------------------------------------
# cross-dataset (fed_noniid) models
# ---------------------------------------------------------------------------

def cross_cfg(rounds, le):
    cfg = dict(mk.CFG["mia"])
    cfg["_mode"] = "mia"
    cfg["cross_rounds"] = rounds
    cfg["cross_le"] = le
    return cfg


def cross_victim(rounds, le):
    """Full cross victim, identical to matrix_k100 (cap 80000)."""
    m, Ds, offsets, splits, char = mk.train_cross(
        MODEL, SEED, sd.device, cross_cfg(rounds, le))
    return m, Ds, offsets, char


def cross_shadow(shadow_seed, rounds, le):
    """Train one cross shadow on a Bernoulli p_in subsample of each dataset's
    pool. Returns (model, offsets, bgl_present_global_idx)."""
    char = mk.use_char(MODEL)
    total_n, Ds, offsets, splits, _full_chunks, _rng = mk.build_cross_data(
        "mia", SEED, MODEL)
    rng = np.random.RandomState(shadow_seed)
    chunks = []
    bgl_present = None
    for ds in mk.DATASETS:
        D = Ds[ds]
        o = offsets[ds]
        tr_idx = sd.split_data(D, SEED, False, ds)[0]
        r0 = np.random.RandomState(SEED)
        tr_idx = tr_idx[r0.permutation(len(tr_idx))[:mk.N_FIX]]
        keep = rng.random(len(tr_idx)) < P_IN
        tr_sub = tr_idx[keep]
        if ds == DS:
            bgl_present = tr_sub
        n_pool = mk.POOL_COUNTS[ds]
        parts = np.array_split(rng.permutation(len(tr_sub)), n_pool)
        for p in parts:
            idx = tr_sub[p]
            Xc = D["Xchar"][idx] if char else D["X"][idx] + o
            yc = D["ynext"][idx] + o
            chunks.append((Xc, yc))
    torch.manual_seed(shadow_seed); np.random.seed(shadow_seed)
    torch.cuda.manual_seed_all(shadow_seed)
    m = build_model(MODEL, total_n, 130, 40).to(sd.device)
    crng = np.random.RandomState(shadow_seed)
    for rnd in range(rounds):
        active = select_clients(mk.K, CROSS_C, crng)
        old = m.state_dict()
        states, sizes = [], []
        for ci in active:
            Xc, yc = chunks[ci]
            local = build_model(MODEL, total_n, 130, 40).to(sd.device)
            local.load_state_dict(old)
            opt = torch.optim.Adam(local.parameters(), 1e-3)
            dl = DataLoader(TensorDataset(torch.from_numpy(Xc).long(),
                                          torch.from_numpy(yc).long()),
                            512, shuffle=True)
            for _ in range(le):
                local.train()
                for xb, yb in dl:
                    xb, yb = xb.to(sd.device), yb.to(sd.device)
                    opt.zero_grad()
                    F.cross_entropy(local(xb), yb).backward(); opt.step()
            states.append(local.state_dict()); sizes.append(len(yc))
        m.load_state_dict(fed_avg(states, sizes))
    return m, offsets, bgl_present


# ---------------------------------------------------------------------------
# LiRA likelihood ratio
# ---------------------------------------------------------------------------

def lira_auc(victim_loss, shadow_loss, inc, M):
    """victim_loss (2M,); shadow_loss (S,2M); inc (S,M) member inclusion mask."""
    mem = shadow_loss[:, :M]
    non = shadow_loss[:, M:]

    in_mean = np.full(M, np.nan)
    out_mean = np.zeros(2 * M)
    in_res, out_res = [], []

    for t in range(M):
        in_l = mem[inc[:, t] == 1, t]
        out_l = mem[inc[:, t] == 0, t]
        if len(in_l) > 0:
            in_mean[t] = in_l.mean()
        if len(out_l) == 0:
            out_l = in_l
        out_mean[t] = out_l.mean()
        if len(in_l) > 1:
            in_res += list(in_l - in_mean[t])
        if len(out_l) > 1:
            out_res += list(out_l - out_mean[t])

    for t in range(M):
        out_l = non[:, t]
        out_mean[M + t] = out_l.mean()
        if len(out_l) > 1:
            out_res += list(out_l - out_mean[M + t])

    glob_in = np.nanmean(in_mean)
    in_mean = np.where(np.isnan(in_mean), glob_in, in_mean)
    var_in = max(np.var(in_res), 1e-8)
    var_out = max(np.var(out_res), 1e-8)

    def loggau(x, mu, v):
        return -0.5 * np.log(2 * np.pi * v) - (x - mu) ** 2 / (2 * v)

    score = np.zeros(2 * M)
    for t in range(M):
        score[t] = loggau(victim_loss[t], in_mean[t], var_in) \
            - loggau(victim_loss[t], out_mean[t], var_out)
    for t in range(M):
        score[M + t] = loggau(victim_loss[M + t], glob_in, var_in) \
            - loggau(victim_loss[M + t], out_mean[M + t], var_out)

    y = np.r_[np.ones(M), np.zeros(M)]
    auc = roc_auc_score(y, score)
    best_f1 = -1
    for th in np.quantile(score, np.linspace(0, 1, 400)):
        pred = (score > th).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
        f1 = 2 * P * R / max(1e-9, P + R)
        if f1 > best_f1:
            best_f1 = f1
    return auc, best_f1


def done():
    if os.path.exists(OUT_CSV):
        return {r["paradigm"] for r in csv.DictReader(open(OUT_CSV))}
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", required=True,
                    choices=["central", "fed_iid", "fed_noniid"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shadows", type=int, default=32)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run; does not write the CSV")
    a = ap.parse_args()
    sd.device = torch.device(a.device)

    if a.quick:
        n_targets, cep, iid_r, iid_le, cross_r, cross_le = \
            100, 10, 3, 1, 2, 1
        n_sh = min(a.shadows, 2)
        print("[QUICK] cheap smoke run, no CSV written")
    else:
        n_targets, cep, iid_r, iid_le, cross_r, cross_le = \
            N_TARGETS, CENTRAL_EP, IID_ROUNDS, IID_LE, CROSS_ROUNDS, CROSS_LE
        n_sh = a.shadows

    D, n, nc, mlen, char, tr_idx, Xcol, mi, ni = base_pool(n_targets)
    M = len(mi)
    ynext = D["ynext"]
    print(f"[LiRA {DS} {MODEL}] paradigm={a.paradigm} M={M} S={n_sh}")

    # --- victim ---
    if a.paradigm == "central":
        victim = train_central(Xcol[tr_idx], ynext[tr_idx], n, nc, mlen,
                               SEED, cep)
        victim_loss = target_loss(victim, Xcol, ynext, mi, ni)
    elif a.paradigm == "fed_iid":
        victim = train_iid(Xcol[tr_idx], ynext[tr_idx], n, nc, mlen,
                           SEED, iid_r, iid_le)
        victim_loss = target_loss(victim, Xcol, ynext, mi, ni)
    else:
        victim, Ds, offsets, _ = cross_victim(cross_r, cross_le)
        o = offsets[DS]
        Xe = Ds[DS]["Xchar"] if char else Ds[DS]["X"] + o
        ye = Ds[DS]["ynext"] + o
        victim_loss = target_loss(victim, Xe, ye, mi, ni)

    # --- shadows ---
    shadow_loss = np.zeros((n_sh, 2 * M))
    inc = np.zeros((n_sh, M), dtype=np.int8)
    for s in range(n_sh):
        sseed = 10000 + s * 7
        if a.paradigm == "central":
            rng = np.random.RandomState(sseed)
            sub = tr_idx[rng.random(len(tr_idx)) < P_IN]
            present = set(sub.tolist())
            mdl = train_central(Xcol[sub], ynext[sub], n, nc, mlen, sseed, cep)
            loss = target_loss(mdl, Xcol, ynext, mi, ni)
        elif a.paradigm == "fed_iid":
            rng = np.random.RandomState(sseed)
            sub = tr_idx[rng.random(len(tr_idx)) < P_IN]
            present = set(sub.tolist())
            mdl = train_iid(Xcol[sub], ynext[sub], n, nc, mlen, sseed,
                            iid_r, iid_le)
            loss = target_loss(mdl, Xcol, ynext, mi, ni)
        else:
            mdl, offsets, bgl_sub = cross_shadow(sseed, cross_r, cross_le)
            present = set(bgl_sub.tolist())
            o = offsets[DS]
            Xe = Ds[DS]["Xchar"] if char else Ds[DS]["X"] + o
            ye = Ds[DS]["ynext"] + o
            loss = target_loss(mdl, Xe, ye, mi, ni)
        shadow_loss[s] = loss
        for t in range(M):
            inc[s, t] = int(int(mi[t]) in present)
        n_in = int((inc[:s + 1].sum(axis=0) > 0).sum())
        print(f"  shadow {s+1}/{n_sh} done; targets with >=1 in: {n_in}/{M}")

    auc, f1 = lira_auc(victim_loss, shadow_loss, inc, M)
    print(f"[LiRA {a.paradigm}] {DS} {MODEL} AUC={auc:.4f} F1={f1:.4f}")

    if a.quick:
        print("[QUICK] not writing CSV.")
        return
    if a.paradigm in done():
        print(f"[skip] {a.paradigm} already in {OUT_CSV}; delete row to rerun")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "model", "paradigm", "shadows",
                        "p_in", "M", "LiRA_AUC", "LiRA_F1"])
        w.writerow([DS, MODEL, a.paradigm, n_sh, P_IN, M,
                    f"{auc:.4f}", f"{f1:.4f}"])
    print(f"[done] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
