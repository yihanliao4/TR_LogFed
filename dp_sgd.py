"""Sample-level DP-SGD federated training (Abadi et al. 2016), the DP-SGD
comparison requested in the revision. Unlike the post-training Gaussian
mechanism (mia_dp.py), this injects noise *during* training with genuine
per-example gradient clipping.

At each client local SGD step, for a logical batch of B examples:
  g_i = grad(loss_i; W)
  g_i <- C * g_i / max(C, ||g_i||)              (per-example L2 clipping)
  g   = (1/B) * sum_i g_i + (sigma*C/B) * N(0,1) (noise multiplier sigma)
  W   <- W - lr * g
Client updates are then combined with ordinary FedAvg. sigma=0 is the
clipping-only baseline (isolates the clipping effect). We report the same
fused LAD F1 and loss-based MIA as mia_dp.py, so the DP-SGD row is directly
comparable to the Gaussian-mechanism rows of Table VI.

The models are small (0.58-1.3M params) and have no BatchNorm, so per-example
gradients are computed by independent single-example forward/backward passes,
which is exact and avoids functorch/Transformer compatibility issues.

Smoke test first (cheap, verifies the models still train and prints the
clipping fraction and per-round timing):
  python dp_sgd.py --ds BGL --quick --device cuda:0
Full:
  python dp_sgd.py --ds BGL --device cuda:0
"""
import os, csv, time, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import standard_detection as sd
from mia_matrix import attack
import mia_matrix as mm
from models_next import build_model
from fed_core import fed_avg, select_clients, iid_partition

DATA = sd.DATA
MODEL = "neurallog"
N_FIX = 80000
FED_ROUNDS, FED_LE, C = 60, 2, 0.5
CLIP = 1.0
BATCH = 256
SIGMAS = [0.0, 0.1, 0.5, 1.0]


def dpsgd_local(model, Xc, yc, opt, sigma, clip, B, nrng, dev):
    """One local epoch of sample-level DP-SGD; returns (clipped, total)."""
    model.train()
    perm = np.random.permutation(len(yc))
    n_clip = n_tot = 0
    for st in range(0, len(perm), B):
        bidx = perm[st:st + B]
        model.zero_grad()
        gsum = [torch.zeros_like(p) for p in model.parameters()]
        for j in bidx:
            model.zero_grad()
            xb = torch.from_numpy(Xc[j:j + 1]).long().to(dev)
            yb = torch.from_numpy(np.array([yc[j]])).long().to(dev)
            F.cross_entropy(model(xb), yb).backward()
            sq = 0.0
            grads = []
            for p in model.parameters():
                g = p.grad.detach()
                grads.append(g)
                sq += float(g.pow(2).sum())
            norm = sq ** 0.5
            fac = min(1.0, clip / (norm + 1e-8))
            if norm > clip:
                n_clip += 1
            n_tot += 1
            with torch.no_grad():
                for gs, g in zip(gsum, grads):
                    gs.add_(fac * g)
        bb = len(bidx)
        with torch.no_grad():
            for p, gs in zip(model.parameters(), gsum):
                g = gs / bb
                if sigma > 0:
                    std = sigma * clip / bb
                    z = torch.from_numpy(
                        nrng.normal(0, std, tuple(p.shape)).astype(np.float32)
                    ).to(dev)
                    g = g + z
                p.grad = g
        opt.step()
    return n_clip, n_tot


def run_fed_dpsgd(n, nc, mlen, X, y, K, rounds, le, Cpart, sigma, clip, B,
                  seed, dev, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    nrng = np.random.RandomState(seed + 999)
    m = build_model(MODEL, n, nc, mlen).to(dev)
    chunks = iid_partition(len(y), K, rng)
    for rnd in range(rounds):
        t0 = time.time()
        active = select_clients(K, Cpart, rng)
        old = m.state_dict()
        states, sizes = [], []
        clip_c = tot_c = 0
        for ci in active:
            Xc, yc = X[chunks[ci]], y[chunks[ci]]
            local = build_model(MODEL, n, nc, mlen).to(dev)
            local.load_state_dict(old)
            opt = torch.optim.Adam(local.parameters(), 1e-3)
            for _ in range(le):
                a, b = dpsgd_local(local, Xc, yc, opt, sigma, clip, B, nrng, dev)
                clip_c += a; tot_c += b
            states.append(local.state_dict()); sizes.append(len(chunks[ci]))
        m.load_state_dict(fed_avg(states, sizes))
        if verbose and (rnd + 1) % 5 == 0:
            print(f"  [dpsgd sigma={sigma}] round {rnd+1}/{rounds} "
                  f"clip_frac={clip_c/max(1,tot_c):.3f} "
                  f"{time.time()-t0:.1f}s/round")
    return m


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigmas", default=None,
                    help="comma list, e.g. 0,0.1,0.5,1.0")
    ap.add_argument("--clip", type=float, default=CLIP)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    sd.device = torch.device(a.device)
    sigmas = SIGMAS if a.sigmas is None else [float(x) for x in a.sigmas.split(",")]

    out_csv = f"{DATA}/mia_dpsgd_table_{a.ds}.csv"
    pred_dir = f"{DATA}/mia_dpsgd_predlogs_{a.ds}"
    os.makedirs(f"{DATA}", exist_ok=True)

    D = sd.pickle.load(open(f"{DATA}/{a.ds}.pkl", "rb"))
    n, nc, mlen = D["num_event_ids"], D.get("num_chars", 130), 40
    group, glabel, X, ynext = D["group"], D["group_label"], D["X"], D["ynext"]
    tr_idx, te_idx, te_groups, train_g, n_norm = sd.split_data(
        D, a.seed, False, a.ds)
    if a.quick:
        r0 = np.random.RandomState(a.seed)
        tr_idx = tr_idx[r0.permutation(len(tr_idx))[:4000]]
        rounds, le, K = 2, 1, 20
    else:
        if len(tr_idx) > N_FIX:
            r0 = np.random.RandomState(a.seed)
            tr_idx = tr_idx[r0.permutation(len(tr_idx))[:N_FIX]]
        rounds, le, K = FED_ROUNDS, FED_LE, 100
    Xtr, ytr = X[tr_idx], ynext[tr_idx]

    # member / non-member pools (same as mia_dp)
    keys = [tuple(x) for x in X[tr_idx]]
    tcount = Counter(keys)
    fv = np.array([tcount[tuple(x)] for x in X[tr_idx]])
    member_pool = tr_idx[(fv >= 1) & (fv <= 2)]
    normal_test_g = te_groups[:n_norm]
    norm_te_idx = te_idx[np.isin(group[te_idx], normal_test_g)]
    train_keys = set(tcount.keys())
    novel = np.array([tuple(x) not in train_keys for x in X[norm_te_idx]])
    non_pool = norm_te_idx[novel]
    rng = np.random.RandomState(a.seed)
    m_n = min(len(member_pool), len(non_pool))
    member_idx = member_pool[rng.permutation(len(member_pool))[:m_n]]
    non_idx = non_pool[rng.permutation(len(non_pool))[:m_n]]

    done = load_done(out_csv)
    new_file = not os.path.exists(out_csv)
    fout = open(out_csv, "a", newline="")
    w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "model", "seed", "setting", "LAD_F1",
                    "MIA_AUC", "MIA_P", "MIA_R", "MIA_FPR", "MIA_F1"])

    for sig in sigmas:
        tag = f"fed_dpsgd{int(round(sig*100)):02d}"
        if (a.ds, str(a.seed), tag) in done and not a.quick:
            print(f"skip {tag}"); continue
        sd.set_seed(a.seed + 1000)
        mdl = run_fed_dpsgd(n, nc, mlen, Xtr, ytr, K, rounds, le, C,
                            sig, a.clip, a.batch, a.seed, sd.device)
        if a.quick:
            continue
        dauc, dP, dR, dF1 = det_fused(mdl, D, tr_idx, te_idx, te_groups,
                                      train_g, n_norm)
        mm.PRED = pred_dir
        r = attack(mdl, X, member_idx, non_idx, ynext,
                   f"{a.ds}_neurallog_s{a.seed}_{tag}")
        print(f"[{a.ds} s{a.seed} {tag}] LAD_F1={dF1:.4f} "
              f"MIA_AUC={r['auc']:.4f} MIA_F1={r['F1']:.4f} FPR={r['FPR']:.4f}")
        w.writerow([a.ds, MODEL, a.seed, tag, f"{dF1:.4f}",
                    f"{r['auc']:.4f}", f"{r['P']:.4f}", f"{r['R']:.4f}",
                    f"{r['FPR']:.4f}", f"{r['F1']:.4f}"])
        fout.flush()
    fout.close()
    print(f"\n[done] -> {out_csv}")


if __name__ == "__main__":
    main()
