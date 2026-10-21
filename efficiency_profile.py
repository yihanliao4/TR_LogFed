"""Efficiency & communication profiling for Table VII / Table VIII (K=100).

Replaces the old K=10 timing. Measures, on the actual unified data and the
K=100 protocol:
  * parameter count and FP32 model size per (dataset, model) -- the event
    embedding table scales with the event vocabulary (BGL 350 / HDFS 29 /
    Thunderbird 3336 / merged cross 3715), so size is reported per dataset;
  * wall-clock seconds per CENTRAL epoch (the "CT" baseline) and per federated
    round under IID (C=0.5, le=1) and cross-dataset non-IID (C=1.0, le=3);
  * throughput (samples/s);
  * Table VIII communication: per-round round-trip (upload + download) latency
    for a single client at 1 / 5 / 20 Mbps (RTT = 2 * model_size / bandwidth).

Only a handful of epochs/rounds are timed (with a warm-up), since every unit
processes a fixed number of samples. CUDA is synchronized around the timed
region. Outputs:
  revision_id/unified/eff_params.csv
  revision_id/unified/eff_timing.csv
Usage:
  python efficiency_profile.py --device cuda:0
  python efficiency_profile.py --device cuda:0 --model neurallog
"""
import os, csv, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import standard_detection as sd
from models_next import build_model
from fed_core import iid_partition, select_clients, fed_avg
import matrix_k100 as mk

OUT_DIR = f"{sd.DATA}/efficiency"
DATASETS = ["BGL", "HDFS", "Thunderbird"]
K = 100
BATCH = 512
LE = 1
IID_C = 0.5
CROSS_C = 1.0


def model_params(model_name, n, nc=130, mlen=40):
    m = build_model(model_name, n, nc, mlen)
    return sum(p.numel() for p in m.parameters())


def time_central(model_name, X, y, n, nc, mlen, device,
                 n_warm=1, n_time=3):
    m = build_model(model_name, n, nc, mlen).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(X).long(),
                                  torch.from_numpy(y).long()),
                    BATCH, shuffle=True)

    def one_epoch():
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
    for _ in range(n_warm):
        one_epoch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        one_epoch()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_time, len(y)


def time_fed(model_name, chunks, n, nc, mlen, device, C,
             n_warm=1, n_time=4, le=LE):
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    rng = np.random.RandomState(0)
    m = build_model(model_name, n, nc, mlen).to(device)

    def one_round():
        active = select_clients(K, C, rng)
        old = m.state_dict()
        states, sizes = [], []
        for ci in active:
            Xc, yc = chunks[ci]
            local = build_model(model_name, n, nc, mlen).to(device)
            local.load_state_dict(old)
            opt = torch.optim.Adam(local.parameters(), 1e-3)
            dl = DataLoader(TensorDataset(torch.from_numpy(Xc).long(),
                                          torch.from_numpy(yc).long()),
                            BATCH, shuffle=True)
            for _ in range(le):
                local.train()
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); F.cross_entropy(local(xb), yb).backward(); opt.step()
            states.append(local.state_dict()); sizes.append(len(yc))
        m.load_state_dict(fed_avg(states, sizes))
        return sum(sizes)
    for _ in range(n_warm):
        nproc = one_round()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tot = 0
    for _ in range(n_time):
        tot += one_round()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_time, tot / n_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    models = [a.model] if a.model else mk.MODELS

    # ---- parameters / model size / communication (Table VIII) ----
    pv_rows = []
    vocab = {"BGL": None, "HDFS": None, "Thunderbird": None}
    for ds in DATASETS:
        D, n, nc, mlen = mk.load_ds(ds)
        vocab[ds] = (n, nc, mlen)
    cross_n = sum(vocab[ds][0] for ds in DATASETS)
    for model in models:
        cases = [(ds, vocab[ds][0]) for ds in DATASETS] + [("non-IID", cross_n)]
        for label, n in cases:
            P = model_params(model, n, vocab["BGL"][1], vocab["BGL"][2])
            bytes_fp32 = P * 4
            size_mb = bytes_fp32 / (1024 ** 2)
            # per-client round-trip (upload + download) = 2 * size
            rt = {bw: (2 * bytes_fp32 * 8) / (bw * 1e6) for bw in (1, 5, 20)}
            pv_rows.append((model, label, n, P, f"{bytes_fp32/1024:.1f}",
                            f"{size_mb:.4f}", f"{rt[1]:.3f}", f"{rt[5]:.3f}",
                            f"{rt[20]:.3f}"))
    with open(f"{OUT_DIR}/eff_params.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "vocab_n", "params", "size_kb",
                    "size_mb_fp32", "rtt_1Mbps_s", "rtt_5Mbps_s",
                    "rtt_20Mbps_s"])
        w.writerows(pv_rows)
    print("=== params / communication (Table VIII) ===")
    for r in pv_rows:
        print(f"  {r[0]:11s} {r[1]:10s} n={r[2]:5d} params={r[3]:>9d} "
              f"size={r[5]}MB  RTT@1/5/20={r[6]}/{r[7]}/{r[8]}s")

    # ---- wall-clock timing (Table VII) ----
    timing = []
    for ds in DATASETS:
        D, n, nc, mlen = mk.load_ds(ds)
        tr_idx, _, _, _, _ = sd.split_data(D, 0, False, ds)
        # keep models lightweight and fair for timing; char models use Xchar
        for model in models:
            char = mk.use_char(model)
            Xtr = D["Xchar"][tr_idx] if char else D["X"][tr_idx]
            ytr = D["ynext"][tr_idx]
            ct, nproc = time_central(model, Xtr, ytr, n, nc, mlen, device)
            timing.append((ds, model, "central_epoch", f"{ct:.3f}",
                           nproc, f"{nproc/ct:.1f}"))
            print(f"  {ds:10s} {model:11s} CT epoch {ct:.3f}s")

            rng = np.random.RandomState(0)
            chunks = iid_partition(len(ytr), K, rng)
            chunks = [(Xtr[c], ytr[c]) for c in chunks]
            ft, fp = time_fed(model, chunks, n, nc, mlen, device, IID_C)
            timing.append((ds, model, "fed_iid_round", f"{ft:.3f}",
                           int(fp), f"{fp/ft:.1f}"))
            print(f"  {ds:10s} {model:11s} Fed IID round {ft:.3f}s")

    # cross-dataset non-IID (one merged model, n=3715)
    for model in models:
        total_n, Ds, offsets, splits, chunks, _ = mk.build_cross_data(
            "detection", 0, model)
        nc, mlen = 130, 40
        ft, fp = time_fed(model, chunks, total_n, nc, mlen, device, CROSS_C,
                          le=3)
        timing.append(("non-IID", model, "fed_cross_round", f"{ft:.3f}",
                       int(fp), f"{fp/ft:.1f}"))
        print(f"  non-IID    {model:11s} Fed cross round {ft:.3f}s")

    with open(f"{OUT_DIR}/eff_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "unit", "seconds_per_unit",
                    "samples_processed", "samples_per_sec"])
        w.writerows(timing)
    print(f"\n[done] -> {OUT_DIR}/eff_params.csv, eff_timing.csv")


if __name__ == "__main__":
    main()
