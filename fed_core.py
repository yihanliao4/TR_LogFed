"""Core federated training utilities, kept separate from standard_detection.
Supports:
  - FedAvg / coordinate-wise Trimmed-Mean aggregation,
  - central (trusted-server) DP-SGD: clients train cleanly and may have their
    update clipped; the server aggregates once and adds a single Gaussian to
    the aggregated update. The noise is calibrated relative to the per-round
    update magnitude (sigma=1 -> noise RMS equals the aggregated update RMS),
    so it is reproducible across datasets and does not depend on parameter
    count or per-client data size.
  - partial client participation,
  - per-dataset IID partitions (cross-dataset non-IID pools are built in
    robust_experiment.py, which imports fed_avg/trimmed_mean directly).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models_next import build_model


def fed_avg(states, sizes):
    tot = sum(sizes)
    out = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, n in zip(states, sizes):
            acc += s[k].float() * (n / tot)
        out[k] = acc.to(states[0][k].dtype)
    return out


def trimmed_mean(states, sizes, beta):
    """Coordinate-wise Trimmed Mean: sort per-client values, drop the most
    extreme beta fraction at each end, average the rest (equal weight)."""
    n = len(states)
    k_trim = max(1, int(np.floor(beta * n)))
    out = {}
    for key in states[0]:
        stack = torch.stack([s[key].float() for s in states], dim=0)
        stack, _ = torch.sort(stack, dim=0)
        if 2 * k_trim < n:
            kept = stack[k_trim:n - k_trim]
        else:
            kept = stack[n // 2:n // 2 + 1]
        out[key] = kept.mean(0).to(states[0][key].dtype)
    return out


def clip_client_update(old_state, new_state, C):
    """Clip the flat client update (new-old) to global L2 norm C."""
    deltas = [new_state[k].float() - old_state[k].float() for k in old_state]
    norm = torch.norm(torch.stack([torch.norm(d) for d in deltas]), 2)
    factor = min(1.0, C / (norm.item() + 1e-8))
    out = {}
    for k in old_state:
        out[k] = (old_state[k].float()
                  + factor * (new_state[k].float() - old_state[k].float())
                  ).to(old_state[k].dtype)
    return out


def add_central_noise(old_state, agg_state, sigma, device, rng):
    """Add Gaussian noise to the aggregated update, std = sigma * (RMS of the
    aggregated update across all parameters)."""
    sq, npar = 0.0, 0
    for key in old_state:
        d = agg_state[key].float() - old_state[key].float()
        sq += float((d ** 2).sum()); npar += d.numel()
    update_rms = (sq / max(1, npar)) ** 0.5
    std = sigma * update_rms
    out = {}
    for key in old_state:
        d = agg_state[key].float() - old_state[key].float()
        noise = torch.from_numpy(
            rng.normal(0, 1, d.shape).astype(np.float32)).to(device)
        out[key] = (old_state[key].float() + d + std * noise
                    ).to(old_state[key].dtype)
    return out


def select_clients(K, C, rng):
    active = rng.permutation(K)
    return sorted(active[:max(1, int(round(C * K)))])


def iid_partition(N, K, rng):
    return np.array_split(rng.permutation(N), K)


def run_fed(model_name, n, nc, mlen, X, y, K, rounds, le, C,
            dp_sigma=0.0, dp_clip=1.0, robust="fedavg", trim_beta=0.1,
            seed=0, device="cuda", lr=1e-3, batch=512, verbose=True):
    """Federated training on one dataset (IID partition)."""
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    nrng = np.random.RandomState(seed + 12345)
    m = build_model(model_name, n, nc, mlen).to(device)
    chunks = iid_partition(len(y), K, rng)
    for r in range(rounds):
        active = select_clients(K, C, rng)
        old_state = m.state_dict()
        states, sizes = [], []
        for ci in active:
            Xc, yc = X[chunks[ci]], y[chunks[ci]]
            local = build_model(model_name, n, nc, mlen).to(device)
            local.load_state_dict(old_state)
            opt = torch.optim.Adam(local.parameters(), lr)
            dl = DataLoader(TensorDataset(torch.from_numpy(Xc).long(),
                                          torch.from_numpy(yc).long()),
                            batch, shuffle=True)
            for _ in range(le):
                local.train()
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    F.cross_entropy(local(xb), yb).backward()
                    opt.step()
            states.append(local.state_dict()); sizes.append(len(chunks[ci]))
        if dp_sigma > 0:
            states = [clip_client_update(old_state, s, dp_clip) for s in states]
        if robust == "trimmed":
            agg = trimmed_mean(states, sizes, trim_beta)
        else:
            agg = fed_avg(states, sizes)
        if dp_sigma > 0:
            agg = add_central_noise(old_state, agg, dp_sigma, device, nrng)
        m.load_state_dict(agg)
    return m


def add_model_noise(model, sigma, seed=0):
    """Gaussian mechanism applied once to the trained parameters:
        W <- W + sigma * RMS(W) * N(0,1).
    Noise is calibrated relative to the model's own weight scale, so it is
    reproducible across models and does not depend on the number of rounds.
    Returns the same model (perturbed in place)."""
    if sigma <= 0:
        return model
    rng = np.random.RandomState(seed)
    sq, npar = 0.0, 0
    with torch.no_grad():
        for p in model.parameters():
            sq += float((p ** 2).sum()); npar += p.numel()
        w_rms = (sq / max(1, npar)) ** 0.5
        for p in model.parameters():
            z = torch.from_numpy(rng.normal(0, 1, tuple(p.shape)).astype(
                np.float32)).to(p.device)
            p.add_(sigma * w_rms * z)
    return model
