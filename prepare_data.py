import os
import re
import glob
import pickle
import argparse
import numpy as np
import torch


def numeric_client_id(path: str) -> int:
    m = re.search(r"client_(\d+)_data\.pkl", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _to_numpy(x):
    """Convert torch.Tensor / numpy / list to numpy (without forcing stack)."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _pad_or_truncate_2d(a2d: np.ndarray, L: int, D: int) -> np.ndarray:
    """
    a2d: (l, d)
    return (L, D)
    """
    if a2d.ndim != 2:
        raise ValueError(f"Expected 2D array per sample, got shape={a2d.shape}")
    l, d = a2d.shape
    if d != D:
        raise ValueError(f"Embedding dim mismatch: got {d}, expected {D}")
    if l == L:
        return a2d.astype(np.float32, copy=False)
    if l > L:
        return a2d[:L].astype(np.float32, copy=False)
    # l < L: pad zeros
    out = np.zeros((L, D), dtype=np.float32)
    out[:l] = a2d.astype(np.float32, copy=False)
    return out


def _normalize_x_to_NLD(x, L=10, D=768) -> np.ndarray:
    """
    Acceptable inputs:
      - np.ndarray (N, L, D)
      - torch.Tensor (N, L, D)
      - list of (L,D) arrays/tensors
      - list of (variable_l, D) arrays/tensors  -> pad/truncate to L
    Return: np.ndarray float32 (N, L, D)
    """
    x = _to_numpy(x)

    # Case 1: already ndarray with 3 dims
    if isinstance(x, np.ndarray) and x.ndim == 3:
        if x.shape[1] != L or x.shape[2] != D:
            # allow pad/truncate if only seq len differs
            if x.shape[2] != D:
                raise ValueError(f"X has D={x.shape[2]} expected {D}")
            # pad/truncate each sample along L
            xs = [_pad_or_truncate_2d(x[i], L, D) for i in range(x.shape[0])]
            return np.stack(xs, axis=0).astype(np.float32, copy=False)
        return x.astype(np.float32, copy=False)

    # Case 2: list/tuple of samples
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.zeros((0, L, D), dtype=np.float32)

        xs = []
        for i, item in enumerate(x):
            item = _to_numpy(item)
            item = np.asarray(item)
            # item expected (l, d) OR (L, D)
            if item.ndim != 2:
                raise ValueError(f"Sample {i} has ndim={item.ndim}, shape={item.shape}, expected 2D (l,d)")
            xs.append(_pad_or_truncate_2d(item, L, D))
        return np.stack(xs, axis=0)

    # Case 3: ndarray but not 3D
    if isinstance(x, np.ndarray):
        raise ValueError(f"Unsupported ndarray shape for x: {x.shape} (ndim={x.ndim})")

    raise ValueError(f"Unsupported type for x: {type(x)}")


def load_client_pkls(pkl_dir: str, L=10, D=768):
    paths = sorted(glob.glob(os.path.join(pkl_dir, "client_*_data.pkl")), key=numeric_client_id)
    if len(paths) == 0:
        raise FileNotFoundError(f"No client_*_data.pkl found in: {pkl_dir}")

    X_all, y_all = [], []
    for p in paths:
        with open(p, "rb") as f:
            x, y = pickle.load(f)

        X = _normalize_x_to_NLD(x, L=L, D=D)

        y = _to_numpy(y)
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1:
            y = y.reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"[{p}] size mismatch: X={X.shape} y={y.shape}")

        X_all.append(X)
        y_all.append(y)

        # optional debug for first file of each dataset
        # print(f"[DEBUG] {os.path.basename(p)} X={X.shape} y={y.shape} y_unique={np.unique(y)}")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y, paths


def train_test_split(X, y, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0], dtype=np.int64)
    rng.shuffle(idx)

    n_test = int(round(len(idx) * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def make_iid_partitions(n_train: int, K: int, seed: int):
    """
    IID virtual clients: shuffle indices, split into K chunks.
    Returns dtype=object array length K, each is np.int64 array.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n_train, dtype=np.int64)
    rng.shuffle(idx)
    parts = np.array_split(idx, K)
    return np.array([p.astype(np.int64) for p in parts], dtype=object)


def assign_clients_to_datasets(K: int):
    """
    Uniform assignment: roughly K/3 each.
    Returns ds_ids length K, values in {0,1,2} meaning {BGL,HDFS,Thunderbird}.
    """
    ds_ids = np.empty(K, dtype=np.int64)
    # e.g., K=100 -> 34/33/33
    counts = [K // 3 + (1 if i < (K % 3) else 0) for i in range(3)]
    cursor = 0
    for ds, c in enumerate(counts):
        ds_ids[cursor:cursor + c] = ds
        cursor += c
    return ds_ids, counts


def make_mixed_parts(train_sizes, K: int, samples_per_client: int, seed: int):
    """
    Build mixed non-IID client partitions:
      - each client belongs to one dataset pool (0/1/2)
      - each client gets S indices from that pool's TRAIN set
    Try without replacement within each dataset pool.
    If not enough samples, fallback to with replacement for that pool.

    Returns:
      ds_ids: (K,) int64
      idxs_list: object array length K, each is np.int64 indices into the selected pool's train set
      counts: number of clients per dataset [c0,c1,c2]
      flags: dict pool-> {"replacement": bool, "needed": int, "available": int}
    """
    rng = np.random.default_rng(seed)
    ds_ids, counts = assign_clients_to_datasets(K)

    idxs_list = np.empty(K, dtype=object)
    flags = {}

    # Prepare shuffled index streams per pool
    streams = {}
    for ds in range(3):
        n = int(train_sizes[ds])
        idx = np.arange(n, dtype=np.int64)
        rng.shuffle(idx)
        streams[ds] = idx

    # allocate
    # For each dataset, gather clients, then assign contiguous chunks of size S
    for ds in range(3):
        client_ids = np.where(ds_ids == ds)[0]
        need = len(client_ids) * samples_per_client
        avail = int(train_sizes[ds])

        if need <= avail:
            flags[ds] = {"replacement": False, "needed": need, "available": avail}
            take = streams[ds][:need]
            splits = np.array_split(take, len(client_ids))
            for cid, part in zip(client_ids, splits):
                idxs_list[cid] = part.astype(np.int64)
        else:
            # fallback: with replacement
            flags[ds] = {"replacement": True, "needed": need, "available": avail}
            for cid in client_ids:
                idxs_list[cid] = rng.integers(0, avail, size=samples_per_client, dtype=np.int64)

    return ds_ids, idxs_list, counts, flags


def save_dataset_cache(ds_name: str, out_dir: str, X_train, y_train, X_test, y_test, K: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "data.npz"),
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    parts = make_iid_partitions(len(y_train), K=K, seed=seed)
    np.savez_compressed(os.path.join(out_dir, f"parts_K{K}_seed{seed}.npz"), parts=parts)

    print(f"[{ds_name}] saved -> {os.path.abspath(out_dir)} | "
          f"train={X_train.shape} test={X_test.shape} y_unique={np.unique(y_train)}")


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", default="../LogMIA/FedLog/iid_datasets/client_data")
    ap.add_argument("--out_root", default="./datasets/partitions", help="output cache root")
    ap.add_argument("--K", type=int, default=100, help="number of virtual clients for IID caches")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--test_ratio", type=float, default=0.2)

    # mixed non-IID
    ap.add_argument("--make_mixed", default=True, action="store_true")
    ap.add_argument("--mixed_name", default="MIXED_NonIID", help="subdir name under out_root for mixed cache")
    ap.add_argument("--mixed_K", type=int, default=100, help="number of clients in mixed non-IID")
    ap.add_argument("--samples_per_client", type=int, default=10000,
                    help="S samples per client in mixed non-IID (from its assigned dataset pool train split)")

    args = ap.parse_args()

    datasets = ["BGL", "HDFS", "Thunderbird"]
    ds_data = {}  # ds -> dict(train/test arrays)

    # 1) build per-dataset IID cache
    for ds in datasets:
        ds_dir = os.path.join(args.root, ds)
        X, y, paths = load_client_pkls(ds_dir)
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, test_ratio=args.test_ratio, seed=args.seed)

        out_dir = os.path.join(args.out_root, ds)
        save_dataset_cache(ds, out_dir, X_train, y_train, X_test, y_test, K=args.K, seed=args.seed)

        ds_data[ds] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "n_train": len(y_train)
        }
        print(f"[{ds}] pkls={len(paths)} global={X.shape} train_n={len(y_train)} test_n={len(y_test)}")

    # 2) build mixed non-IID cache (optional)
    if args.make_mixed:
        mixed_dir = os.path.join(args.out_root, args.mixed_name)
        os.makedirs(mixed_dir, exist_ok=True)

        # union test set (fair evaluation)
        X_test_union = np.concatenate([ds_data[ds]["X_test"] for ds in datasets], axis=0)
        y_test_union = np.concatenate([ds_data[ds]["y_test"] for ds in datasets], axis=0)
        np.savez_compressed(os.path.join(mixed_dir, "mixed_test.npz"),
                            X_test=X_test_union.astype(np.float32),
                            y_test=y_test_union.astype(np.int64))

        train_sizes = [ds_data["BGL"]["n_train"], ds_data["HDFS"]["n_train"], ds_data["Thunderbird"]["n_train"]]
        ds_ids, idxs_list, counts, flags = make_mixed_parts(
            train_sizes=train_sizes,
            K=args.mixed_K,
            samples_per_client=args.samples_per_client,
            seed=args.seed
        )

        np.savez_compressed(os.path.join(mixed_dir, f"mixed_parts_K{args.mixed_K}_S{args.samples_per_client}_seed{args.seed}.npz"),
                            ds_ids=ds_ids.astype(np.int64),
                            idxs_list=idxs_list)

        # write a small meta txt for reproducibility
        meta_path = os.path.join(mixed_dir, "meta.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"mixed_K={args.mixed_K}\n")
            f.write(f"samples_per_client={args.samples_per_client}\n")
            f.write(f"seed={args.seed}\n")
            f.write(f"datasets={datasets}\n")
            f.write(f"clients_per_dataset={counts}\n")
            f.write(f"train_sizes={train_sizes}\n")
            f.write("replacement_flags:\n")
            for ds_i, info in flags.items():
                name = datasets[ds_i]
                f.write(f"  {name}: replacement={info['replacement']} needed={info['needed']} available={info['available']}\n")

        print(f"[MIXED] saved -> {os.path.abspath(mixed_dir)}")
        print(f"[MIXED] test_union: X={X_test_union.shape} y={y_test_union.shape}")
        print(f"[MIXED] clients_per_dataset={counts} | train_sizes={train_sizes}")
        for ds_i, info in flags.items():
            name = datasets[ds_i]
            if info["replacement"]:
                print(f"[WARN][MIXED] {name}: NOT enough train samples for S; used sampling WITH replacement "
                      f"(needed={info['needed']} avail={info['available']})")

if __name__ == "__main__":
    main()
