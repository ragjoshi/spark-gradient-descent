# numpy_baseline.py: the same batch gradient descent on one machine, no Spark.
#
# Usage:
#   python numpy_baseline.py [--data FILE] [--label COL] [--iters N] [--threads T]
#   python numpy_baseline.py --npy PREFIX [--stream] [--chunk-mb MB] [--iters N]
#
# --data loads a CSV into memory and standardizes it like train.standardize:
# without --label it is the HIGGS layout (no header, label in column 0), with
# --label COL the first line is a header and COL is the label (as in bench.py). --npy reads the output of
# make_big_data.py (PREFIX.X.npy, PREFIX.y.npy) instead: loaded whole into RAM by default, or with
# --stream read from disk in chunks on every iteration, which is what one
# machine has to do when the data is larger than its memory.
#
# Each iteration computes w -= lr * X.T @ (sigmoid(X @ w) - y) / n and is
# timed. This is the bar Spark has to beat.
#
# --threads caps the BLAS thread pool (default: the library's own choice,
# usually every core). Use --threads 1 for a strictly single-core number.
# The last line of stdout is one JSON object.
import argparse
import json
import os
import statistics
import time

p = argparse.ArgumentParser(description="Single-machine NumPy baseline.")
p.add_argument("--data", default="higgs_1m.csv")
p.add_argument("--label", help="label column name; the CSV then needs a header row")
p.add_argument("--npy", help="output prefix of make_big_data.py (used instead of --data)")
p.add_argument("--stream", action="store_true",
               help="with --npy: re-read the file from disk in chunks every iteration")
p.add_argument("--chunk-mb", type=int, default=256)
p.add_argument("--iters", type=int, default=50)
p.add_argument("--threads", type=int)
args = p.parse_args()

# BLAS reads these when NumPy is first imported, so set them before that.
if args.threads is not None:
    for var in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

import numpy as np

LR = 0.5


def sigmoid(z):
    # Same formula as gradient.sigmoid, copied so this baseline runs without
    # PySpark installed (gradient.py imports it).
    return 1 / (1 + np.exp(-z))


def load_csv(path, label=None):
    import pandas as pd
    if label is None:
        raw = pd.read_csv(path, header=None, dtype=np.float64).to_numpy()
        y = raw[:, 0]
        feats = raw[:, 1:]
    else:
        df = pd.read_csv(path, dtype=np.float64)
        y = df[label].to_numpy()
        feats = df.drop(columns=label).to_numpy()
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X = np.hstack([np.ones((len(y), 1)), (feats - mean) / std])
    return [(y, X)]


def npy_chunks(y, X, chunk_mb):
    """(y, X) pairs over consecutive row ranges, read from disk if X is a memmap."""
    rows = max(1, chunk_mb * 2**20 // (X.shape[1] * 8))
    for a in range(0, X.shape[0], rows):
        yield y[a:a + rows], np.asarray(X[a:a + rows])


if args.npy and not args.stream:
    # Loading needs room for all of X and y; report "does not fit" as the
    # result instead of letting the machine run out of memory.
    rows, cols = np.load(f"{args.npy}.X.npy", mmap_mode="r").shape
    need_gb = rows * (cols + 1) * 8 / 1e9
    ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    if need_gb > 0.85 * ram_gb:
        msg = (f"The data ({need_gb:.1f} GB) does not fit in this machine's "
               f"memory ({ram_gb:.0f} GB).")
        print(msg)
        print(json.dumps({"error": msg, "kind": "memory", "records": rows,
                          "features": cols - 1, "gb": need_gb, "ram_gb": ram_gb}))
        raise SystemExit(0)

t0 = time.perf_counter()
if args.npy:
    # Labels are 1/30 of the data; keep them in memory either way.
    y_all = np.load(f"{args.npy}.y.npy")
    X_all = np.load(f"{args.npy}.X.npy", mmap_mode="r" if args.stream else None)
    source = "npy-stream" if args.stream else "npy-memory"
    blocks = [(y_all, X_all)]
    chunks = ((lambda: npy_chunks(y_all, X_all, args.chunk_mb)) if args.stream
              else (lambda: blocks))
else:
    source = "csv-memory"
    blocks = load_csv(args.data, args.label)
    chunks = lambda: blocks
load_s = time.perf_counter() - t0

n, d = blocks[0][1].shape

w = np.zeros(d)
per_iter = []
for _ in range(args.iters):
    t = time.perf_counter()
    g = np.zeros(d)
    loss_sum = 0.0
    for y, X in chunks():
        z = X @ w
        g += X.T @ (sigmoid(z) - y)
        loss_sum += np.sum(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))
    loss = loss_sum / n
    w = w - LR * g / n
    per_iter.append(time.perf_counter() - t)
    print(f"iter {len(per_iter) - 1:3d}   loss = {loss:.6f}   ({per_iter[-1]:.3f}s)",
          flush=True)

warm = per_iter[1:] if len(per_iter) > 1 else per_iter
out = {
    "source": source, "records": n, "features": d - 1, "iters": len(per_iter),
    "threads": args.threads, "load_s": load_s,
    "gb": n * (d + 1) * 8 / 1e9,
    "sec_per_iter_warm": statistics.median(warm), "final_loss": float(loss),
    "per_iter": per_iter,
}
print(f"{source}: records = {n:,}   load = {load_s:.2f}s   "
      f"sec/iter (warm) = {out['sec_per_iter_warm']:.4f}   loss = {loss:.6f}")
print(json.dumps(out))
