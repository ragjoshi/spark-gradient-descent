# numpy_baseline.py: the same batch gradient descent on one machine, no Spark.
#
# Usage:
#   python numpy_baseline.py [--data FILE] [--iters N] [--threads T]
#
# Loads the HIGGS layout (no header, label in column 0) into memory,
# standardizes it like train.standardize, and times each iteration of
# w -= lr * X.T @ (sigmoid(X @ w) - y) / n. This is the bar Spark has to beat.
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
p.add_argument("--iters", type=int, default=50)
p.add_argument("--threads", type=int)
args = p.parse_args()

# BLAS reads these when NumPy is first imported, so set them before that.
if args.threads is not None:
    for var in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)

import numpy as np
import pandas as pd

from gradient import sigmoid

LR = 0.5

t0 = time.perf_counter()
raw = pd.read_csv(args.data, header=None, dtype=np.float64).to_numpy()
load_s = time.perf_counter() - t0

y = raw[:, 0]
feats = raw[:, 1:]
mean = feats.mean(axis=0)
std = feats.std(axis=0)
std = np.where(std < 1e-12, 1.0, std)
X = np.hstack([np.ones((len(y), 1)), (feats - mean) / std])
del raw, feats
n, d = X.shape

w = np.zeros(d)
per_iter = []
for _ in range(args.iters):
    t = time.perf_counter()
    z = X @ w
    grad = X.T @ (sigmoid(z) - y) / n
    loss = np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))
    w = w - LR * grad
    per_iter.append(time.perf_counter() - t)

warm = per_iter[1:] if len(per_iter) > 1 else per_iter
out = {
    "records": n, "features": d - 1, "iters": len(per_iter),
    "threads": args.threads, "load_s": load_s,
    "sec_per_iter_warm": statistics.median(warm), "final_loss": float(loss),
}
print(f"records = {n:,}   load = {load_s:.2f}s   "
      f"sec/iter (warm) = {out['sec_per_iter_warm']:.4f}   loss = {loss:.6f}")
print(json.dumps(out))
