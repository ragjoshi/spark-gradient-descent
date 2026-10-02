# make_big_data.py: build the single-machine copy of the big dataset.
#
# Usage:
#   python make_big_data.py SRC.csv OUT [--copies K]
#
# Reads a HIGGS-layout CSV (no header, label in column 0), standardizes the
# features the same way train.standardize does, and writes K back-to-back
# copies as two float64 arrays: OUT.X.npy with columns [1.0, x1 ... xd]
# and OUT.y.npy with the labels. numpy_baseline.py --npy OUT reads them.
# X is kept separate from y so it is contiguous, which BLAS needs for speed. Repeating the data leaves the mean and
# standard deviation unchanged, so the stats from one copy are the stats of
# all K, and the result matches K copies of the CSV standardized in Spark.
import argparse
import time

import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description="Write K standardized copies of a CSV as .npy")
p.add_argument("src")
p.add_argument("out", help="output prefix; writes OUT.X.npy and OUT.y.npy")
p.add_argument("--copies", type=int, default=1)
args = p.parse_args()

t0 = time.perf_counter()
raw = pd.read_csv(args.src, header=None, dtype=np.float64).to_numpy()
n, cols = raw.shape
feats = raw[:, 1:]
mean = feats.mean(axis=0)
std = feats.std(axis=0)
std = np.where(std < 1e-12, 1.0, std)

y = raw[:, 0].copy()
block = np.empty((n, cols))
block[:, 0] = 1.0
block[:, 1:] = (feats - mean) / std
del raw, feats
print(f"read + standardized {n:,} rows in {time.perf_counter() - t0:.1f}s")

np.save(f"{args.out}.y.npy", np.tile(y, args.copies))
out = np.lib.format.open_memmap(f"{args.out}.X.npy", mode="w+", dtype=np.float64,
                                shape=(n * args.copies, cols))
for k in range(args.copies):
    out[k * n:(k + 1) * n] = block
    out.flush()
    print(f"copy {k + 1}/{args.copies} written")
del out
print(f"wrote {n * args.copies:,} rows, {n * args.copies * (cols + 1) * 8 / 1e9:.1f} GB "
      f"to {args.out}.X.npy / .y.npy in {time.perf_counter() - t0:.1f}s")
