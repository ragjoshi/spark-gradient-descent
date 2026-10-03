# make_big_data.py: build the single-machine copy of a (big) dataset.
#
# Usage:
#   python make_big_data.py OUT SRC [SRC ...] [--label COL] [--copies K]
#
# SRC is a local CSV, an s3:// object, or an s3:// prefix ending in "/"
# (every object under it, in name order). .gz files are decompressed.
# Without --label the files are in the HIGGS layout (no header, label in
# column 0); with --label COL each file starts with a header row and COL is
# the label. Values must all be numeric (validate a sample with the app).
#
# Writes OUT.X.npy (float64, columns [1.0, x1 ... xd], features standardized
# the way train.standardize does it) and OUT.y.npy (labels), with the data
# repeated K times. numpy_baseline.py --npy OUT reads them. X is kept
# separate from y so it is contiguous, which BLAS needs for speed.
#
# The data is streamed in chunks, twice (statistics, then writing), so
# neither memory nor local disk has to hold the CSV.
import argparse
import subprocess
import time

import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description="Write K standardized copies of CSV data as .npy")
p.add_argument("out", help="output prefix; writes OUT.X.npy and OUT.y.npy")
p.add_argument("src", nargs="+")
p.add_argument("--label", help="label column; files then have a header row")
p.add_argument("--copies", type=int, default=1)
p.add_argument("--chunk-rows", type=int, default=500_000)
args = p.parse_args()


def expand(sources):
    """Local paths and s3:// URIs, with s3:// prefixes expanded to their objects."""
    out = []
    for src in sources:
        if src.startswith("s3://") and src.endswith("/"):
            listing = subprocess.run(["aws", "s3", "ls", "--recursive", src],
                                     check=True, capture_output=True, text=True).stdout
            bucket = src[5:].split("/", 1)[0]
            keys = sorted(line.split(None, 3)[3] for line in listing.splitlines()
                          if line.strip() and int(line.split(None, 3)[2]) > 0)
            out += [f"s3://{bucket}/{k}" for k in keys]
        else:
            out.append(src)
    if not out:
        raise SystemExit(f"no input files found in {sources}")
    return out


def chunks(sources):
    """(y, features) float64 arrays, chunk by chunk, across every source."""
    for src in sources:
        proc = None
        if src.startswith("s3://"):
            proc = subprocess.Popen(["aws", "s3", "cp", src, "-"], stdout=subprocess.PIPE)
            f = proc.stdout
        else:
            f = open(src, "rb")
        reader = pd.read_csv(f, header=0 if args.label else None, dtype=np.float64,
                             chunksize=args.chunk_rows,
                             compression="gzip" if src.endswith(".gz") else None)
        for df in reader:
            if args.label:
                if args.label not in df.columns:
                    raise SystemExit(f"label column {args.label!r} not in header of {src}")
                yield df[args.label].to_numpy(), df.drop(columns=args.label).to_numpy()
            else:
                a = df.to_numpy()
                yield a[:, 0], a[:, 1:]
        f.close()
        if proc is not None and proc.wait() != 0:
            raise SystemExit(f"reading {src} failed")


sources = expand(args.src)
print(f"{len(sources)} input file(s)", flush=True)
t0 = time.perf_counter()

# Pass 1: row count and per-feature sums, for the same mean and population
# standard deviation train.standardize computes.
n, s, ss, labels = 0, None, None, set()
for y, x in chunks(sources):
    if s is None:
        s, ss = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    n += len(y)
    s += x.sum(axis=0)
    ss += (x * x).sum(axis=0)
    labels.update(np.unique(y).tolist())
if not labels <= {0.0, 1.0}:
    raise SystemExit(f"labels must be 0 or 1, found {sorted(labels)[:10]}")
mean = s / n
std = np.sqrt(np.maximum(ss / n - mean * mean, 0.0))
std = np.where(std < 1e-12, 1.0, std)
d = len(mean)
print(f"pass 1: {n:,} rows, {d} features ({time.perf_counter() - t0:.0f}s)", flush=True)

# Pass 2: write the first copy straight from the source, then duplicate it
# within the output file.
X = np.lib.format.open_memmap(f"{args.out}.X.npy", mode="w+", dtype=np.float64,
                              shape=(n * args.copies, d + 1))
Y = np.lib.format.open_memmap(f"{args.out}.y.npy", mode="w+", dtype=np.float64,
                              shape=(n * args.copies,))
a = 0
for y, x in chunks(sources):
    b = a + len(y)
    X[a:b, 0] = 1.0
    X[a:b, 1:] = (x - mean) / std
    Y[a:b] = y
    a = b
for k in range(1, args.copies):
    for a in range(0, n, args.chunk_rows):
        b = min(a + args.chunk_rows, n)
        X[k * n + a:k * n + b] = X[a:b]
        Y[k * n + a:k * n + b] = Y[a:b]
X.flush()
Y.flush()
gb = n * args.copies * (d + 2) * 8 / 1e9
print(f"wrote {n * args.copies:,} rows ({args.copies} x {n:,}), {gb:.1f} GB to "
      f"{args.out}.X.npy / .y.npy in {time.perf_counter() - t0:.0f}s")
