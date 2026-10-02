# train.py: distributed logistic regression on any numeric CSV (+ timing)
import csv
import gzip
import itertools
import os
import sys
import tempfile
import time
import numpy as np
import statistics
from pyspark import SparkConf, SparkContext
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from gradient import sigmoid


def make_context(cores, app_name):
    # spark.log.level applies from JVM startup; setLogLevel covers older Sparks.
    conf = SparkConf().setMaster(f"local[{cores}]").setAppName(app_name) \
                      .set("spark.log.level", "WARN") \
                      .set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    return sc


def _first_line(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return next(csv.reader(f))


def load_csv(sc, path, label_col=None, n_parts=8):
    """
    Load a numeric CSV as an RDD of (label, feature_vector) pairs.

    label_col=None: no header row, label in column 0 (the HIGGS layout).
    label_col=name: first line is a header; the label is the named column.

    Every non-header value must parse as a float; run preprocess.clean_csv
    first on untrusted files. Returns (rdd, d_feat).
    """
    first = _first_line(path)
    d_feat = len(first) - 1

    if label_col is None:
        label_idx = 0
    elif label_col in first:
        label_idx = first.index(label_col)
    else:
        raise ValueError(f"label column {label_col!r} not in header of {path}")

    def parse(line):
        vals = line.split(",")
        y = float(vals[label_idx])
        x = np.array(vals[:label_idx] + vals[label_idx + 1:], dtype=np.float64)
        return (y, x)

    lines = sc.textFile(path)
    if label_col is not None:
        lines = lines.mapPartitionsWithIndex(
            lambda i, it: itertools.islice(it, 1, None) if i == 0 else it)
    return lines.map(parse).repartition(n_parts), d_feat


def load_higgs(sc, path, n_parts=8):
    return load_csv(sc, path, None, n_parts)[0]


def standardize(sc, data, d_feat):
    def seq_op(acc, row):
        s, ss, c = acc
        _, x = row
        return (s + x, ss + x * x, c + 1)

    def comb_op(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    s, ss, n = data.treeAggregate(
        (np.zeros(d_feat), np.zeros(d_feat), 0), seq_op, comb_op, depth=2)

    mean = s / n
    var = ss / n - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std < 1e-12, 1.0, std)

    mean_bc = sc.broadcast(mean)
    std_bc = sc.broadcast(std)

    def transform(row):
        y, x = row
        xs = (x - mean_bc.value) / std_bc.value
        x_full = np.concatenate(([1.0], xs))
        return (y, x_full)

    return data.map(transform)


def train(sc, data, d, lr=0.5, max_iter=200, tol=1e-6, verbose=True):
    data = data.cache()
    n = data.count()                      # materializes cache -> loop timing is "warm"
    n_parts = data.getNumPartitions()     # actual partition count, for the scaling log
    cores = sc.defaultParallelism         # = N in local[N]
    w = np.zeros(d)
    prev_loss = float("inf")

    per_iter = []                         # wall-clock seconds per iteration
    t_start = time.perf_counter()

    for it in range(max_iter):
        t_iter = time.perf_counter()
        w_bc = sc.broadcast(w)

        def seq_op(acc, row):
            g, loss, c = acc
            y, x = row
            z = float(x.dot(w_bc.value))
            g = g + (sigmoid(z) - y) * x
            loss += max(z, 0.0) - z * y + np.log1p(np.exp(-abs(z)))
            return (g, loss, c + 1)

        def comb_op(a, b):
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        g_sum, loss_sum, cnt = data.treeAggregate(
            (np.zeros(d), 0.0, 0), seq_op, comb_op, depth=2)
        w_bc.destroy()

        grad = g_sum / n
        loss = loss_sum / n
        w = w - lr * grad

        per_iter.append(time.perf_counter() - t_iter)
        if verbose: 
            print(f"iter {it:3d}   loss = {loss:.6f}   ({per_iter[-1]:.3f}s)")
        if abs(prev_loss - loss) < tol:
            if verbose:
                print(f"converged at iter {it}")
            break
        prev_loss = loss

    total = time.perf_counter() - t_start
    iters = len(per_iter)
    warm = per_iter[1:] if iters > 1 else per_iter   # drop iter 0 (JVM/JIT warmup)

    

    stats = {
        "cores": cores, "records": n, "partitions": n_parts, "iters": iters,
        "total_s": total,
        "sec_per_iter": total / iters,
        "sec_per_iter_warm": statistics.median(warm),
        "per_iter": per_iter,
    }

    if verbose: 
        print("\n--- timing ---")
        print(f"cores (local[N])   = {cores}")
        print(f"records            = {n:,}")
        print(f"partitions         = {n_parts}")
        print(f"iterations run     = {iters}")
        print(f"total train time   = {total:.2f} s")
        print(f"sec / iteration    = {stats['sec_per_iter']:.4f}")
        print(f"sec / iter (warm)  = {stats['sec_per_iter_warm']:.4f}")

    return w, stats


def validate(data, w, max_rows=None, verbose=True):
    """
    Compare w against scikit-learn's unregularized fit on a driver-side sample.

    max_rows=None: sample 20% of the data (the original HIGGS setting).
    max_rows=k:    sample about k rows, or all rows if the data is smaller.
    """
    frac = 0.2
    if max_rows is not None:
        frac = min(1.0, max_rows / data.count())
    rows = data.sample(False, frac, seed=0).collect()
    y = np.array([r[0] for r in rows])
    X = np.vstack([r[1] for r in rows])

    ref = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=1000)
    ref.fit(X, y)
    w_ref = ref.coef_.ravel()

    p_mine = sigmoid(X.dot(w))
    p_ref  = ref.predict_proba(X)[:, 1]

    cos = w.dot(w_ref) / (np.linalg.norm(w) * np.linalg.norm(w_ref))
    result = {
        "cosine": float(cos),
        "max_abs_diff": float(np.max(np.abs(w - w_ref))),
        "acc_mine": float(accuracy_score(y, p_mine > 0.5)),
        "acc_sklearn": float(accuracy_score(y, p_ref > 0.5)),
        "logloss_mine": float(log_loss(y, p_mine)),
        "logloss_sklearn": float(log_loss(y, p_ref)),
        "sample_rows": len(rows),
        # sklearn fit the sample perfectly: the classes are linearly separable,
        # its unregularized weights grow without bound, and cosine stops being
        # a meaningful comparison.
        "sklearn_separable": bool(accuracy_score(y, p_ref > 0.5) == 1.0),
    }
    if verbose:
        print(f"cosine(w, w_ref)   = {result['cosine']:.6f}")
        print(f"max |w - w_ref|    = {result['max_abs_diff']:.6f}")
        print(f"mine  acc / logloss = {result['acc_mine']:.4f}"
              f" / {result['logloss_mine']:.6f}")
        print(f"skl   acc / logloss = {result['acc_sklearn']:.4f}"
              f" / {result['logloss_sklearn']:.6f}")
        if result["sklearn_separable"]:
            print("note: sample is linearly separable; sklearn's weights "
                  "diverge, so cosine is not a reliable check here")
    return result


def reference_check(data, w, lr, iters, max_rows=200_000, verbose=True):
    """
    Re-run the same batch gradient descent on one machine with NumPy.

    Same rows, same starting point (zeros), same learning rate, same number
    of iterations, so the weights must agree up to floating-point summation
    order. This checks the claim the project makes: distributing the
    gradient computation does not change the result.

    Skipped above max_rows, since every row has to fit on the driver.
    """
    n = data.count()
    if n > max_rows:
        result = {"ref_checked": False,
                  "ref_reason": f"{n:,} rows exceeds the {max_rows:,}-row "
                                "limit for a single-machine copy"}
        if verbose:
            print(f"numpy reference    = skipped ({result['ref_reason']})")
        return result

    rows = data.collect()
    y = np.array([r[0] for r in rows])
    X = np.vstack([r[1] for r in rows])

    w_ref = np.zeros(X.shape[1])
    for _ in range(iters):
        grad = X.T @ (sigmoid(X @ w_ref) - y) / n
        w_ref = w_ref - lr * grad

    result = {"ref_checked": True,
              "ref_max_abs_diff": float(np.max(np.abs(w - w_ref))),
              "ref_match": bool(np.allclose(w, w_ref, rtol=1e-9, atol=1e-9))}
    if verbose:
        print(f"numpy reference    = max |w - w_np| {result['ref_max_abs_diff']:.2e}"
              f"  ({'MATCH' if result['ref_match'] else 'MISMATCH'})")
    return result


if __name__ == "__main__":
    # Usage: python train.py [data_path] [label_col]
    # No arguments: HIGGS layout (higgs_1m.csv, no header, label in column 0).
    CORES = 4          # Week 3: rerun with 1, 2, 4 and compare timing
    N_PARTS = 8        # keep FIXED across all runs -> honest strong-scaling
    LR = 0.5

    path = sys.argv[1] if len(sys.argv) > 1 else "higgs_1m.csv"
    label_col = sys.argv[2] if len(sys.argv) > 2 else None

    if label_col is not None:
        from preprocess import DataError, clean_csv
        cleaned = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        try:
            summary = clean_csv(path, label_col, cleaned)
        except DataError as e:
            os.remove(cleaned)
            sys.exit(f"error: {e}")
        print(f"rows used = {summary['rows']:,}  "
              f"(dropped {summary['rows_dropped']:,} with missing values)")
        if summary["ignored_columns"]:
            print(f"ignored index columns: {summary['ignored_columns']}")
        path = cleaned

    sc = make_context(CORES, "logreg")

    raw, d_feat = load_csv(sc, path, label_col, n_parts=N_PARTS)
    data = standardize(sc, raw, d_feat=d_feat)
    d = d_feat + 1

    w, stats = train(sc, data, d, lr=LR, max_iter=200)
    validate(data, w)
    reference_check(data, w, LR, stats["iters"])
    print("final weights:", np.round(w, 4))
    sc.stop()
    if label_col is not None:
        os.remove(path)