# train.py — Week 2: distributed logistic regression on HIGGS (+ timing)
import time
import numpy as np
import statistics
from pyspark import SparkContext
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from gradient import sigmoid


def load_higgs(sc, path, n_parts=8):
    def parse(line):
        vals = line.split(",")
        y = float(vals[0])
        x = np.array(vals[1:], dtype=np.float64)
        return (y, x)
    return sc.textFile(path).map(parse).repartition(n_parts)


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


def validate(data, w):
    rows = data.sample(False, 0.2, seed=0).collect()
    y = np.array([r[0] for r in rows])
    X = np.vstack([r[1] for r in rows])

    ref = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=1000)
    ref.fit(X, y)
    w_ref = ref.coef_.ravel()

    p_mine = sigmoid(X.dot(w))
    p_ref  = ref.predict_proba(X)[:, 1]

    cos = w.dot(w_ref) / (np.linalg.norm(w) * np.linalg.norm(w_ref))
    print(f"cosine(w, w_ref)   = {cos:.6f}")
    print(f"max |w - w_ref|    = {np.max(np.abs(w - w_ref)):.6f}")
    print(f"mine  acc / logloss = {accuracy_score(y, p_mine > 0.5):.4f}"
          f" / {log_loss(y, p_mine):.6f}")
    print(f"skl   acc / logloss = {accuracy_score(y, p_ref  > 0.5):.4f}"
          f" / {log_loss(y, p_ref):.6f}")


if __name__ == "__main__":
    CORES = 4          # Week 3: rerun with 1, 2, 4 and compare timing
    N_PARTS = 8        # keep FIXED across all runs -> honest strong-scaling

    sc = SparkContext(f"local[{CORES}]", "logreg-higgs")
    sc.setLogLevel("WARN")

    raw = load_higgs(sc, "higgs_1m.csv", n_parts=N_PARTS)
    data = standardize(sc, raw, d_feat=28)
    d = 29

    w, stats = train(sc, data, d, lr=0.5, max_iter=200)
    validate(data, w)
    print("final weights:", np.round(w, 4))
    sc.stop()