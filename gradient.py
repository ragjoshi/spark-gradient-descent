

import numpy as np
from pyspark.sql import SparkSession
from sklearn.datasets import make_classification


# ----------------------------------------------------------------------
# 1. sigmoid
# ----------------------------------------------------------------------
def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------------------------
# 2. per-partition gradient  (the heart of the project)
# ----------------------------------------------------------------------
def partition_gradient(rows, w):
    
    acc = np.zeros(len(w))
    for (y, x) in rows:
        z = np.dot(w, x)      # scalar:  w · x   (NOT w * x, which is elementwise)
        p = sigmoid(z)        # scalar probability
        acc += (p - y) * x    # this row's vector contribution
    return acc


def spark_gradient(rdd, w):
    n = rdd.count()

    partials = rdd.mapPartitions(
        lambda rows: iter([partition_gradient(rows, w)])
    )
    total = partials.reduce(lambda a, b: a + b)
    return total / n


# ----------------------------------------------------------------------
# 3. NumPy reference gradient  (ground truth to check against)
# ----------------------------------------------------------------------
def numpy_gradient(X, y, w):
    N = len(y)
    p = sigmoid(X @ w)
    grad = X.T @ (p - y) / N
    return grad


# ----------------------------------------------------------------------
# Harness: generate data, run both, compare
# ----------------------------------------------------------------------
def main():
    spark = (
        SparkSession.builder
        .appName("logreg-week1-gradient")
        .master("local[8]")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")  # quiet the INFO wall

    # Small, deterministic dataset so you can trust the comparison.
    N, D = 500, 5
    X, y = make_classification(
        n_samples=N, n_features=D, n_informative=D,
        n_redundant=0, random_state=42,
    )
    X = X.astype(np.float64)
    y = y.astype(np.float64)

    # Build the RDD of (label, feature_vector) pairs across 8 partitions.
    rows = [(float(y[i]), X[i]) for i in range(N)]
    rdd = sc.parallelize(rows, numSlices=8)

    # Evaluate the gradient at SEVERAL weight points, not just the origin.
    # Why not only w = 0? Because sigmoid(0) = 0.5 no matter whether the 0 is
    # a scalar or a whole vector — so a scalar-vs-vector bug in `z = w·x`
    # produces the identical answer at w = 0 and stays completely hidden.
    # A nonzero w is what actually exercises the dot product. A test that can
    # only pass for the right reason is worth more than one that might pass
    # for the wrong reason.
    rng = np.random.default_rng(0)
    test_points = {
        "w = zeros ": np.zeros(D),
        "w = ones  ": np.ones(D),
        "w = random": rng.standard_normal(D),
    }

    all_ok = True
    for name, w in test_points.items():
        g_spark = spark_gradient(rdd, w)
        g_numpy = numpy_gradient(X, y, w)
        diff = np.max(np.abs(g_spark - g_numpy))
        ok = np.allclose(g_spark, g_numpy, atol=1e-9)
        all_ok &= ok
        print(f"{name}   max abs diff = {diff:.2e}   {'OK' if ok else 'MISMATCH'}")

    assert all_ok, \
        "MISMATCH — the distributed gradient disagrees with NumPy at some weight point"
    print("\nGradients match at every test point. Distributed gradient step is correct.")

    spark.stop()


if __name__ == "__main__":
    main()