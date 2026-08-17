"""
Week 1: Distributed logistic regression — a single, correct gradient step.

Goal: compute ONE gradient of the logistic-regression log-loss over an RDD,
and prove it matches a plain NumPy reference to floating-point tolerance.
No iteration loop yet. No performance work yet. Just correctness.

The math you're implementing:
    For one example (x, y) with weights w:
        z        = w · x
        p        = sigmoid(z)          # predicted probability
        grad_i   = (p - y) * x         # this example's gradient contribution
    The full gradient is the MEAN of grad_i over all N examples.

The distributed structure (this is the "I understand the engine" part):
    - Each PARTITION computes a local partial sum of grad_i over only its rows.
    - We then sum those partials across partitions into the full gradient.
    That map-side partial reduction is what makes this scale.
"""

import numpy as np
from pyspark.sql import SparkSession
from sklearn.datasets import make_classification


# ----------------------------------------------------------------------
# 1. sigmoid
# ----------------------------------------------------------------------
def sigmoid(z):
    """
    Logistic sigmoid: 1 / (1 + exp(-z)).

    TODO: implement this.

    Watch out for numerical overflow: for very negative z, exp(-z) blows up
    to inf. The naive formula still returns ~0.0 in that case (inf in the
    denominator), but you'll get a RuntimeWarning. A numerically stable
    version handles the sign of z separately. For Week 1's small data a
    naive version is fine to start — get it working, then harden it.

    Accepts a scalar or a NumPy array; should work elementwise either way.
    """
    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------------------------
# 2. per-partition gradient  (the heart of the project)
# ----------------------------------------------------------------------
def partition_gradient(rows, w):
    """
    Compute the LOCAL partial gradient sum over the rows in ONE partition.

    `rows` is an iterator of (label, feature_vector) tuples, where
    feature_vector is a 1-D NumPy array. `w` is the current weight vector
    (same length as feature_vector).

    Return: a single NumPy array — the SUM (not mean) of (p - y) * x over
    just the rows this partition holds. We sum here and divide by N once,
    globally, at the end. That keeps the per-partition step a pure sum,
    which is what aggregates cleanly.

    TODO:
      - initialise an accumulator of zeros, shape (len(w),)
      - for each (y, x): z = w·x, p = sigmoid(z), acc += (p - y) * x
      - return acc

    Note: `rows` is an ITERATOR — you can only walk it once. Don't try to
    len() it or index into it.
    """
    acc = np.zeros(len(w))
    for (y, x) in rows:
        z = np.dot(w, x)      # scalar:  w · x   (NOT w * x, which is elementwise)
        p = sigmoid(z)        # scalar probability
        acc += (p - y) * x    # this row's vector contribution
    return acc


def spark_gradient(rdd, w):
    """
    Full distributed gradient = (1/N) * sum of per-partition partial sums.

    mapPartitions runs partition_gradient on each partition, yielding one
    partial-sum array per partition. reduce() then sums those partials.

    (In Week 2 you'll swap this reduce for treeAggregate, so the summing
    happens in a tree instead of dumping every partial on the driver at
    once — that's the real distributed-aggregation upgrade. For now, a
    plain reduce is correct and easy to reason about.)
    """
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
    """
    The same gradient computed directly on the full dataset with NumPy,
    fully vectorised. This is your source of truth.

    X is (N, d), y is (N,), w is (d,). Return a (d,) array.

    TODO: implement in a vectorised way (no Python loop over rows).
    Hint: predictions p = sigmoid(X @ w); the gradient is
          X.T @ (p - y) / N.
    """
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
    print("\n✅  Gradients match at every test point. Distributed gradient step is correct.")

    spark.stop()


if __name__ == "__main__":
    main()