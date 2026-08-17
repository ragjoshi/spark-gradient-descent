# Distributed Logistic Regression on Spark

A from-scratch implementation of batch gradient descent for logistic regression on
Apache Spark (PySpark, RDD API), benchmarked for strong scaling on a 1M-row physics
dataset. No MLlib — the distributed gradient computation, aggregation, and training
loop are all hand-written to expose the actual mechanics of at-scale optimization.

## Results

Strong-scaling benchmark on a 1M-row subset of the HIGGS dataset, logistic regression
via distributed batch gradient descent. Times are the **median warm per-iteration time**
across 3 repetitions per core count (iteration 0 dropped for JVM warmup).

| Cores | Median s/iter | Speedup | Parallel efficiency |
|------:|--------------:|--------:|--------------------:|
| 1     | 3.1032        | 1.00×   | 100.0%              |
| 2     | 1.7015        | 1.82×   | 91.2%               |
| 4     | 0.8649        | 3.59×   | 89.7%               |

**3.59× speedup on 4 cores at ~90% efficiency.** Scaling is sublinear by design —
the estimated serial fraction (Amdahl) sits in the single-digit percent range,
attributable to Python task serialization and fixed per-iteration driver overhead
(broadcast + result collection). See [Scaling analysis](#scaling-analysis).

Correctness: the trained weight vector matches a scikit-learn `LogisticRegression`
baseline at **0.9915 cosine similarity** (lr=0.5, 200 iterations).

Reproduce the results table:
```bash
python analyze.py
```

## What it does

- Loads the HIGGS dataset (28 features + binary label) as a Spark RDD.
- Runs batch gradient descent for logistic regression, where each iteration computes
  the full-dataset gradient in a single distributed pass.
- Uses `treeAggregate` to accumulate the gradient, loss, and record count together in
  one scan, combining partition-level **sums** (not means) for correctness across
  unevenly sized partitions.
- Broadcasts the current weight vector each iteration and destroys it afterward, to
  prevent driver-side memory accumulation over a long training run.

## Key engineering decisions

**`treeAggregate` over `aggregate`.** A flat `aggregate` funnels every partition's
partial result directly to the driver, making the driver a bottleneck as partition
count grows. `treeAggregate` combines partials in a multi-level tree, so the reduction
scales with the cluster rather than the driver.

**Partition count is a floor, not a target.** `sc.textFile(path, n)` treats `n` as a
*minimum* — the loader was silently producing 22 partitions against an intended 8,
inflating scheduling and serialization overhead and roughly **tripling** per-iteration
time. Fixed with an explicit `.repartition(n_parts)` in the load step. This was the
single largest performance correction in the project.

**Single-pass accumulation.** Gradient, loss, and count are accumulated in one scan
rather than three separate passes over the data — cutting the per-iteration data
movement by ~3×.

**Broadcast-destroy per iteration.** The weight vector is broadcast at the start of
each iteration and explicitly destroyed at the end, rather than relying on Spark to
garbage-collect stale broadcasts.

## Benchmark methodology

The machine is an M2 MacBook Pro, so thermal throttling is a real confound —
two identical `local[4]` runs were observed to vary by ~24%. The benchmark controls
for this:

- **Interleaved core order.** Runs cycle 1 → 2 → 4 cores per repetition rather than
  running all reps of one core count back-to-back, so thermal drift doesn't align with
  any single condition.
- **Median, not mean.** The reported statistic is the median across repetitions, which
  is robust to the occasional thermally throttled outlier.
- **Warm iterations only.** Iteration 0 is dropped from every run (JVM warmup, ~2.5s);
  the remaining iterations are flat, and their median is the per-iteration time.
- **Cooldown gaps + closed background apps** between runs.

The raw, unfiltered measurement log is preserved in `results_raw.csv`. It contains
three contaminated rows caught during analysis — two cold-start `local[4]` warmups and
one stray 130-iteration run — which are excluded from the clean `results.csv`. Keeping
both files documents the filtering rather than hiding it.

## Scaling analysis

Speedup is sublinear (3.59× on 4 cores, not 4×), and this is expected rather than a
defect. Two effects account for the gap:

1. **Python task serialization.** Each task's closures and data are pickled and
   unpickled across the JVM/Python boundary. This is a fixed cost per task that does
   not parallelize.
2. **Per-iteration driver overhead.** Broadcasting the weight vector and collecting the
   aggregated gradient each iteration is serial driver work.

The Amdahl-inverted serial fraction is not a single constant (≈9.7% at 2 cores vs
≈3.8% at 4 cores) — the fact that it *shifts* is itself the tell that this isn't a pure
compute-bound Amdahl system, but one where fixed overhead is a larger relative share at
low core counts. The honest headline is therefore **~90% parallel efficiency**, with
the serial fraction cited as an estimated single-digit-percent range.

## Repository structure

```
spark-project/
├── gradient.py      # Core math: gradient, loss, sigmoid (validated, standalone)
├── train.py         # Training driver: distributed GD loop over the RDD
├── bench.py         # Strong-scaling benchmark harness
├── analyze.py       # Computes speedup / efficiency / serial fraction from results.csv
├── results.csv      # Clean benchmark results (9 runs: 1/2/4 cores × 3 reps)
├── results_raw.csv  # Unfiltered measurement log (includes caught contamination)
└── higgs_1m.csv     # 1M-row HIGGS subset (28 features + label) — not committed
```

## Running it

Requirements: Python 3.12, PySpark, NumPy, scikit-learn (for the validation baseline),
and a JDK (developed against OpenJDK 21, ARM64).

```bash
pip install pyspark numpy scikit-learn
```

A single benchmark run takes the core count and repetition index as positional
arguments:

```bash
python bench.py <cores> <rep>     # e.g. python bench.py 4 0
```

`train.py` is imported by `bench.py` (it exposes the loader, standardizer, and training
loop) rather than run directly. Once results are collected, compute the speedup /
efficiency / serial-fraction table with:

```bash
python analyze.py
```

The HIGGS dataset is available from the UCI Machine Learning Repository; `higgs_1m.csv`
is a 1,000,000-row subset. It is not committed to the repo due to size.

## Notes

All metrics in this README are personally reproducible via the committed scripts and
`results.csv`. Speedup and efficiency come from `analyze.py`; the correctness figure
comes from the scikit-learn comparison in the training validation step.