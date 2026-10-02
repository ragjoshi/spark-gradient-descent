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
├── train.py         # Training driver: CSV loader, distributed GD loop, correctness checks
├── preprocess.py    # Validates and cleans uploaded CSVs with readable errors
├── bench.py         # Strong-scaling benchmark harness (one run per process, JSON output)
├── app.py           # Streamlit front end: upload a CSV, run the scaling benchmark
├── analyze.py       # Computes speedup / efficiency / serial fraction from results.csv
├── requirements.txt # Pinned Python dependencies
├── results.csv      # Clean benchmark results (9 runs: 1/2/4 cores × 3 reps)
├── results_raw.csv  # Unfiltered measurement log (includes caught contamination)
└── higgs_1m.csv     # 1M-row HIGGS subset (28 features + label) — not committed
```

## Running it

Requirements: Python 3.12 and a JDK supported by Spark 4 (17 or 21; developed against
OpenJDK 21, ARM64). Python dependencies are pinned in `requirements.txt`: PySpark,
NumPy, scikit-learn (validation baseline), pandas (CSV preprocessing), and Streamlit
(the app).

```bash
pip install -r requirements.txt
```

A single benchmark run takes the core count and repetition index as positional
arguments:

```bash
python bench.py <cores> <rep>     # e.g. python bench.py 4 0
```

With no other options this runs the HIGGS benchmark (50 iterations) and appends the
result to `results.csv`. To benchmark any numeric CSV with a header row and a 0/1
label column instead:

```bash
python bench.py 4 0 --data mydata.csv --label target --iters 30 --check
```

Custom-data runs never write to `results.csv`. `--check` adds the correctness checks
(scikit-learn cosine similarity, and an exact comparison against the same gradient
descent run in plain NumPy). The last line of output is always a JSON object.

`train.py` is imported by `bench.py` (it exposes the loader, standardizer, and training
loop). It can also be run directly for a single 200-iteration training run with full
diagnostics: `python train.py` for HIGGS, or `python train.py mydata.csv target`.
Once results are collected, compute the speedup / efficiency / serial-fraction table
with:

```bash
python analyze.py
```

The HIGGS dataset is available from the UCI Machine Learning Repository; `higgs_1m.csv`
is a 1,000,000-row subset. It is not committed to the repo due to size.

## Running the app

The Streamlit app lets anyone upload a CSV and watch how training time changes with
more cores.

```bash
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501) and:

1. Upload a CSV with a header row, numeric feature columns, and a label column
   containing only 0 and 1. Comma, semicolon, and tab separators are all accepted.
2. Pick the label column. The app validates the file immediately: non-numeric columns,
   bad labels, and similar problems are listed in plain language, rows with missing
   values are dropped (and counted), and columns that look like row IDs are flagged.
3. Choose the maximum core count and number of iterations, then click
   **Run benchmark**.

The app runs `bench.py` once per core count (1, 2, 4, ... up to the maximum), each in
its own process so every run gets a fresh `SparkContext`. It then shows:

- **Speedup vs 1 core**, actual against ideal linear speedup
- **Seconds per iteration** and parallel efficiency for each core count
- **Correctness**: whether the Spark weights match the same gradient descent run in
  plain NumPy on one machine (datasets up to 200,000 rows), and cosine similarity
  against scikit-learn. On linearly separable data scikit-learn's unregularized
  weights diverge, so the app says when a low cosine is expected.

Things to know:

- All runs use Spark `local[N]` mode on one machine: N worker threads, not a
  multi-machine cluster.
- Each core count runs once, so app timings are demonstrations. The results in this
  README come from `bench.py` with 3 interleaved repetitions per core count.
- Below about 100,000 rows, Spark's fixed per-iteration overhead outweighs the
  gradient math, so speedup says little about how the training scales. The app shows
  a note in that case.
- The app sets `PYSPARK_PYTHON` to its own interpreter, and on macOS sets
  `JAVA_HOME` to Java 21 if it is installed and `JAVA_HOME` is not already set.
- Uploads up to 1 GB are allowed (set in `.streamlit/config.toml`, which Streamlit
  reads when the app is started from the project folder). The full 1M-row HIGGS file
  with a header row is about 700 MB.
- Restart the app after editing `train.py` or `preprocess.py`; Streamlit does not
  reload imported modules.

## Notes

All metrics in this README are personally reproducible via the committed scripts and
`results.csv`. Speedup and efficiency come from `analyze.py`; the correctness figure
comes from the scikit-learn comparison in the training validation step.