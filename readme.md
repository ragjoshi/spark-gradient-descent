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

## Spark vs. plain NumPy

The scaling numbers above say how Spark compares with *itself* on fewer cores. The
more important question is how it compares with no Spark at all. `numpy_baseline.py`
runs the same gradient descent on one machine in plain NumPy:

| Version (1M HIGGS rows, M2 MacBook Pro) | Best s/iter | vs. NumPy |
|---|--:|--:|
| Plain NumPy, one process (`numpy_baseline.py`) | 0.030 | 1× |
| Spark, `--mode blocks`, 4 cores | 0.326 | ~11× slower |
| Spark, `--mode rows` (original), 4 cores | 0.889 | ~30× slower |

`--mode rows` runs the gradient one row at a time in Python, so each task pickles and
loops over 125,000 tuples. `--mode blocks` (`train.to_blocks`) packs each partition into
NumPy matrices once, so the gradient is one `X @ w` per block. That is 3.8× faster on
1 core (0.825 vs. 3.10 s/iter) and gives the same weights to ~1e-16.

Blocks mode, median of 3 interleaved reps:

| Cores | Median s/iter | Speedup |
|------:|--------------:|--------:|
| 1     | 0.825         | 1.00×   |
| 2     | 0.486         | 1.70×   |
| 4     | 0.326         | 2.53×   |
| 8     | 0.454         | noisy (0.30–0.63); the M2 has 4 performance + 4 efficiency cores |

Blocks mode scales worse than rows mode because there is less work left to
parallelize. What remains is mostly **fixed per-iteration overhead**. The same job on
a 20,000-row file still takes 0.26 s/iter on 4 cores, so roughly 80% of the 0.326 s is
coordination, not math. Most of it is a ~50 ms stall per Python task in PySpark 4.1:
the JVM-side Python runner and the worker each wait on the other before a task starts.
A job that runs entirely in the JVM costs ~5 ms per task; a Python task costs ~50 ms,
even with a Unix domain socket instead of TCP.

So on data that fits in one machine's RAM, NumPy wins: it reads 230 MB from memory in
30 ms, which is less than Spark's fixed cost for a single iteration. Spark's overhead
only becomes negligible when each iteration has much more data to process than one
machine can hold in memory.

## Repository structure

```
spark-project/
├── gradient.py      # Core math: gradient, loss, sigmoid (validated, standalone)
├── train.py         # Training driver: CSV loader, distributed GD loop, correctness checks
├── preprocess.py    # Validates and cleans uploaded CSVs with readable errors
├── bench.py         # Strong-scaling benchmark harness (one run per process, JSON output)
├── numpy_baseline.py # The same gradient descent in plain NumPy on one machine
├── app.py           # Streamlit front end: upload a CSV, run the scaling benchmark
├── analyze.py       # Computes speedup / efficiency / serial fraction from results.csv
├── requirements.txt # Pinned Python dependencies
├── Dockerfile       # Container image for the app (Python 3.12 + Java 21)
├── deploy/          # EC2 app deploy; bigdata/ = cluster benchmark (see Big-data benchmark)
├── make_big_data.py # Builds K standardized copies of a CSV as .npy for numpy_baseline.py
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

Add `--mode blocks` to any run to use the vectorized gradient (one NumPy matrix per
block instead of one Python row at a time); HIGGS runs in that mode append to
`results_blocks.csv`. Custom-data runs never write to `results.csv`. `--check` adds the correctness checks
(scikit-learn cosine similarity, and an exact comparison against the same gradient
descent run in plain NumPy). The last line of output is always a JSON object.

`train.py` is imported by `bench.py` (it exposes the loader, standardizer, and training
loop). It can also be run directly for a single 200-iteration training run with full
diagnostics: `python train.py` for HIGGS, or `python train.py mydata.csv target`.
For the single-machine baseline, run `python numpy_baseline.py` (`--threads 1` to
cap NumPy at one core). Once results are collected, compute the speedup / efficiency / serial-fraction table
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

The app runs `bench.py` once per core count (1, 2, 4, 8 by default, capped at the
machine's core count and the 8 fixed partitions), each in
its own process so every run gets a fresh `SparkContext`. It then shows:

- **Speedup vs 1 core**, actual against ideal linear speedup
- **Seconds per iteration** and parallel efficiency for each core count
- **Recommended cores**: the smallest core count whose speedup is within 10% of the
  best observed, labeled as a quick estimate for this dataset on this machine. It
  compares Spark runs with each other only, so it is not general Spark guidance and
  does not say whether Spark beats a single machine without Spark
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

## Deploying to AWS

To use the app from other machines, run it on an EC2 instance. The app runs in
Docker (`Dockerfile`: Python 3.12 + Java 21), and Spark still uses `local[N]` on
that one instance.

1. **Launch an instance** in the EC2 console:
   - AMI: Amazon Linux 2023 (x86_64).
   - Type: `c7a.2xlarge` (8 vCPUs, 16 GB, about $0.41/hour). On AMD `c7a`, each
     vCPU is a full physical core, so 1→8 cores is a fair scaling test. On Intel
     types like `c7i`, 8 vCPUs are only 4 physical cores with hyperthreading.
   - Storage: 30 GB (Docker image plus uploads of up to 1 GB each).
   - Key pair: create or choose one and keep the `.pem` file.
   - Security group: SSH (22) from **My IP**, and HTTP (80) from the IPs of the
     machines you will test from. Use "Anywhere" only together with `APP_PASSWORD`.
   - Advanced details → User data: paste the contents of `deploy/user-data.sh`
     (it installs Docker).
2. **Deploy** from this folder, about a minute after the instance starts:
   ```bash
   APP_PASSWORD=choose-a-password deploy/deploy.sh <instance-public-ip> ~/path/to/key.pem
   ```
   This copies the project with rsync, builds the image on the instance, and starts
   the container on port 80, set to restart if it stops. Run the same command again
   to redeploy after code changes. Leave out `APP_PASSWORD` to skip the password page.
3. Open `http://<instance-public-ip>` on any machine.

Notes:

- Only one benchmark runs at a time across all users, so concurrent runs can't
  distort each other's timings. Others see a "wait" message.
- The site is plain HTTP, so the password and uploads are not encrypted. Keep the
  security group restricted to your IPs.
- Uploads stay in the container's `/tmp` until the next redeploy.
- **Stop the instance** when you're done. It is billed by the hour while running.

## Big-data benchmark on AWS

`deploy/bigdata/` tests the case Spark is built for: data larger than one
machine's memory. It uses 10 copies of the full HIGGS dataset (110M rows,
75 GB as CSV, 26 GB as float64) and compares:

- **Spark** on an EMR cluster (4 spot nodes, 8 vCPUs and 64 GB each), data
  cached in memory across the nodes, run on all 4 nodes and on 2.
- **NumPy on a 16 GB machine** (`m6id.xlarge`), which must re-read the data
  from its NVMe disk on every iteration (`numpy_baseline.py --stream`).
- **NumPy on a 64 GB machine** (`r6id.2xlarge`), which holds it all in RAM.

All three report the loss after the same number of iterations, which must
match. Requirements: the AWS CLI, configured with credentials and a region.

```bash
deploy/bigdata/run.sh check     # credentials and vCPU quotas; launches nothing
deploy/bigdata/run.sh up        # launch (about $3-8 in total)
deploy/bigdata/run.sh status
deploy/bigdata/run.sh results   # download and compare
deploy/bigdata/run.sh down      # stop anything still running
```

Every machine terminates itself when its benchmark finishes (hard caps: 3 h
for the cluster's step, 2.5 h for the NumPy instances). The data stays in S3
until you delete the bucket; `down` prints the command.

## Notes

All metrics in this README are personally reproducible via the committed scripts and
`results.csv`. Speedup and efficiency come from `analyze.py`; the correctness figure
comes from the scikit-learn comparison in the training validation step.