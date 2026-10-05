# Distributed Logistic Regression on Spark

This project trains a logistic regression model with batch gradient descent on
Apache Spark. Everything is written from scratch on the PySpark RDD API, without
MLlib: the distributed gradient, how it is combined, and the training loop. That
keeps the cost of distributing the work visible.

It asks one practical question: **when is Spark worth it?** The same training
runs in two setups, and each one is also run without Spark:

| | Where it runs | Spark setup | Data |
|---|---|---|---|
| [Local run](#local-run-the-app-on-your-laptop) | Your laptop | `local[N]`: 1 machine, N cores | Any CSV you upload (tested on 1M rows) |
| [Cluster run](#cluster-run-4-machines-on-aws-emr) | AWS EMR, 4 worker machines | Spark on YARN, 32 cores | 110M rows, 26 GB in memory |

The short answer (details in [Results](#results)): on 1M rows the non-Spark
version on one machine is 11× faster than Spark. On 110M rows, too many for one
machine's memory, Spark on 4 machines is 30× faster than the non-Spark version on
a 16 GB machine.

## Contents

- [Technologies](#technologies)
- [How training works](#how-training-works)
- [Local run: the app on your laptop](#local-run-the-app-on-your-laptop)
- [Cluster run: 4 machines on AWS EMR](#cluster-run-4-machines-on-aws-emr)
- [Results](#results)
- [Challenges](#challenges)

---

## Technologies

| Technology | Purpose | Why this one |
|---|---|---|
| **Apache Spark 4.1** (PySpark, RDD API) on **Java 21** | Splits the data into partitions and computes the gradient on all of them in parallel | The RDD API exposes partitions, broadcasts and aggregation directly. MLlib would hide exactly the parts this project measures. |
| **NumPy** | The matrix math inside each Spark task, and the whole non-Spark version | Vectorized math on a block of rows is far faster than a Python loop, and it gives the non-Spark version a strong, fair implementation to compare against |
| **scikit-learn** | Reference logistic regression to check the trained weights against | An independent, well-tested implementation, so a bug in the hand-written gradient would show up |
| **pandas** | Reads and validates uploaded CSVs (`preprocess.py`); streams big CSVs into a single file for the non-Spark version (`make_big_data.py`) | Detects separators, types and missing values, and reports problems before Spark sees the data, where errors would surface as stack traces from inside tasks |
| **Streamlit** + Altair | The web app: upload a CSV, run the benchmark, see charts and a recommendation | A full interactive app in one Python file, with no separate frontend |
| **AWS EMR 7** (Spark on **YARN**), **Spot** instances | The 4-machine cluster | EMR comes with Spark and YARN already installed and configured. Spot workers cost a fraction of on-demand, and a benchmark can tolerate the occasional interruption. |
| **AWS EC2** (`m6id`, `r6id`) | Two single machines that run the non-Spark version on the big data | One machine with too little memory (16 GB) and one with enough (64 GB): together they show where Spark starts to pay off |
| **AWS S3** | Holds the code, the 75 GB dataset and the results; every machine reads from and writes to it | All machines can read it in parallel, it outlives the machines, and copies inside S3 never pass through a machine |
| **Bash + AWS CLI** | `deploy/bigdata/run.sh` and `deploy/app.sh` launch everything with one command | Repeatable, scriptable, and no console clicking |
| **Docker** | Packages the app (Python 3.12 + Java 21) for hosting on EC2 | The same environment on the laptop and the server |

Python dependencies are pinned in `requirements.txt`.

---

## How training works

The model is logistic regression trained by **batch gradient descent**. Each
iteration reads all the data once, computes the gradient (the direction that
reduces the loss), and moves the weights one step that way. That full pass over
the data is the expensive part, and it is what Spark parallelizes.

One iteration, in both setups:

```
                          ┌──────────────────────────────┐
                          │  Driver (one Python process) │
                          │  holds the weight vector w   │
                          └──────────────┬───────────────┘
                     1. broadcast w      │      (a few hundred bytes)
          ┌──────────────────┬───────────┴──────┬──────────────────┐
          ▼                  ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ partition 1 │    │ partition 2 │    │ partition 3 │ …  │ partition P │   data cached in
   │ rows 1…k    │    │ rows k+1…   │    │             │    │             │   worker memory
   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │ 2. each worker computes, for its own rows only:        │
          │    sum of gradients, sum of losses, row count          │
          └────────┬─────────┘                  └────────┬─────────┘
                   ▼  3. treeAggregate: partial sums     ▼
                   │     combined in a tree, not all at the driver
                   └────────────────────┬────────────────┘
                                        ▼
                     4. driver: w = w − learning_rate × (gradient sum / n)
                        destroy the old broadcast, start the next iteration
```

Before the first iteration, the data is loaded as an RDD of
`(label, feature vector)` rows, split into partitions, and standardized (one
`treeAggregate` pass for the means and standard deviations). It is then cached in
memory, so later iterations never re-read the file.

Design choices in `train.py` and `gradient.py`:

- **`treeAggregate`, not `aggregate`.** A flat `aggregate` sends every
  partition's partial result straight to the driver, which becomes a bottleneck as
  the partition count grows. `treeAggregate` combines them in a tree first.
- **One pass per iteration.** Gradient, loss and row count are accumulated in the
  same scan instead of three separate passes over the data.
- **Sums, not means.** Partitions combine sums, and the driver divides once by
  the total row count, so unevenly sized partitions still give the exact gradient.
- **Broadcast, then destroy.** The weights are broadcast at the start of each
  iteration and destroyed at the end, so stale copies don't accumulate during a
  long run.
- **Two ways to compute a partition's gradient.** `--mode rows` loops over rows
  one at a time in Python (the original). `--mode blocks` packs each partition
  into NumPy matrices once (`train.to_blocks`), so the gradient is one `X @ w` per
  block. Both give the same weights; blocks is 3.8× faster on 1 core.

The only thing that changes between the two setups is **where the driver and
workers live**:

| Setup | Driver | Workers | Data comes from |
|---|---|---|---|
| Local | A process on your laptop | N threads on your laptop (`local[N]`) | The uploaded CSV |
| Cluster | The EMR master machine | 4 executors, one per worker machine, 8 cores each | S3 |

**Every run checks itself.** The non-Spark version (`numpy_baseline.py`) runs
the same gradient descent with the same data, starting weights, learning rate and
iteration count, and must reach the same weights (locally) or the same final loss
(on the cluster). The weights are also compared with scikit-learn's.

---

## Local run: the app on your laptop

Spark runs in `local[N]` mode: one machine, N worker threads. The app takes an
uploaded CSV, trains on it with 1, 2, 4 and 8 cores, trains it again without
Spark, and shows how the times compare.

### Set up and start

Requirements: Python 3.12 and a JDK supported by Spark 4 (17 or 21; developed
against OpenJDK 21 on ARM64).

1. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app from the project folder (so Streamlit picks up
   `.streamlit/config.toml`):

   ```bash
   streamlit run app.py
   ```

3. Open the URL it prints (usually http://localhost:8501).

The page has two tabs: **Run on this machine** (the local run) and **Recorded
cluster runs (AWS)**, which shows the results of the
[cluster run](#cluster-run-4-machines-on-aws-emr) from files in the repo, so it
works offline.

### Run it

1. Upload a CSV with a header row, numeric feature columns, and a label column
   containing only 0 and 1. Comma, semicolon and tab separators all work. To try
   it on HIGGS, download it from the UCI Machine Learning Repository; the results
   below use its first 1M rows (`higgs_1m.csv`, not committed because of its size).
2. Pick the label column. The app checks the file right away: non-numeric
   columns, bad labels and similar problems are listed in plain language, rows
   with missing values are dropped (and counted), and columns that look like row
   IDs are flagged.
3. Choose the maximum core count, the number of iterations, and the gradient code
   (vectorized blocks, the default, or row by row).
4. Click **Run benchmark**.

### What happens when you click Run benchmark

```
 Browser ──upload──▶ Streamlit (app.py)
                        │ 1. preprocess.clean_csv: validate, drop bad rows ─▶ clean.csv
                        │
                        │ 2. for N in 1, 2, 4, 8 (one at a time):
                        ├──▶ python bench.py N … ─▶ new JVM + SparkContext local[N]
                        │        load ─▶ standardize ─▶ cache ─▶ train (timed) ─▶ JSON
                        │
                        │ 3. python numpy_baseline.py …  (non-Spark, same training)
                        │
                        ▼ 4. charts, Spark vs. non-Spark recommendation, correctness
 Browser ◀──────────────┘
```

1. **The upload is saved and checked.** `preprocess.py` reads it with pandas,
   detects the separator, checks that every column is numeric and the label is
   0/1, drops rows with missing values, and writes a clean copy. This runs once
   per file and label, not on every click.
2. **Spark runs once per core count.** `bench.py` runs in its own process for each
   core count (1, 2, 4, 8, capped at the machine's cores and the 8 partitions),
   because `local[N]` cannot be resized inside a running JVM. Each run:
   - starts a JVM with N worker threads, each feeding a Python worker. NumPy is
     limited to one thread per worker, so N workers really use N cores;
   - splits the CSV into 8 partitions, standardizes the features, and caches them;
   - runs the timed training loop. Iteration 0 (JVM warmup) is dropped and the
     median of the rest is the result;
   - on the last core count, runs the correctness checks after the timed loop, so
     they never affect the timing;
   - prints one JSON line, which the app reads.
3. **The non-Spark version runs the same training** in one process
   (`numpy_baseline.py`).
4. **The app shows the results.** Only one benchmark runs at a time across all
   users (a lock), so runs can't distort each other's timings.

What you see:

- **Speedup vs. 1 core**, against ideal linear speedup
- **Seconds per iteration** and parallel efficiency for each core count
- **Spark vs. non-Spark**: both times on one chart, and a recommendation based on
  which was faster and whether the data fits in this machine's memory
- **Recommended cores**: the smallest core count within 10% of the best speedup
  seen. It compares Spark runs with each other only, so it does not say whether
  Spark beats the non-Spark version
- **Correctness**: whether Spark's weights match the non-Spark version's (for
  datasets up to 200,000 rows), and cosine similarity with scikit-learn. On
  linearly separable data scikit-learn's unregularized weights diverge, so the app
  says when a low cosine is expected

### Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit app: upload, validation, runs `bench.py` and `numpy_baseline.py` as subprocesses, charts, recommendation, and the recorded cluster runs tab |
| `preprocess.py` | Validates and cleans an uploaded CSV, with readable errors |
| `bench.py` | One benchmark run (core count, repetition) per process; prints one JSON line |
| `train.py` | Creates the SparkContext, loads and standardizes the data, runs the distributed training loop and the correctness checks. Imported by `bench.py`; also runnable on its own |
| `gradient.py` | The core math: sigmoid, loss, gradient |
| `numpy_baseline.py` | The non-Spark version: the same gradient descent on one machine |
| `analyze.py` | Computes speedup, efficiency and serial fraction from `results.csv` |
| `smoke_test.py` | Checks that PySpark and Java work (`local[8]`, a `treeReduce` over 1M numbers) |
| `.streamlit/config.toml` | Raises the upload limit to 1 GB (the 1M-row HIGGS CSV is about 700 MB) |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile`, `deploy/app.sh`, `deploy/deploy.sh`, `deploy/user-data.sh` | Host the app on EC2 (see [Optional: hosting the app on AWS](#optional-hosting-the-app-on-aws)) |

### Where things are saved

| What | Where |
|---|---|
| Uploaded file and its cleaned copy | `upload.csv` and `clean.csv` in a temporary folder per browser session (`<system temp>/spark-gd-*`). Deleted when the session ends; leftovers from earlier app runs are removed after an hour |
| App benchmark results | Only in the browser session. The app passes `--data` to `bench.py`, so nothing is written to the repo |
| Command-line HIGGS runs | Appended to `results.csv` (`--mode rows`) or `results_blocks.csv` (`--mode blocks`) |
| Unfiltered measurement log | `results_raw.csv`, kept to show what was filtered out of `results.csv` (see [Challenges](#challenges)) |

### Without the app (command line)

Run one benchmark with a core count and a repetition number:

```bash
python bench.py 4 0
```

With no options this trains on `higgs_1m.csv` for 50 iterations and appends to
`results.csv`. For any numeric CSV with a header and a 0/1 label (results are not
saved to a file):

```bash
python bench.py 4 0 --data mydata.csv --label target --iters 30 --mode blocks --check
```

`--check` adds the correctness checks. The last line of output is always a JSON
object.

Other entry points:

```bash
python train.py mydata.csv target
```

```bash
python numpy_baseline.py --data mydata.csv --label target
```

```bash
python analyze.py
```

`train.py` runs one 200-iteration training with full diagnostics (no arguments:
HIGGS). `numpy_baseline.py` is the non-Spark version (`--threads 1` limits it to
one core). `analyze.py` prints the speedup table from `results.csv`.

### Things to know

- Below about 100,000 rows, Spark's fixed per-iteration cost outweighs the math,
  so the speedup says little about scaling. The app shows a note in that case.
- The app runs each core count once, so its timings are demonstrations. The
  numbers in [Results](#results) come from `bench.py` with 3 interleaved
  repetitions per core count.
- The app sets `PYSPARK_PYTHON` to its own interpreter, and on macOS sets
  `JAVA_HOME` to Java 21 if it is installed and `JAVA_HOME` is not already set.
- Restart the app after editing `train.py` or `preprocess.py`; Streamlit does not
  reload imported modules.

### Optional: hosting the app on AWS

The same app can run on an EC2 machine so anyone with the link can use it. Spark
still runs in `local[N]`, on the machine's 8 cores instead of the laptop's. It
runs in Docker (`Dockerfile`), and `deploy/app.sh` does the whole setup with the
AWS CLI (configured with credentials and a region):

```bash
deploy/app.sh up        # first time: SSH key, firewall, instance, deploy (~8 min)
deploy/app.sh stop      # pause it: no compute charge while stopped
deploy/app.sh start     # resume; prints the new address (it changes on each start)
deploy/app.sh deploy    # push code changes to the running instance
deploy/app.sh allow-ip  # let in the network you are on now
deploy/app.sh status
deploy/app.sh down      # delete the instance and firewall
```

`up` prints the address (`http://<ip>`); the app works exactly as on the laptop.

- **Instance:** `c7a.2xlarge`, 8 vCPUs, 16 GB, about $0.41/hour while running.
  On AMD `c7a` each vCPU is a full physical core, so 1 → 8 cores is a fair scaling
  test (on Intel `c7i`, 8 vCPUs are 4 cores with hyperthreading).
- **Access:** only the IP you ran `up` from can open it. To share it, set a
  password and open it up with
  `APP_PASSWORD=choose-a-password deploy/app.sh deploy`, then
  `deploy/app.sh public`. The site is plain HTTP, so keep it restricted unless
  you need to share it.
- **Under the hood:** `up` creates a key pair (`~/.ssh/spark-gd-app.pem`) and a
  security group, launches the instance with `deploy/user-data.sh` (installs
  Docker), then `deploy/deploy.sh` copies the project with rsync, builds the image
  on the instance, and starts the container on port 80.
- **Stop the instance** when you're done; it is billed by the hour while running.

---

## Cluster run: 4 machines on AWS EMR

The local run shows that the non-Spark version wins when the data fits in one
machine's memory. The cluster run tests the case Spark is built for: data larger
than one machine's memory. By default it uses 10 copies of the full HIGGS dataset
(110M rows, 75 GB as CSV, 26 GB in memory) and compares three setups:

- **Spark** on an EMR cluster: 4 spot worker machines, 8 cores and 64 GB each,
  with the data cached in memory across them. It runs on all 4 machines and on 2.
- **Non-Spark on a 16 GB machine** (`m6id.xlarge`): the data doesn't fit in
  memory, so it re-reads it from its local NVMe disk every iteration.
- **Non-Spark on a 64 GB machine** (`r6id.2xlarge`): the data fits in memory, so
  it loads it once.

All three must reach the same loss after the same number of iterations.

### Set up and start

Requirements: the AWS CLI, configured with credentials and a region
(`aws configure`). Everything is driven from your laptop by `deploy/bigdata/run.sh`.

1. Check credentials and vCPU quotas (launches nothing):

   ```bash
   deploy/bigdata/run.sh check
   ```

2. Launch everything (about $3–8 in total). It returns once everything is
   launched; the machines then work on their own and shut themselves down:

   ```bash
   deploy/bigdata/run.sh up
   ```

3. Watch progress (which machines are running, which results are in):

   ```bash
   deploy/bigdata/run.sh status
   ```

4. When all results are in, download and compare them:

   ```bash
   deploy/bigdata/run.sh results
   ```

5. Stop anything still running (the S3 data is kept; `down` prints the command
   to delete the bucket):

   ```bash
   deploy/bigdata/run.sh down
   ```

Then open the app's **Recorded cluster runs (AWS)** tab to see the new run.

### What `up` launches

```
 Your laptop: deploy/bigdata/run.sh up
   │ 1. check credentials and vCPU quotas
   │ 2. create S3 bucket, upload code (+ your data, if local)
   │ 3. launch the three setups below; all share the bucket
   ▼
 ┌────────────────────────── S3 bucket: spark-gd-bench-<account>-<region> ─────────────────────────┐
 │  code/        higgs/HIGGS.csv (7.5 GB)      data/<run>/ (10 copies, 75 GB)      results/<run>/  │
 └──────┬────────────────────────────────────────────┬──────────────────────────────────┬──────────┘
        │                                            │                                  │
 ┌──────▼──────────────────────────────────┐ ┌───────▼───────────────────┐ ┌────────────▼─────────────┐
 │ EMR cluster (Spark on YARN)             │ │ non-Spark, 16 GB          │ │ non-Spark, 64 GB         │
 │                                         │ │ m6id.xlarge, 4 cores      │ │ r6id.2xlarge, 8 cores    │
 │  master: m6i.xlarge (on-demand)         │ │ 16 GB RAM + NVMe disk     │ │ 64 GB RAM + NVMe disk    │
 │   runs the driver, emr_bench.sh         │ │                           │ │                          │
 │                                         │ │ data doesn't fit in RAM:  │ │ data fits in RAM:        │
 │  4 workers: r5/r6i/r7i/r6a.2xlarge      │ │ re-reads 26 GB from disk  │ │ loads it once            │
 │   (spot), 8 cores + 64 GB each          │ │ every iteration           │ │                          │
 │   = 32 cores, data cached in memory     │ │                           │ │                          │
 └─────────────────────────────────────────┘ └───────────────────────────┘ └──────────────────────────┘
   all three write JSON results to S3, then terminate themselves
```

### What happens on the cluster, step by step

1. **Every node is set up** (`emr_bootstrap.sh`): before Spark starts, each
   machine gets a Python environment with NumPy, pandas and scikit-learn, so the
   workers can run `train.py`.
2. **The data is prepared** (`emr_bench.sh`, on the master): HIGGS (11M rows) is
   downloaded from UCI straight into S3, then copied 10 times inside S3 to make the
   110M-row dataset. A `READY` marker in S3 tells the non-Spark machines they can
   start.
3. **Spark runs** with `spark-submit` on YARN: one executor per worker machine,
   8 cores and 34 GB each. It uses the same `bench.py` and `train.py` as the local
   run, with `--cluster`: the data is read directly from S3 and split into 128
   partitions (4 per core), and the gradient runs in blocks mode for 20 iterations.
   It runs on 4 machines, then on 2, twice each, interleaved. Each result is
   uploaded to S3 as soon as it finishes.
4. **The non-Spark machines run at the same time** (`numpy_node.sh`): each one
   formats its local NVMe disk, waits for `READY`, streams the data from S3 into a
   standardized file on disk (`make_big_data.py`), clears the file cache so the
   first pass isn't flattered, runs `numpy_baseline.py`, and uploads its result.
   The 16 GB machine, which re-reads 26 GB per iteration, stops after 6 iterations.
5. **Everything shuts down on its own**: the cluster when its step ends, the
   single machines when they finish. Hard caps make sure of it even if something
   fails: 3 h for the cluster's step, 2.5 h for the single machines.

### Files

| File | Runs on | What it does |
|---|---|---|
| `deploy/bigdata/run.sh` | Your laptop | One command: `check`, `up`, `status`, `results`, `down` (plus `emr` and `numpy` to relaunch one part) |
| `deploy/bigdata/emr_bootstrap.sh` | Every cluster node | Installs the Python environment before Spark starts |
| `deploy/bigdata/emr_bench.sh` | Cluster master | Prepares the data in S3, runs Spark on 4 and 2 machines, uploads results |
| `deploy/bigdata/numpy_node.sh` | Each single machine | Builds the data on local disk, runs the non-Spark version, uploads, shuts down |
| `make_big_data.py` | Each single machine | Streams the CSV from S3 and writes K standardized copies as `.npy` files |
| `bench.py`, `train.py`, `gradient.py` | Cluster | The same code as the local run, with `--cluster` |
| `numpy_baseline.py` | Each single machine | The non-Spark version; `--stream` re-reads the data from disk every iteration |

### Where things are saved

| What | Where |
|---|---|
| Code | `s3://spark-gd-bench-<account>-<region>/code/` (copied to each machine at start) |
| HIGGS download | `s3://…/higgs/HIGGS.csv` (kept, so later runs skip the download) |
| Your data, if local | `s3://…/input/<run>/` |
| The copies | `s3://…/data/<run>/` |
| Results and logs | `s3://…/results/<run>/`: `spark-e<machines>-r<rep>.json`, `numpy-16gb.json`, `numpy-64gb.json`, `meta.json`, and `logs/` |
| Downloaded results | `deploy/bigdata/results/<run>/` in the repo, read by the app's **Recorded cluster runs** tab |

Everything is tagged `project=spark-gd-bench`. The S3 data stays until you delete
the bucket.

### Running it on your own data

Pass your data to `check` and `up`:

```bash
deploy/bigdata/run.sh up --data s3://my-bucket/sales/ --label churned
```

```bash
deploy/bigdata/run.sh up --data ~/data/big.csv --label target --copies 3 --nodes 2
```

`--data` takes a local file or folder (uploaded to S3 first), an `s3://` file, or
an `s3://` folder ending in `/`. Without `--label` the files are in the HIGGS
layout (no header, label in column 0); with it, each file has a header row. The
data must already be clean: all values numeric, no missing values, and a 0/1
label (the cluster does not run the app's file checker, which would load
everything on one machine). `--copies K` repeats the data K times, `--nodes N`
sets the number of worker machines, and `--name` sets the results folder.

---

## Results

### Local run (M2 MacBook Pro, 1M HIGGS rows)

Median warm seconds per iteration across 3 interleaved repetitions per core count
(iteration 0 dropped). Row-by-row gradient (`--mode rows`):

| Cores | s/iter | Speedup | Parallel efficiency |
|------:|-------:|--------:|--------------------:|
| 1 | 3.103 | 1.00× | 100.0% |
| 2 | 1.702 | 1.82× | 91.2% |
| 4 | 0.865 | 3.59× | 89.7% |

Vectorized gradient (`--mode blocks`):

| Cores | s/iter | Speedup |
|------:|-------:|--------:|
| 1 | 0.825 | 1.00× |
| 2 | 0.486 | 1.70× |
| 4 | 0.326 | 2.53× |
| 8 | 0.454 | noisy (0.30–0.63); the M2 has 4 performance + 4 efficiency cores |

Spark against the non-Spark version:

| Version | Best s/iter | vs. non-Spark |
|---|--:|--:|
| Non-Spark, one process | 0.030 | 1× |
| Spark, blocks, 4 cores | 0.326 | ~11× slower |
| Spark, rows, 4 cores | 0.889 | ~30× slower |

Correctness: the trained weights match scikit-learn's `LogisticRegression` at
**0.9915 cosine similarity** (lr = 0.5, 200 iterations), and blocks and rows mode
agree to ~1e-16. `python analyze.py` reproduces the first table.

### Cluster run (October 2026, us-east-1, 110M rows, 26.4 GB)

Raw JSON is in `deploy/bigdata/results/higgs10x/`.

| Setup | s/iter | vs. Spark on 4 machines | Final loss |
|---|--:|--:|--:|
| Spark, 4 × r5.2xlarge (32 cores, data cached) | **2.73** | 1× | 0.650322 |
| Spark, 2 × r5.2xlarge (16 cores, data cached) | 5.31 | 1.9× slower | 0.650322 |
| Non-Spark, 1 × r6id.2xlarge (64 GB, data in memory) | 5.53 | 2.0× slower | 0.650322 |
| Non-Spark, 1 × m6id.xlarge (16 GB, re-reads from disk) | 80.93 | 30× slower | 0.668668 (6 iterations) |

### What we learned

- **Spark only pays off when the data is too big for one machine.** On 1M rows
  the non-Spark version is 11× faster than Spark on the same laptop. On 110M rows
  the 16 GB machine spends every iteration re-reading 26 GB from disk (~0.33 GB/s),
  while the cluster keeps everything in memory and is 30× faster.
- **Spark has a fixed cost per iteration that small data can't hide.** The same
  job on a 20,000-row file still takes 0.26 s per iteration on 4 cores, so about
  80% of the 0.326 s on 1M rows is coordination, not math. Most of it is a ~50 ms
  stall per Python task in PySpark 4.1, while the JVM and the Python worker wait on
  each other before a task starts. A task that runs entirely in the JVM costs about
  5 ms. The non-Spark version reads the whole 230 MB dataset from memory in 30 ms,
  less than Spark's overhead for a single iteration.
- **Faster code scales worse.** Vectorizing made Spark 3.8× faster on 1 core but
  cut its 4-core speedup from 3.59× to 2.53×: with less math per task, the fixed
  overhead is a larger share of each iteration.
- **Scaling across machines is nearly linear.** Going from 2 to 4 machines is
  1.95× faster, and repetitions agreed to within 1%.
- **One big machine is the real competitor.** A single 64 GB machine running the
  non-Spark version matches 2 Spark machines. Spark only pulls ahead with more
  machines, and that machine had a newer CPU than the r5 cluster nodes.
- **Local scaling is sublinear, for expected reasons.** 3.59× on 4 cores (~90%
  efficiency) rather than 4×: each task pickles data across the JVM/Python
  boundary, and broadcasting the weights and collecting the gradient is serial
  driver work. The serial fraction estimated with Amdahl's law shifts with core
  count (≈9.7% at 2 cores, ≈3.8% at 4), which points to fixed overhead rather than
  a constant serial share of the computation.
- **Every setup gets the same answer.** After 20 iterations the loss agrees to 13
  decimal places across Spark on 2 and 4 machines and the 64 GB machine. The 16 GB
  machine ran 6 iterations, and its loss matches the 64 GB run's 6th iteration.

---

## Challenges

**A partition count that was silently ignored.** `sc.textFile(path, n)` treats `n`
as a *minimum*: the loader produced 22 partitions instead of the intended 8, which
added scheduling and serialization overhead and roughly **tripled** the time per
iteration. An explicit `.repartition(n)` fixed it. It was the single largest
performance correction in the project.

**Thermal throttling on the laptop.** On an M2 MacBook Pro, two identical
`local[4]` runs varied by ~24%. The benchmark controls for it: core counts are
interleaved (1 → 2 → 4 per repetition) so drift doesn't line up with one condition,
the median is used instead of the mean, iteration 0 (JVM warmup, ~2.5 s) is
dropped, and runs are separated by cooldowns with background apps closed. The raw
log, `results_raw.csv`, keeps three contaminated rows (two cold-start `local[4]`
warmups and one stray 130-iteration run) that are excluded from `results.csv`, so
the filtering is documented rather than hidden.

**Spark's fixed per-task overhead.** Vectorizing the gradient made the math fast
enough that PySpark's ~50 ms per-task stall dominated. Switching the JVM-to-Python
connection from TCP to a Unix domain socket didn't remove it, which is why Spark
can't beat the non-Spark version on data that fits in memory.

**Resizing Spark inside one app.** `local[N]` can't change N once the JVM is
running, so the app runs each core count in its own `bench.py` process with a fresh
`SparkContext`. NumPy is limited to one thread per Python worker, so N workers use
N cores rather than each grabbing every core.

**A fair single-machine comparison on the cluster run.** The 16 GB machine can't
hold the data, so the non-Spark version needed a streaming mode (`--stream`) that
reads from local NVMe in chunks. The file cache is cleared before the first pass so
it isn't flattered, and `make_big_data.py` streams the CSV from S3 twice
(statistics, then writing) so neither memory nor disk ever holds the CSV itself.

**Breaking the AWS CLI with pip.** Installing pandas into the system Python on EMR
and EC2 replaced the `python-dateutil` that the `aws` command depends on, which
broke every S3 call. The benchmark's Python packages now go in a separate virtual
environment on every machine.

**Keeping the cloud bill bounded.** Spot workers keep the cluster cheap, `run.sh
check` verifies vCPU quotas before launching anything, and every machine terminates
itself when its work is done, with hard time limits in case something hangs. Big
copies stay inside S3, so the 75 GB dataset never passes through a machine.

**Diverging reference weights.** On linearly separable data, scikit-learn's
unregularized weights grow without bound, so cosine similarity with them is low
even when Spark's weights are correct. The app compares against the non-Spark
version's weights for an exact check and explains when a low scikit-learn cosine
is expected.
