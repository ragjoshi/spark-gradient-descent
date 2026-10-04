# Distributed Logistic Regression on Spark

A from-scratch implementation of batch gradient descent for logistic regression on
Apache Spark (PySpark, RDD API), benchmarked for strong scaling on a 1M-row physics
dataset. No MLlib — the distributed gradient computation, aggregation, and training
loop are all hand-written to expose the actual mechanics of at-scale optimization.

The project answers one practical question: **when is Spark worth it?** It runs the
same training three ways, each one a step up in scale:

| | Where it runs | Spark setup | Data | What it shows |
|---|---|---|---|---|
| [1. Local](#walkthrough-1-the-app-on-your-laptop) | Your laptop | `local[N]`: 1 machine, N cores | Any CSV you upload (tested on 1M rows) | How Spark scales with cores, and that NumPy beats it on small data |
| [2. App on AWS](#walkthrough-2-the-app-on-aws-one-ec2-machine) | One EC2 machine (8 cores) | `local[N]` inside Docker | Same as local | The same app, reachable from any browser |
| [3. Cluster on AWS](#walkthrough-3-the-4-machine-spark-cluster-on-aws-emr) | EMR cluster (4 worker machines) vs. 2 single machines | YARN, 32 cores across 4 machines | 110M rows, 26 GB | Spark winning once the data outgrows one machine |

**The short answer:** on 1M rows, plain NumPy on one machine is 11× faster than
Spark. On 110M rows that don't fit in one machine's memory, Spark on 4 machines is
30× faster than NumPy on a 16 GB machine. Spark only pays off when the data is too
big for one machine.

## Contents

- [Tech stack](#tech-stack)
- [How the training works](#how-the-training-works) (the same in all three setups)
- [Walkthrough 1: the app on your laptop](#walkthrough-1-the-app-on-your-laptop)
- [Walkthrough 2: the app on AWS (one EC2 machine)](#walkthrough-2-the-app-on-aws-one-ec2-machine)
- [Walkthrough 3: the 4-machine Spark cluster on AWS (EMR)](#walkthrough-3-the-4-machine-spark-cluster-on-aws-emr)
- [Deep dive: the laptop benchmark](#deep-dive-the-laptop-benchmark) (results,
  engineering decisions, methodology, scaling analysis, Spark vs. NumPy)
- [Repository structure](#repository-structure)

## Tech stack

| Layer | Technology | What it does here |
|---|---|---|
| Distributed compute | **Apache Spark 4.1 (PySpark, RDD API)** on **Java 21** | Splits the data into partitions and computes the gradient in parallel. No MLlib: the math is hand-written. |
| Math | **NumPy** | The gradient on each block of rows, and the single-machine baseline Spark is compared against |
| Correctness check | **scikit-learn** | Reference logistic regression the trained weights are compared with |
| Data checks | **pandas** | Reads and validates uploaded CSVs (`preprocess.py`) |
| Web app | **Streamlit** + Altair charts | Upload a CSV, run the benchmark, see charts and a recommendation |
| Packaging | **Docker** (Python 3.12 + Java 21) | Same app environment on any machine |
| App hosting | **AWS EC2** (`c7a.2xlarge`, Amazon Linux 2023) | One 8-core machine running the app container |
| Cluster | **AWS EMR 7** (Spark on **YARN**), **Spot** instances | 1 master + 4 worker machines (8 cores, 64 GB each) |
| Storage | **AWS S3** | Code, the 75 GB dataset, and results, shared by every machine |
| Automation | **Bash + AWS CLI** | `deploy/app.sh` and `deploy/bigdata/run.sh`: everything launched with one command, no console clicking |

Python dependencies are pinned in `requirements.txt`.

## How the training works

The model is logistic regression trained by **batch gradient descent**: every
iteration looks at all the data once, computes the gradient (which direction to move
the weights), and takes one step. That full pass over the data is the expensive part,
and it is what Spark parallelizes.

One iteration, in every setup:

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

What the code does (`train.py`, `gradient.py`):

- Loads the data (the HIGGS dataset by default: 28 features + binary label, or any
  numeric CSV) as a Spark RDD of `(label, feature vector)` rows, standardizes the
  features, and caches it in memory.
- Runs batch gradient descent for logistic regression, where each iteration computes
  the full-dataset gradient in a single distributed pass.
- Uses `treeAggregate` to accumulate the gradient, loss, and record count together in
  one scan, combining partition-level **sums** (not means) for correctness across
  unevenly sized partitions.
- Broadcasts the current weight vector each iteration and destroys it afterward, to
  prevent driver-side memory accumulation over a long training run.
- Has two ways to compute each partition's gradient: `--mode rows` (the original:
  one Python row at a time) and `--mode blocks` (packs each partition into NumPy
  matrices once, so the gradient is one `X @ w` per block). Both give the same
  weights; blocks is 3.8× faster on 1 core.

The only thing that changes between the three setups is **where the driver and
workers live**:

| Setup | Driver | Workers | Data comes from |
|---|---|---|---|
| Local | A process on your laptop | N threads on your laptop (`local[N]`) | The uploaded CSV |
| App on AWS | A process in the Docker container on EC2 | N threads on that EC2 machine | The uploaded CSV |
| Cluster | The EMR master machine | 4 executors, one per worker machine, 8 cores each (YARN) | S3 |

Every run checks itself: the same gradient descent is re-run in plain NumPy and the
results must match (same weights, or the same final loss), and the weights are
compared with scikit-learn's.

---

## Walkthrough 1: the app on your laptop

### Set up and start

Requirements: Python 3.12 and a JDK supported by Spark 4 (17 or 21; developed against
OpenJDK 21, ARM64). Python dependencies are pinned in `requirements.txt`: PySpark,
NumPy, scikit-learn (validation baseline), pandas (CSV preprocessing), and Streamlit
(the app).

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501). The page has two tabs:
**Run on this machine** (a live benchmark on your data) and **Recorded cluster runs
(AWS)** (the results of walkthrough 3, read from files in the repo, so it works
offline).

### Using it

1. Upload a CSV with a header row, numeric feature columns, and a label column
   containing only 0 and 1. Comma, semicolon, and tab separators are all accepted.
2. Pick the label column. The app validates the file immediately: non-numeric columns,
   bad labels, and similar problems are listed in plain language, rows with missing
   values are dropped (and counted), and columns that look like row IDs are flagged.
3. Choose the maximum core count, the number of iterations, and the gradient code
   (vectorized blocks, the default, or row by row), then click **Run benchmark**.

### What happens when you click Run benchmark

```
 Browser ──upload──▶ Streamlit (app.py)
                        │ 1. preprocess.clean_csv: validate, drop bad rows ─▶ clean.csv
                        │
                        │ 2. for N in 1, 2, 4, 8 … (one at a time):
                        ├──▶ python bench.py N … ─▶ new JVM + SparkContext local[N]
                        │        load ─▶ standardize ─▶ cache ─▶ train (timed) ─▶ JSON
                        │
                        │ 3. python numpy_baseline.py …  (same training, no Spark)
                        │
                        ▼ 4. charts, Spark-vs-NumPy recommendation, correctness
 Browser ◀──────────────┘
```

1. **The upload is saved and checked.** The file goes to a temporary folder for your
   browser session (deleted when the session ends). `preprocess.py` reads it with
   pandas, detects the separator, checks every column is numeric and the label is
   0/1, drops rows with missing values, and writes a clean copy. This runs once per
   file and label, not on every click.
2. **Spark runs once per core count.** The app runs `bench.py` once per core count
   (1, 2, 4, 8 by default, capped at the machine's core count and the 8 fixed
   partitions), each in its own process so every run gets a fresh `SparkContext`
   (`local[N]` cannot be resized inside one JVM). Inside each run:
   - Spark starts a JVM with N worker threads, each feeding a Python worker process.
     NumPy is limited to one thread per worker, so N workers really use N cores.
   - The CSV is split into 8 partitions, the features are standardized (one
     `treeAggregate` pass for means and standard deviations), and the data is cached.
   - The timed training loop runs (see [How the training works](#how-the-training-works)).
     Each iteration is timed; iteration 0 (JVM warmup) is dropped and the median of
     the rest is the result.
   - On the last core count, correctness checks run after the timed loop, so they
     never affect the timing.
   - The run prints one JSON line, which the app reads.
3. **NumPy runs the same training** in one process with no Spark
   (`numpy_baseline.py`): same data, starting point, learning rate and iterations.
4. **The app shows the results.** Only one benchmark runs at a time across all users
   (a lock), so runs can't distort each other's timings.

### What you see

- **Speedup vs 1 core**, actual against ideal linear speedup
- **Seconds per iteration** and parallel efficiency for each core count
- **Spark vs. plain NumPy**: both times on one chart, and a recommendation (use Spark
  or use plain NumPy) based on which was faster and whether the data fits in this
  machine's memory
- **Recommended cores**: the smallest core count whose speedup is within 10% of the
  best observed, labeled as a quick estimate for this dataset on this machine. It
  compares Spark runs with each other only, so it is not general Spark guidance and
  does not say whether Spark beats a single machine without Spark
- **Correctness**: whether the Spark weights match the same gradient descent run in
  plain NumPy on one machine (datasets up to 200,000 rows), and cosine similarity
  against scikit-learn. On linearly separable data scikit-learn's unregularized
  weights diverge, so the app says when a low cosine is expected.

On the 1M-row HIGGS file the typical outcome is: Spark speeds up with more cores, but
plain NumPy is still about 11× faster, so the app recommends NumPy. See
[Spark vs. plain NumPy](#spark-vs-plain-numpy) for why.

### Things to know

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

### Command-line tools (no app)

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

---

## Walkthrough 2: the app on AWS (one EC2 machine)

The same app, hosted so anyone with the link can use it. Nothing about the Spark
code changes: Spark still uses `local[N]`, now on the EC2 machine's 8 cores instead
of the laptop's. The difference is a fair, quiet machine: 8 identical physical cores,
no thermal throttling, no other apps.

### Set up and start

Requirements: the AWS CLI, configured with credentials and a region. The app runs in
Docker (`Dockerfile`: Python 3.12 + Java 21). `deploy/app.sh` does the whole setup
with the AWS CLI (no console clicking):

```bash
deploy/app.sh up        # first time: SSH key, firewall, instance, deploy (~8 min)
deploy/app.sh stop      # pause it: no compute charge while stopped
deploy/app.sh start     # resume; prints the new address (it changes on each start)
deploy/app.sh deploy    # push code changes to the running instance
deploy/app.sh allow-ip  # let in the network you are on now
deploy/app.sh status
deploy/app.sh down      # delete the instance and firewall
```

`up` prints the address (`http://<ip>`). Open it and use the app exactly as in
walkthrough 1.

### What `up` does, step by step

```
 Your laptop                                   AWS (your default region)
 ───────────                                   ─────────────────────────
 deploy/app.sh up
   1. create SSH key pair ───────────────────▶ key pair "spark-gd-app"
   2. create firewall, allow only your IP ───▶ security group: ports 22 + 80
   3. launch instance ───────────────────────▶ EC2 c7a.2xlarge, Amazon Linux 2023
                                                 └─ user data installs Docker
   4. deploy/deploy.sh:
        rsync the project ───────────────────▶ ~/spark-app/
        ssh: docker build, docker run ───────▶ container "spark-app"
                                                 (Python 3.12, Java 21, Streamlit,
                                                  restarts on its own after reboots)
   5. wait for the health check, print URL
 Browser ── http://<ip> (port 80) ───────────▶ container port 8501 ─▶ app.py
```

- **Instance:** `c7a.2xlarge` (8 vCPUs, 16 GB, about $0.41/hour while running,
  about $2.40/month for the 30 GB disk while stopped). On AMD `c7a`, each vCPU is
  a full physical core, so 1→8 cores is a fair scaling test. On Intel types like
  `c7i`, 8 vCPUs are only 4 physical cores with hyperthreading.
- **Access:** only the IP address you ran `up` from can open the page (and SSH
  in). On a different network, run `allow-ip`. To share the page with anyone,
  set a password and open it up:
  ```bash
  APP_PASSWORD=choose-a-password deploy/app.sh deploy
  deploy/app.sh public
  ```
- **Under the hood:** `up` creates the key pair (`~/.ssh/spark-gd-app.pem`) and a
  security group, launches the instance with `deploy/user-data.sh` (installs
  Docker), then runs `deploy/deploy.sh`, which copies the project with rsync,
  builds the image on the instance, and starts the container on port 80, set to
  restart if it stops (including after `stop` / `start`). The password is sent over
  SSH's input, not the command line, so it never appears in the remote process list.

### Things to know

- Only one benchmark runs at a time across all users, so concurrent runs can't
  distort each other's timings. Others see a "wait" message.
- The site is plain HTTP, so the password and uploads are not encrypted. Keep
  access restricted to your IPs unless you need to share it.
- Uploads stay in the container's `/tmp`; the app deletes those of ended sessions.
- **Stop the instance** when you're done. It is billed by the hour while running.

---

## Walkthrough 3: the 4-machine Spark cluster on AWS (EMR)

Walkthroughs 1 and 2 show that on data that fits in one machine's memory, plain NumPy
wins. `deploy/bigdata/` tests the case Spark is built for: data larger than one
machine's memory. By default it uses 10 copies of the full HIGGS dataset
(110M rows, 75 GB as CSV, 26 GB as float64); `--data` runs it on any CSV
instead. It compares:

- **Spark** on an EMR cluster (4 spot nodes by default, 8 vCPUs and 64 GB
  each), data cached in memory across the nodes, run on all nodes and on half.
- **NumPy on a 16 GB machine** (`m6id.xlarge`), which must re-read the data
  from its NVMe disk on every iteration (`numpy_baseline.py --stream`).
- **NumPy on a 64 GB machine** (`r6id.2xlarge`), which holds it all in RAM,
  or reports that it does not fit.

All three report the loss after the same number of iterations, which must
match.

### Set up and start

Requirements: the AWS CLI, configured with credentials and a region.

```bash
deploy/bigdata/run.sh check     # credentials and vCPU quotas; launches nothing
deploy/bigdata/run.sh up        # launch (about $3-8 in total)
deploy/bigdata/run.sh status
deploy/bigdata/run.sh results   # download and compare
deploy/bigdata/run.sh down      # stop anything still running
```

`up` returns as soon as everything is launched; the machines then work on their own
and shut themselves down. `status` shows progress, and `results` downloads the
result files into the repo.

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
 │ EMR cluster (Spark on YARN)             │ │ numpy-16gb                │ │ numpy-64gb               │
 │                                         │ │ m6id.xlarge, 4 cores      │ │ r6id.2xlarge, 8 cores    │
 │  master: m6i.xlarge (on-demand)         │ │ 16 GB RAM + NVMe disk     │ │ 64 GB RAM + NVMe disk    │
 │   runs the driver, emr_bench.sh         │ │                           │ │                          │
 │                                         │ │ data doesn't fit in RAM:  │ │ data fits in RAM:        │
 │  4 workers: r5/r6i/r7i/r6a.2xlarge      │ │ re-reads 26 GB from disk  │ │ loads once, plain NumPy  │
 │   (spot), 8 cores + 64 GB each          │ │ every iteration           │ │                          │
 │   = 32 cores, data cached in memory     │ │                           │ │                          │
 └─────────────────────────────────────────┘ └───────────────────────────┘ └──────────────────────────┘
   all three write JSON results to S3, then terminate themselves
```

### What happens on the cluster, step by step

1. **Setup on every node** (`emr_bootstrap.sh`): before Spark starts, each machine
   gets a Python environment with NumPy, pandas and scikit-learn, so the workers can
   run `train.py`.
2. **Data** (`emr_bench.sh`, on the master): HIGGS (11M rows) is downloaded from UCI
   straight into S3, then copied 10 times inside S3 (server-side copies, nothing passes
   through the machine) to make the 110M-row dataset. A `READY` marker in S3 tells the
   NumPy machines they can start.
3. **Spark runs** with `spark-submit` on YARN: one executor per worker machine, 8
   cores each, 34 GB of memory each. The same `bench.py` and `train.py` as the laptop,
   with `--cluster`: data is read directly from S3 and split into 128 partitions (4
   per core). Each run trains for 20 iterations in blocks mode. It runs on all 4
   machines, then on 2, twice each, interleaved. Every result is uploaded to S3 as
   soon as it finishes.
4. **NumPy runs in parallel** on the two single machines (`numpy_node.sh`): each one
   formats its local NVMe disk, waits for `READY`, streams the data from S3 into a
   standardized NumPy file on disk (`make_big_data.py`), clears the file cache so the
   first pass is not flattered, runs `numpy_baseline.py`, and uploads its result.
5. **Everything shuts down on its own.** The cluster terminates when its step ends;
   the NumPy machines terminate when they finish.
6. **`run.sh results`** downloads every result into `deploy/bigdata/results/<run>/`.
   The app's **Recorded cluster runs** tab reads those files and shows the chart,
   speedups, and a recommendation: plain NumPy, one big machine, or Spark.

Every machine terminates itself when its benchmark finishes (hard caps: 3 h
for the cluster's step, 2.5 h for the NumPy instances). The data stays in S3
until you delete the bucket; `down` prints the command.

### Running it on your own data

To benchmark your own data, pass it to `check` and `up`:

```bash
deploy/bigdata/run.sh up --data s3://my-bucket/sales/ --label churned
deploy/bigdata/run.sh up --data ~/data/big.csv --label target --copies 3 --nodes 2
```

`--data` takes a local file or folder (uploaded to S3 first), an `s3://` file,
or an `s3://` folder ending in `/`. Without `--label` the files are in the
HIGGS layout (no header, label in column 0); with it, each file has a header
row. The data must already be clean: all values numeric, no missing values,
and a 0/1 label (the cluster does not run the app's file checker, which loads
everything on one machine). `--copies K` repeats the data K times, `--nodes N`
sets the cluster size, and `--name` sets the results folder. Each run's
results go to `deploy/bigdata/results/<name>/`, and the app's
**Recorded cluster runs** tab shows every run there, with a recommendation:
plain NumPy, one big machine, or Spark.

### Results (October 2026, us-east-1)

Same data in every row: 110M rows, 26.4 GB as float64. Raw JSON is in
`deploy/bigdata/results/higgs10x/`.

| Setup | s/iter | vs. Spark on 4 nodes | Final loss |
|---|--:|--:|--:|
| Spark, 4 × r5.2xlarge (32 cores, data cached) | **2.73** | 1× | 0.650322 |
| Spark, 2 × r5.2xlarge (16 cores, data cached) | 5.31 | 1.9× slower | 0.650322 |
| NumPy, 1 × r6id.2xlarge (64 GB, data in RAM) | 5.53 | 2.0× slower | 0.650322 |
| NumPy, 1 × m6id.xlarge (16 GB, re-reads from NVMe) | 80.93 | 30× slower | 0.668668 (6 iters) |

- **Spark wins once the data no longer fits on one machine.** The 16 GB machine
  spends every iteration re-reading 26 GB from disk (~0.33 GB/s); the cluster
  keeps it all in memory and is 30× faster.
- **Scaling across machines is nearly linear:** 2 → 4 nodes is 1.95× faster, and
  both repetitions agreed to within 1%.
- **One big machine is the real competitor.** A single 64 GB machine running NumPy
  matches 2 Spark nodes. Spark only pulls ahead with more nodes, and that machine
  had a newer CPU than the r5 cluster nodes.
- **Same answer everywhere:** after 20 iterations the loss agrees to 13 decimal
  places (the 16 GB run did 6 iterations; it matches the 64 GB run's 6th, 0.668668).

Combined with the laptop numbers below: on 1M rows NumPy is 11× faster than
Spark; on 110M rows that don't fit in one machine's memory, Spark is 30× faster.

---

## Deep dive: the laptop benchmark

### Results

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

### Key engineering decisions

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

### Benchmark methodology

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

### Scaling analysis

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

### Spark vs. plain NumPy

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
machine can hold in memory, which is exactly what
[walkthrough 3](#walkthrough-3-the-4-machine-spark-cluster-on-aws-emr) measures.

---

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
├── deploy/          # EC2 app deploy (walkthrough 2)
│   ├── app.sh       #   one command: create, start, stop, deploy, delete the app machine
│   ├── deploy.sh    #   copy the code to the machine, build and run the container
│   ├── user-data.sh #   first-boot script: installs Docker
│   └── bigdata/     # Cluster benchmark (walkthrough 3)
│       ├── run.sh          # one command: check quotas, launch, status, results, down
│       ├── emr_bootstrap.sh # runs on every cluster node: Python environment
│       ├── emr_bench.sh    # runs on the master: prepare data, run Spark on 4 and 2 nodes
│       ├── numpy_node.sh   # runs on each NumPy machine: build data, run, upload, shut down
│       └── results/        # downloaded results, one folder per run (read by the app)
├── make_big_data.py # Builds K standardized copies of a CSV as .npy for numpy_baseline.py
├── results.csv      # Clean benchmark results (9 runs: 1/2/4 cores × 3 reps)
├── results_raw.csv  # Unfiltered measurement log (includes caught contamination)
└── higgs_1m.csv     # 1M-row HIGGS subset (28 features + label) — not committed
```

## Notes

All metrics in this README are personally reproducible via the committed scripts and
`results.csv`. Speedup and efficiency come from `analyze.py`; the correctness figure
comes from the scikit-learn comparison in the training validation step.
