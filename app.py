# app.py: Streamlit front end for the strong-scaling benchmark.
#
# Run with:  streamlit run app.py
#
# Each core count runs bench.py in its own subprocess, so every run gets a
# fresh SparkContext (local[N] cannot be resized inside one JVM).
import json
import os
import subprocess
import sys
import tempfile
import threading

import pandas as pd
import streamlit as st

from preprocess import DataError, clean_csv, read_header

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench.py")
MAX_CORES = min(os.cpu_count() or 1, 8)   # partitions are fixed at 8
SMALL_DATA_ROWS = 100_000
LABEL_GUESSES = ["label", "target", "y", "class", "outcome"]


@st.cache_resource
def run_lock():
    # One benchmark at a time: overlapping runs would compete for the same
    # cores and make every timing meaningless.
    return threading.Lock()


@st.cache_resource
def spark_env():
    env = dict(os.environ)
    # Spark's Python workers must use this interpreter (the venv), not
    # whatever "python3" is first on PATH.
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if "JAVA_HOME" not in env and sys.platform == "darwin":
        # Spark 4 supports Java 17 and 21; prefer 21 if it is installed.
        r = subprocess.run(["/usr/libexec/java_home", "-v", "21"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            env["JAVA_HOME"] = r.stdout.strip()
    return env


def core_counts(max_cores):
    counts, c = [], 1
    while c < max_cores:
        counts.append(c)
        c *= 2
    return counts + [max_cores]


def session_dir():
    if "tmpdir" not in st.session_state:
        st.session_state.tmpdir = tempfile.mkdtemp(prefix="spark-gd-")
    return st.session_state.tmpdir


def save_upload(uploaded):
    """Write the upload to disk once per file, not on every rerun."""
    path = os.path.join(session_dir(), "upload.csv")
    if st.session_state.get("upload_id") != uploaded.file_id:
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.upload_id = uploaded.file_id
        st.session_state.pop("results", None)
    return path


def run_bench(cores, path, label, iters, check):
    cmd = [sys.executable, BENCH, str(cores), "0", "--data", path,
           "--label", label, "--iters", str(iters)]
    if check:
        cmd.append("--check")
    p = subprocess.run(cmd, cwd=HERE, env=spark_env(),
                       capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    try:
        out = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        out = {"error": "The benchmark process exited without a result.",
               "kind": "spark"}
    out["stderr"] = p.stderr
    return out


def show_error(result):
    st.error(result["error"])
    if result.get("kind") == "spark" and result.get("stderr"):
        with st.expander("Technical details"):
            st.code(result["stderr"][-4000:])


def show_results(res):
    runs = res["runs"]
    t1 = runs[0]["sec_per_iter_warm"]
    table = pd.DataFrame({
        "Cores": [r["cores"] for r in runs],
        "Seconds per iteration": [r["sec_per_iter_warm"] for r in runs],
        "Speedup": [t1 / r["sec_per_iter_warm"] for r in runs],
    })
    table["Ideal speedup"] = table["Cores"].astype(float)
    table["Parallel efficiency"] = table["Speedup"] / table["Cores"]

    st.subheader("Scaling")
    st.caption(
        "Every run uses Spark local[N] on this one machine: N worker threads "
        "sharing one computer, not a multi-machine cluster. Times are the "
        "median per-iteration time with iteration 0 (JVM warmup) dropped, "
        "from a single run per core count.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Speedup vs 1 core**")
        chart = table.set_index("Cores")[["Speedup", "Ideal speedup"]]
        chart.columns = ["Actual", "Ideal (linear)"]
        st.line_chart(chart, x_label="Cores", y_label="Speedup (x)")
    with right:
        st.markdown("**Seconds per iteration**")
        st.bar_chart(table.set_index("Cores")["Seconds per iteration"],
                     x_label="Cores", y_label="Seconds")

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Seconds per iteration": st.column_config.NumberColumn(format="%.4f"),
            "Speedup": st.column_config.NumberColumn(format="%.2fx"),
            "Ideal speedup": st.column_config.NumberColumn(format="%.0fx"),
            "Parallel efficiency": st.column_config.NumberColumn(format="percent"),
        },
    )

    if res["rows"] < SMALL_DATA_ROWS:
        st.info(
            f"This dataset has {res['rows']:,} rows. Below about "
            f"{SMALL_DATA_ROWS:,} rows, most of each iteration is Spark's "
            "fixed overhead (scheduling tasks, moving data between the JVM "
            "and Python, broadcasting weights), not gradient math. Some of "
            "that overhead runs in parallel too, so a speedup can still "
            "appear, but it says little about how the training math scales. "
            "Use a larger dataset for a meaningful scaling measurement.")

    check = res["check"]
    st.subheader("Correctness")
    a, b = st.columns(2)
    with a:
        if check.get("ref_checked"):
            st.metric("Matches single-machine NumPy",
                      "Yes" if check["ref_match"] else "No",
                      help="The same gradient descent (same data, start, "
                           "learning rate and iterations) rerun with plain "
                           "NumPy on one machine.")
            st.caption(f"Largest weight difference: {check['ref_max_abs_diff']:.1e}. "
                       "Differences near 1e-16 are floating-point rounding "
                       "from adding partition sums in a different order.")
        else:
            st.metric("Matches single-machine NumPy", "Skipped")
            st.caption(f"Skipped: {check.get('ref_reason', 'not run')}.")
    with b:
        st.metric("Cosine similarity vs scikit-learn", f"{check['cosine']:.4f}",
                  help="Direction agreement between our weights and "
                       "scikit-learn's unregularized LogisticRegression. "
                       "1.0 means the same direction.")
        st.caption(f"Accuracy: ours {check['acc_mine']:.1%}, "
                   f"scikit-learn {check['acc_sklearn']:.1%} "
                   f"(on {check['sample_rows']:,} rows).")
    if check["sklearn_separable"]:
        st.warning(
            "scikit-learn classified every row correctly, which means the "
            "classes can be split perfectly by a straight line. On data like "
            "this its unregularized weights grow without limit, so a low "
            "cosine here does not indicate a bug. The NumPy comparison is "
            "the reliable check.")
    elif check["cosine"] < 0.95:
        st.caption(
            "A cosine below about 0.95 usually means training has not "
            "converged yet. Try more iterations.")


st.set_page_config(page_title="Spark Scaling Lab", layout="wide")
st.title("Spark Scaling Lab")
st.write(
    "Upload a CSV and see how distributed logistic regression speeds up as "
    "Spark gets more cores. Training is batch gradient descent written from "
    "scratch on Spark's RDD API.")

uploaded = st.file_uploader(
    "CSV with a header row, numeric columns, and a 0/1 label column",
    type=["csv", "tsv", "txt"])
if uploaded is None:
    st.stop()

raw_path = save_upload(uploaded)
try:
    columns = read_header(raw_path)
except DataError as e:
    st.error(str(e))
    st.stop()

guess = next((c for c in columns if str(c).strip().lower() in LABEL_GUESSES),
             columns[-1])
label = st.selectbox("Label column (what to predict)", columns,
                     index=columns.index(guess))

clean_path = os.path.join(session_dir(), "clean.csv")
# Cleaning a large file takes seconds (about 20 s for 1M rows), and Streamlit
# reruns this script on every widget change, so clean once per (file, label).
clean_key = (st.session_state.upload_id, label)
if st.session_state.get("clean_key") != clean_key:
    with st.spinner("Checking the file..."):
        try:
            st.session_state.clean_result = clean_csv(raw_path, label, clean_path)
        except DataError as e:
            st.session_state.clean_result = DataError(str(e))
    st.session_state.clean_key = clean_key
summary = st.session_state.clean_result
if isinstance(summary, DataError):
    st.error(str(summary))
    st.stop()

notes = [f"{summary['rows']:,} rows", f"{summary['n_features']} features"]
if summary["rows_dropped"]:
    notes.append(f"{summary['rows_dropped']:,} rows with missing values dropped")
if summary["ignored_columns"]:
    notes.append("ignored index column(s): " + ", ".join(summary["ignored_columns"]))
st.success("Ready: " + "; ".join(notes) + ".")
if summary["likely_id_columns"]:
    st.warning(
        "These columns look like row numbers (whole numbers counting up with "
        "no gaps): " + ", ".join(summary["likely_id_columns"]) + ". They are "
        "being used as features, which teaches the model nothing real. "
        "Remove them from the file if they are IDs.")

c1, c2 = st.columns(2)
with c1:
    max_cores = st.select_slider("Max cores", options=list(range(1, MAX_CORES + 1)),
                                 value=min(4, MAX_CORES))
with c2:
    iters = st.number_input("Iterations", min_value=5, max_value=500, value=30,
                            help="Gradient descent steps per run. The first "
                                 "one is excluded from timing (JVM warmup).")

counts = core_counts(max_cores)
st.caption(f"Will run with {', '.join(map(str, counts))} core(s), one Spark "
           "process each.")

if st.button("Run benchmark", type="primary"):
    lock = run_lock()
    if not lock.acquire(blocking=False):
        st.warning("Another benchmark is running. Wait for it to finish, "
                   "since overlapping runs would distort both timings.")
        st.stop()
    try:
        st.session_state.pop("results", None)
        runs, check = [], None
        progress = st.progress(0.0)
        for i, cores in enumerate(counts):
            progress.progress(i / len(counts),
                              text=f"Running with {cores} core(s) "
                                   f"({i + 1} of {len(counts)})...")
            # Correctness checks run once, on the last run, after its timed loop.
            last = i == len(counts) - 1
            result = run_bench(cores, clean_path, label, iters, check=last)
            if "error" in result:
                progress.empty()
                show_error(result)
                st.stop()
            runs.append(result)
            if last:
                check = result
        progress.empty()
        st.session_state.results = {"runs": runs, "check": check,
                                    "rows": summary["rows"],
                                    "label": label, "iters": iters}
    finally:
        lock.release()

res = st.session_state.get("results")
if res and res["label"] == label:
    show_results(res)
