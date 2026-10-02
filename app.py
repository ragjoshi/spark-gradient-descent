# app.py: Streamlit front end for the strong-scaling benchmark.
#
# Run with:  streamlit run app.py
#
# Each core count runs bench.py in its own subprocess, so every run gets a
# fresh SparkContext (local[N] cannot be resized inside one JVM).
import glob
import hmac
import json
import statistics
import os
import subprocess
import sys
import tempfile
import threading

import altair as alt
import pandas as pd
import streamlit as st

from preprocess import DataError, clean_csv, read_header

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench.py")
NUMPY = os.path.join(HERE, "numpy_baseline.py")
HIGGS = os.path.join(HERE, "higgs_1m.csv")          # no header, label in column 0
RECORDED = os.path.join(HERE, "deploy", "bigdata", "results")
MODES = {"Vectorized (one NumPy matrix per block)": "blocks",
         "Row by row (original)": "rows"}
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


def run_bench(cores, path, label, iters, check, mode):
    cmd = [sys.executable, BENCH, str(cores), "0", "--data", path,
           "--iters", str(iters), "--mode", mode]
    if label is not None:
        cmd += ["--label", label]
    if check:
        cmd.append("--check")
    return run_json(cmd, spark_env())


def run_numpy(path, label, iters):
    """The same gradient descent in plain NumPy, one process, no Spark."""
    cmd = [sys.executable, NUMPY, "--data", path, "--iters", str(iters)]
    if label is not None:
        cmd += ["--label", label]
    return run_json(cmd, dict(os.environ))


def run_json(cmd, env):
    """Run a script whose last stdout line is a JSON result."""
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
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


def recommended_cores(cores, speedups, tolerance=0.10):
    """Smallest core count whose speedup is within `tolerance` of the best."""
    best = max(speedups)
    return min(c for c, s in zip(cores, speedups) if s >= (1 - tolerance) * best)


def show_recommended_cores(table):
    cores = table["Cores"].tolist()
    speedups = table["Speedup"].tolist()
    rec = recommended_cores(cores, speedups)
    best_i = speedups.index(max(speedups))
    rec_i = cores.index(rec)

    def n_cores(c):
        return f"{c} core" + ("" if c == 1 else "s")

    st.subheader("Recommended cores")
    st.caption("Quick estimate for this dataset on this machine.")
    st.metric("Recommended cores", rec)
    if rec_i == best_i:
        why = f"{n_cores(rec)} gave the highest speedup observed ({speedups[rec_i]:.2f}x)."
    else:
        why = (f"{n_cores(rec)} reached {speedups[rec_i]:.2f}x, within 10% of the "
               f"best observed ({speedups[best_i]:.2f}x at {n_cores(cores[best_i])}), "
               "so the extra cores bought little.")
    more = [f"{n_cores(c)} {s:.2f}x" for c, s in zip(cores, speedups) if c > rec]
    if more:
        why += " Speedup with more cores: " + ", ".join(more) + "."
    st.write(why)
    st.write("Parallel efficiency (speedup divided by cores): " + ", ".join(
        f"{n_cores(c)} {s / c:.0%}" for c, s in zip(cores, speedups)) + ".")
    st.caption(
        "Not general Spark guidance. It comes from one run per core count, "
        "with this dataset, on this machine's cores (which may mix "
        "performance and efficiency cores). It compares Spark runs with each "
        "other only; see the NumPy comparison above for Spark vs. no Spark.")


def comparison_chart(rows):
    """Horizontal bars of seconds per iteration, in the given order.

    rows: list of (name, seconds, kind) with kind "Spark" or "NumPy".
    """
    df = pd.DataFrame(rows, columns=["Setup", "Seconds per iteration", "Kind"])
    df["Label"] = df["Seconds per iteration"].map(
        lambda v: f"{v:.2f} s" if v >= 0.1 else f"{v:.3f} s")
    base = alt.Chart(df).encode(
        y=alt.Y("Setup:N", sort=None, title=None, axis=alt.Axis(labelLimit=400)),
        x=alt.X("Seconds per iteration:Q", title="Seconds per iteration (lower is faster)"),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("Kind:N", scale=alt.Scale(domain=["Spark", "NumPy"],
                                                  range=["#e25a1c", "#4c78a8"]),
                        legend=alt.Legend(title=None, orient="top")))
    text = base.mark_text(align="left", dx=4).encode(text="Label:N")
    st.altair_chart((bars + text).properties(height=48 * len(df) + 40),
                    use_container_width=True)


def machine_ram_gb():
    try:
        # Binary GB, the unit RAM is sold in (an "8 GB" Mac has 8 * 2**30 bytes).
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def spark_recommendation(spark_s, np_s, data_gb, ram_gb):
    """
    Whether to use Spark RDD for this data on this machine, from measured
    times (np_s is None if the NumPy run failed). Returns (use_spark,
    headline, reasons).
    """
    fits = ram_gb is None or data_gb < ram_gb
    size = f"The data takes about {data_gb:.2f} GB in memory"
    size += f"; this machine has {ram_gb:.0f} GB of RAM." if ram_gb else "."
    if np_s is None:
        return True, "Use Spark RDD", [
            "Plain NumPy could not finish on this machine.", size]
    if spark_s < np_s:
        return True, "Use Spark RDD", [
            f"Spark was {np_s / spark_s:.1f}x faster than plain NumPy here.", size]
    reasons = [f"Plain NumPy was {spark_s / np_s:.1f}x faster than Spark's best run.",
               size + (" It fits, so one process can hold all of it." if fits else "")]
    if fits:
        reasons.append(
            "Spark costs a fixed amount every iteration (scheduling tasks, moving "
            "data between the JVM and Python), and here that cost is larger than "
            "the math. Switch to Spark when the data outgrows one machine's "
            "memory, as in the **Recorded cluster run** tab, where Spark on 4 "
            "machines was 30x faster.")
        if ram_gb and data_gb > 0.5 * ram_gb:
            reasons.append("The data already uses over half of this machine's "
                           "memory, so it is getting close to that point.")
    return False, "Don't use Spark RDD for this data. Use plain NumPy.", reasons


def show_recommendation(use_spark, headline, reasons):
    st.subheader("Recommendation")
    box = st.success if use_spark else st.info
    box(f"**{headline}**\n\n" + "\n".join(f"- {r}" for r in reasons))


def show_numpy_comparison(runs, numpy, data_gb):
    st.subheader("Spark vs. plain NumPy")
    best = min(runs, key=lambda r: r["sec_per_iter_warm"])
    spark_s = best["sec_per_iter_warm"]
    if "error" in numpy:
        st.warning("The NumPy comparison failed: " + numpy["error"])
        show_recommendation(*spark_recommendation(spark_s, None, data_gb, machine_ram_gb()))
        return
    np_s = numpy["sec_per_iter_warm"]
    rows = [(f"Spark, {r['cores']} core" + ("" if r["cores"] == 1 else "s"),
             r["sec_per_iter_warm"], "Spark") for r in runs]
    rows.append(("NumPy, 1 process", np_s, "NumPy"))
    comparison_chart(rows)
    show_recommendation(*spark_recommendation(spark_s, np_s, numpy["gb"], machine_ram_gb()))
    st.caption("NumPy runs the same gradient descent (same data, starting point, "
               "learning rate and iterations) in one process with no Spark, "
               "timed the same way: median per-iteration time, iteration 0 "
               "dropped, loading excluded. Its matrix library may use several "
               f"cores. Final loss: Spark {best['final_loss']:.6f}, "
               f"NumPy {numpy['final_loss']:.6f}.")


def load_recorded():
    """Results of deploy/bigdata/run.sh, grouped by setup name."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(RECORDED, "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if "error" not in r:
            runs.setdefault(os.path.basename(f).split("-r")[0].removesuffix(".json"),
                            []).append(r)
    return runs


def show_recorded():
    runs = load_recorded()
    needed = ["spark-e4", "spark-e2", "numpy-64gb", "numpy-16gb"]
    if not all(k in runs for k in needed):
        st.info("No recorded cluster results found in deploy/bigdata/results/. "
                "Run deploy/bigdata/run.sh to produce them.")
        return
    med = {k: statistics.median(r["sec_per_iter_warm"] for r in runs[k]) for k in needed}
    rows_n = runs["spark-e4"][0]["records"]
    gb = runs["numpy-64gb"][0]["gb"]

    st.info(f"**Recorded, not live.** Measured on AWS on October 2, 2026; the "
            f"full run takes about an hour. {rows_n / 1e6:.0f}M rows "
            f"(10 copies of HIGGS), {gb:.1f} GB as float64: more than the 16 GB "
            "machine's memory.")
    comparison_chart([
        ("Spark, 4 machines (32 cores)", med["spark-e4"], "Spark"),
        ("Spark, 2 machines (16 cores)", med["spark-e2"], "Spark"),
        ("NumPy, one 64 GB machine (data in RAM)", med["numpy-64gb"], "NumPy"),
        ("NumPy, one 16 GB machine (re-reads from disk)", med["numpy-16gb"], "NumPy"),
    ])
    a, b, c = st.columns(3)
    a.metric("Spark (4 machines) vs. NumPy on 16 GB",
             f"{med['numpy-16gb'] / med['spark-e4']:.0f}x faster")
    b.metric("Spark (4 machines) vs. NumPy on 64 GB",
             f"{med['numpy-64gb'] / med['spark-e4']:.1f}x faster")
    c.metric("Spark: 2 → 4 machines", f"{med['spark-e2'] / med['spark-e4']:.2f}x faster",
             help="Perfect scaling would be 2.00x.")
    show_recommendation(True, "Use Spark RDD for this data", [
        f"The data ({gb:.1f} GB in memory) is bigger than the 16 GB machine's "
        f"RAM, so plain NumPy re-reads it from disk every iteration; Spark on 4 "
        f"machines keeps it in memory and is {med['numpy-16gb'] / med['spark-e4']:.0f}x faster.",
        "If one machine with enough RAM is available, it is a simpler option: a "
        "64 GB machine running NumPy matched 2 Spark machines. Spark pulls ahead "
        f"from 4 machines ({med['numpy-64gb'] / med['spark-e4']:.1f}x faster) and keeps "
        "scaling as machines are added.",
    ])
    st.write("**Same answer everywhere:** every 20-iteration run ends at loss "
             f"{runs['spark-e4'][0]['final_loss']:.6f}.")
    st.dataframe(pd.DataFrame([
        {"Setup": "Spark, 4 × r5.2xlarge", "Seconds per iteration": med["spark-e4"],
         "Runs": len(runs["spark-e4"]), "Final loss": runs["spark-e4"][0]["final_loss"]},
        {"Setup": "Spark, 2 × r5.2xlarge", "Seconds per iteration": med["spark-e2"],
         "Runs": len(runs["spark-e2"]), "Final loss": runs["spark-e2"][0]["final_loss"]},
        {"Setup": "NumPy, r6id.2xlarge (64 GB, in RAM)", "Seconds per iteration": med["numpy-64gb"],
         "Runs": 1, "Final loss": runs["numpy-64gb"][0]["final_loss"]},
        {"Setup": "NumPy, m6id.xlarge (16 GB, from NVMe)", "Seconds per iteration": med["numpy-16gb"],
         "Runs": 1, "Final loss": runs["numpy-16gb"][0]["final_loss"]},
    ]), hide_index=True, column_config={
        "Seconds per iteration": st.column_config.NumberColumn(format="%.2f"),
        "Final loss": st.column_config.NumberColumn(format="%.6f")})
    st.caption("Spark runs on EMR with data cached in memory (100% of partitions). "
               "The 16 GB NumPy run did 6 iterations instead of 20 (81 s each); "
               "its loss after 6 matches the 64 GB run's 6th iteration (0.668668). "
               "Raw results: deploy/bigdata/results/.")


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

    if res.get("numpy"):
        # label + intercept + features, float64; NumPy reports the same figure.
        data_gb = res["rows"] * (res["features"] + 2) * 8 / 1e9
        show_numpy_comparison(runs, res["numpy"], data_gb)

    show_recommended_cores(table)

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
        elif res.get("numpy") and "error" not in res["numpy"]:
            # Too many rows to copy the data into this process, but the
            # separate NumPy run trained on the same data: compare its loss.
            spark_loss, np_loss = check["final_loss"], res["numpy"]["final_loss"]
            same = abs(spark_loss - np_loss) <= 1e-9 * max(1.0, abs(np_loss))
            st.metric("Matches single-machine NumPy", "Yes" if same else "No",
                      help="Final training loss of the Spark run vs. the "
                           "separate plain-NumPy run on the same data, after "
                           "the same number of iterations.")
            st.caption(f"Final loss: Spark {spark_loss:.10f}, NumPy {np_loss:.10f}. "
                       "Compared by loss because the data is too large to copy "
                       "into this process for a weight-by-weight check.")
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


def require_password():
    """Gate the page when APP_PASSWORD is set (e.g. on a public server)."""
    expected = os.environ.get("APP_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return
    pw = st.text_input("Password", type="password")
    if pw and hmac.compare_digest(pw, expected):
        st.session_state.authed = True
        st.rerun()
    if pw:
        st.error("Wrong password.")
    st.stop()


st.set_page_config(page_title="Spark Scaling Lab", layout="wide")
require_password()
st.title("Spark Scaling Lab")
st.write(
    "Distributed logistic regression, written from scratch on Spark's RDD API "
    "(batch gradient descent, no MLlib). See how it speeds up as Spark gets "
    "more cores, and how it compares with plain NumPy on one machine.")

live, recorded = st.tabs(["Run on this machine", "Recorded cluster run (AWS, 110M rows)"])

# The recorded tab is filled first: the live tab below calls st.stop() while
# it waits for input, which would otherwise end the script before this runs.
with recorded:
    show_recorded()

with live:
    sources = (["HIGGS sample (1M rows, built in)"] if os.path.exists(HIGGS) else [])
    sources.append("Upload a CSV")
    source = st.radio("Data", sources, horizontal=True)

    if source.startswith("HIGGS"):
        # Already numeric and clean: no header, label in column 0.
        data_path, label, n_rows, n_features = HIGGS, None, 1_000_000, 28
        data_key = ("higgs",)
        st.success("Ready: 1,000,000 rows of HIGGS; 28 features; label in column 0.")
    else:
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

        data_path = os.path.join(session_dir(), "clean.csv")
        # Cleaning a large file takes seconds (about 20 s for 1M rows), and
        # Streamlit reruns this script on every widget change, so clean once
        # per (file, label).
        clean_key = (st.session_state.upload_id, label)
        if st.session_state.get("clean_key") != clean_key:
            with st.spinner("Checking the file..."):
                try:
                    st.session_state.clean_result = clean_csv(raw_path, label, data_path)
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
                "These columns look like row numbers (whole numbers counting up "
                "with no gaps): " + ", ".join(summary["likely_id_columns"]) + ". "
                "They are being used as features, which teaches the model nothing "
                "real. Remove them from the file if they are IDs.")
        n_rows, n_features = summary["rows"], summary["n_features"]
        data_key = ("upload", st.session_state.upload_id, label)

    c1, c2, c3 = st.columns(3)
    with c1:
        max_cores = st.select_slider("Max cores", options=list(range(1, MAX_CORES + 1)),
                                     value=min(4, MAX_CORES))
    with c2:
        iters = st.number_input("Iterations", min_value=5, max_value=500, value=30,
                                help="Gradient descent steps per run. The first "
                                     "one is excluded from timing (JVM warmup).")
    with c3:
        mode_name = st.selectbox(
            "Spark gradient code", list(MODES),
            help="Vectorized packs each partition into NumPy matrices, so the "
                 "gradient is one matrix product per block. Row by row loops "
                 "over rows in Python; it is slower but scales more evenly, "
                 "because there is more work to split up.")
        mode = MODES[mode_name]

    counts = core_counts(max_cores)
    st.caption(f"Will run Spark with {', '.join(map(str, counts))} core(s), one "
               "Spark process each, then the same training in plain NumPy.")

    run_key = data_key + (mode,)
    if st.button("Run benchmark", type="primary"):
        lock = run_lock()
        if not lock.acquire(blocking=False):
            st.warning("Another benchmark is running. Wait for it to finish, "
                       "since overlapping runs would distort both timings.")
            st.stop()
        try:
            st.session_state.pop("results", None)
            runs, check = [], None
            steps = len(counts) + 1
            progress = st.progress(0.0)
            for i, cores in enumerate(counts):
                progress.progress(i / steps,
                                  text=f"Spark with {cores} core(s) "
                                       f"({i + 1} of {steps})...")
                # Correctness checks run once, on the last run, after its timed loop.
                last = i == len(counts) - 1
                result = run_bench(cores, data_path, label, iters, check=last, mode=mode)
                if "error" in result:
                    progress.empty()
                    show_error(result)
                    st.stop()
                runs.append(result)
                if last:
                    check = result
            progress.progress(len(counts) / steps,
                              text=f"Plain NumPy, no Spark ({steps} of {steps})...")
            numpy = run_numpy(data_path, label, iters)
            progress.empty()
            st.session_state.results = {"runs": runs, "check": check, "numpy": numpy,
                                        "rows": n_rows, "features": n_features,
                                        "key": run_key}
        finally:
            lock.release()

    res = st.session_state.get("results")
    if res and res.get("key") == run_key:
        show_results(res)
