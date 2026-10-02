# bench.py: strong-scaling benchmark, one (cores, rep) run per process
#
# Usage:
#   python bench.py <cores> <rep>
#       HIGGS default: higgs_1m.csv, 50 iterations, appends to results.csv.
#   python bench.py <cores> <rep> --data FILE --label COL [--iters N] [--parts P] [--check]
#       Any numeric CSV with a header. Validated by preprocess.clean_csv.
#       Does not touch results.csv.
#
# --mode rows (default) computes the gradient row by row in Python;
# --mode blocks packs each partition into a NumPy matrix (see train.to_blocks).
# HIGGS runs in blocks mode go to results_blocks.csv instead of results.csv.
#
# --cluster runs under spark-submit on a real cluster (deploy/bigdata/):
# spark-submit picks the master, <cores> is only a label, --data may be an
# s3:// path or directory, and partitions are formed with coalesce instead
# of a full repartition shuffle. The data is not run through
# preprocess.clean_csv, so it must already be numeric with a 0/1 label.
#
# The last line of stdout is always one JSON object, for app.py to parse:
# the run's timing (plus correctness checks with --check), or {"error": ...}.
import argparse
import csv
import json
import os
import sys
import tempfile
import traceback

from train import make_context, load_csv, standardize, train, validate, reference_check

N_PARTS = 8        # FIXED across all runs
MAX_ITER = 50
LR = 0.5
OUT = {"rows": "results.csv", "blocks": "results_blocks.csv"}
HIGGS = "higgs_1m.csv"


def run_once(cores, rep, path=HIGGS, label_col=None, iters=MAX_ITER,
             n_parts=N_PARTS, check=False, mode="rows", cluster=False):
    sc = make_context(None if cluster else cores, f"bench-{cores}c-r{rep}")
    try:
        raw, d_feat = load_csv(sc, path, label_col, n_parts, shuffle=not cluster)
        data = standardize(sc, raw, d_feat=d_feat)

        w, stats = train(sc, data, d_feat + 1, lr=LR,
                         max_iter=iters, tol=0.0, verbose=False, mode=mode)
        stats["features"] = d_feat

        # Correctness checks run after the timed loop, so they never
        # affect the timing numbers.
        if check:
            stats.update(validate(data, w, max_rows=100_000, verbose=False))
            stats.update(reference_check(data, w, LR, stats["iters"], verbose=False))
        return stats
    finally:
        sc.stop()


def append_results(cores, rep, stats, mode="rows"):
    row = [cores, rep, stats["partitions"], stats["records"],
           stats["iters"], stats["sec_per_iter_warm"]]

    out = OUT[mode]
    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["cores", "rep", "partitions",
                             "records", "iters", "sec_per_iter_warm"])
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description="One strong-scaling benchmark run.")
    p.add_argument("cores", type=int)
    p.add_argument("rep", type=int)
    p.add_argument("--data", help="CSV path (default: HIGGS layout, higgs_1m.csv)")
    p.add_argument("--label", help="label column name; requires a header row")
    p.add_argument("--iters", type=int, default=MAX_ITER)
    p.add_argument("--parts", type=int, default=N_PARTS)
    p.add_argument("--check", action="store_true",
                   help="also run the scikit-learn and NumPy correctness checks")
    p.add_argument("--mode", choices=["rows", "blocks"], default="rows",
                   help="row-at-a-time Python gradient, or one NumPy matrix per partition")
    p.add_argument("--cluster", action="store_true",
                   help="running under spark-submit on a cluster; see deploy/bigdata/")
    args = p.parse_args()

    # The HIGGS default keeps its original behavior: no preprocessing,
    # and the run is appended to results.csv for analyze.py.
    default_run = args.data is None
    path = HIGGS if default_run else args.data
    cleaned = None
    out = {"cores": args.cores, "rep": args.rep}

    try:
        if "://" not in path and not os.path.exists(path):
            print(json.dumps({"error": f"File not found: {path}", "kind": "data"}))
            return 2

        # clean_csv loads the whole file with pandas on this machine, so it is
        # skipped for cluster runs: their data must already be clean (numeric,
        # no missing values, 0/1 label).
        if args.label is not None and not args.cluster:
            from preprocess import DataError, clean_csv
            cleaned = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
            try:
                out.update(clean_csv(path, args.label, cleaned))
            except DataError as e:
                print(json.dumps({"error": str(e), "kind": "data"}))
                return 2
            path = cleaned

        stats = run_once(args.cores, args.rep, path, args.label,
                         args.iters, args.parts, args.check, args.mode,
                         args.cluster)
        out.update(stats)

        if default_run:
            print(f"mode={args.mode} cores={args.cores} rep={args.rep} "
                  f"warm={stats['sec_per_iter_warm']:.4f}")
            append_results(args.cores, args.rep, stats, args.mode)

        print(json.dumps(out))
        return 0
    except Exception as e:
        traceback.print_exc()
        first = (str(e).strip().splitlines() or [""])[0][:300]
        print(json.dumps({"error": f"{type(e).__name__}: {first}", "kind": "spark"}))
        return 1
    finally:
        if cleaned is not None and os.path.exists(cleaned):
            os.remove(cleaned)


if __name__ == "__main__":
    sys.exit(main())
