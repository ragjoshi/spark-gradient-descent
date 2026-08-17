# bench.py — Week 3: strong-scaling benchmark
import sys, csv, os
from pyspark import SparkContext
from train import load_higgs, standardize, train

N_PARTS = 8        # FIXED across all runs
MAX_ITER = 50
OUT = "results.csv"

def run_once(cores, rep):
    sc = SparkContext(f"local[{cores}]", f"bench-{cores}c-r{rep}")
    sc.setLogLevel("WARN")
    try:
        raw = load_higgs(sc, "higgs_1m.csv", N_PARTS)
        data = standardize(sc, raw, d_feat=28)
        
        
        w, stats = train(sc, data, 29, lr=0.5,
                               max_iter=MAX_ITER, tol=0.0, verbose=False)
        pi = stats["per_iter"]
        
        return stats
    finally:
        sc.stop()

if __name__ == "__main__":
    cores, rep = int(sys.argv[1]), int(sys.argv[2])
    stats = run_once(cores, rep)
    print(f"cores={cores} rep={rep} warm={stats['sec_per_iter_warm']:.4f}")

    row = [cores, rep, stats["partitions"], stats["records"],
           stats["iters"], stats["sec_per_iter_warm"]]

    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["cores", "rep", "partitions",
                             "records", "iters", "sec_per_iter_warm"])
        writer.writerow(row)
