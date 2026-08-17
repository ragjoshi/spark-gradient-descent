import csv, statistics

rows = []
with open("results.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

# Quarantine contamination: keep only the 50-iter sweep, and for any
# duplicate (cores,rep) keep the LAST appended run (the real interleaved
# sweep ran after the early cold-start warmups).
rows = [r for r in rows if int(r["iters"]) == 50]
dedup = {}
for r in rows:
    dedup[(int(r["cores"]), int(r["rep"]))] = r  # last write wins
rows = list(dedup.values())

by_cores = {}
for r in rows:
    by_cores.setdefault(int(r["cores"]), []).append(r["sec_per_iter_warm"])

T = {c: statistics.median(v) for c, v in by_cores.items()}
T1 = T[1]

print(f"{'cores':>5} {'s/iter':>9} {'speedup':>8} {'eff':>7} {'serial_f':>9}")
for c in sorted(T):
    S = T1 / T[c]
    E = S / c
    s = ((1/S) - (1/c)) / (1 - 1/c) if c > 1 else float('nan')
    print(f"{c:>5} {T[c]:>9.4f} {S:>7.2f}x {E*100:>6.1f}% {s*100:>8.1f}%")