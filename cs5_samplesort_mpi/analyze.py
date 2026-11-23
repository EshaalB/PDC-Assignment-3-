#!/usr/bin/env python3
"""
CS5 - turns results/raw.txt + results/pingpong.txt into tables (results/*.csv)
and graphs (results/*.png). Every number is the MEDIAN of the runs of that configuration.

Usage:  python3 analyze.py
"""
import csv
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PHASES = ["t_localsort", "t_sampling", "t_partition", "t_alltoallv", "t_merge"]


def read_runs():
    rows = []
    with open(os.path.join(RES, "raw.txt")) as f:
        for line in f:
            row = {}
            for token in line.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    row[key] = value
            m = re.search(r"COUNTS((?: \d+)+)", line)
            row["counts"] = m.group(1).split() if m else []
            row["verified"] = "PASS" in line or row.get("impl") == "serial"
            rows.append(row)
    return rows


def median_of(rows, group_keys, value_keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, members in groups.items():
        row = dict(zip(group_keys, key))
        row["runs"] = len(members)
        row["all_verified"] = all(m["verified"] for m in members)
        for v in value_keys:
            row[v] = round(statistics.median(m[v] for m in members), 6)
        row["counts"] = " ".join(members[0]["counts"])
        out.append(row)
    return out


def write_csv(name, rows, fields):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    rows = read_runs()
    dists = ["uniform", "zipf", "dup16"]

    # ---------------- ping-pong -> alpha, beta ----------------
    alpha = beta = None
    with open(os.path.join(RES, "pingpong.txt")) as f:
        for line in f:
            if line.startswith("FIT"):
                alpha = float(re.search(r"alpha=([\d.]+)", line).group(1)) * 1e-6   # s
                beta = float(re.search(r"beta=([\d.]+)", line).group(1)) * 1e-9     # s/byte

    # ---------------- serial ----------------
    n_total = next(r["N"] for r in rows if r["exp"] == "serial")
    serial ={r["dist"]: r for r in median_of([r for r in rows if r["exp"] == "serial"], ["dist"], ["time"])}

    # ---------------- strong scaling ----------------
    strong = median_of([r for r in rows if r["exp"] == "strong"], ["dist", "p"],
                       ["time"] + PHASES + ["imbalance", "max_count"])
    strong.sort(key=lambda r: (dists.index(r["dist"]), r["p"]))
    for r in strong:
        t1 = serial[r["dist"]]["time"]
        r["T_serial"] = t1
        r["speedup"] = round(t1 / r["time"], 3)
        r["efficiency"] = round(t1 / r["time"] / r["p"], 3)
        # predicted Alltoallv: each rank sends (p-1) messages holding (p-1)/p of its N/p keys
        bytes_out = (r["p"] - 1) / r["p"] * (n_total / r["p"]) * 4
        r["alltoallv_predicted"] = round((r["p"] - 1) * alpha + beta * bytes_out, 6) if alpha else ""
    write_csv("strong_scaling.csv", strong,
              ["dist", "p", "runs", "all_verified", "time", "T_serial", "speedup", "efficiency"] + PHASES +
              ["alltoallv_predicted", "imbalance", "max_count", "counts"])

    # speedup graph
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ps = sorted({r["p"] for r in strong})
    ax.plot(ps, ps, "k--", label="ideal")
    for d in dists:
        pts = [r for r in strong if r["dist"] == d]
        ax.plot([r["p"] for r in pts], [r["speedup"] for r in pts], "o-", label=d)
    ax.set_title("CS5 sample sort, N = 1e8 keys")
    ax.set_xlabel("MPI processes p")
    ax.set_ylabel("speedup vs serial qsort")
    ax.set_xticks(ps)
    ax.grid(alpha=0.3)
    ax.legend(title="median of 3 runs", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "strong_scaling.png"), dpi=150)

    # stacked phase-time bars (uniform input)
    fig, ax = plt.subplots(figsize=(6, 4))
    pts = [r for r in strong if r["dist"] == "uniform"]
    bottom = [0.0] * len(pts)
    labels = ["local sort", "sampling", "partition", "Alltoall(v)", "merge"]
    for ph, lab in zip(PHASES, labels):
        vals = [r[ph] for r in pts]
        ax.bar([str(int(r["p"])) for r in pts], vals, bottom=bottom, label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("CS5 time per phase (uniform, N = 1e8)")
    ax.set_xlabel("MPI processes p")
    ax.set_ylabel("time (s), slowest rank per phase")
    ax.legend(title="median of 3 runs", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "phase_breakdown.png"), dpi=150)

    # ---------------- oversampling ----------------
    over = median_of([r for r in rows if r["exp"] == "oversample"], ["dist", "oversample"],
                     ["time", "imbalance", "max_count"])
    over.sort(key=lambda r: (dists.index(r["dist"]), r["oversample"]))
    write_csv("oversampling.csv", over,
              ["dist", "oversample", "runs", "all_verified", "time", "imbalance", "max_count", "counts"])

    print("alpha = %s s, beta = %s s/byte" % (alpha, beta))
    print("serial:", {d: serial[d]["time"] for d in serial})
    for name in ["strong_scaling.csv", "oversampling.csv"]:
        print("\n==", name)
        print(open(os.path.join(RES, name)).read())


if __name__ == "__main__":
    main()
