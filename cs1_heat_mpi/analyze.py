#!/usr/bin/env python3
"""
CS1 - turns results/raw.txt into tables (results/*.csv) and graphs (results/*.png).
Every number is the MEDIAN of the runs of that configuration.

Usage:  python3 analyze.py
"""
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def read_runs():
    """Each line looks like 'exp=strong run=1 RESULT impl=mpi N=1024 p=4 ... time=0.5 ...'."""
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
            rows.append(row)
    return rows


def median_of(rows, group_keys, value_keys):
    """Group rows by group_keys and take the median of every value key."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: tuple(str(x).zfill(8) for x in kv[0])):
        row = dict(zip(group_keys, key))
        row["runs"] = len(members)
        for v in value_keys:
            row[v] = statistics.median(m[v] for m in members)
        out.append(row)
    return out


def write_csv(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = read_runs()
    serial = {r["N"]: r for r in median_of([r for r in rows if r["exp"] == "serial"], ["N"], ["time"])}

    # ---------------- strong scaling ----------------
    strong = median_of([r for r in rows if r["exp"] == "strong"],
                       ["N", "decomp", "mode", "p"], ["time", "comm_max", "allreduce"])
    for r in strong:
        t1 = serial[r["N"]]["time"]
        r["T_serial"] = round(t1, 4)
        r["speedup"] = round(t1 / r["time"], 3)
        r["efficiency"] = round(t1 / r["time"] / r["p"], 3)
        r["comm_fraction"] = round(r["comm_max"] / r["time"], 3)
    write_csv("strong_scaling.csv", strong)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, N in zip(axes, sorted(serial)):
        ps = sorted({r["p"] for r in strong})
        ax.plot(ps, ps, "k--", label="ideal")
        for decomp in ["1D", "2D"]:
            for mode in ["blocking", "nonblocking"]:
                pts = [r for r in strong if r["N"] == N and r["decomp"] == decomp and r["mode"] == mode]
                ax.plot([r["p"] for r in pts], [r["speedup"] for r in pts], "o-", label=f"{decomp} {mode}")
        ax.set_title(f"CS1 strong scaling, N={int(N)}, T=500")
        ax.set_xlabel("MPI processes p")
        ax.set_ylabel("speedup  S(p) = T_serial / T_p")
        ax.set_xticks(ps)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "each point = median of 3 runs", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "strong_scaling.png"), dpi=150, bbox_inches="tight")

    # ---------------- weak scaling ----------------
    weak = median_of([r for r in rows if r["exp"] == "weak"], ["decomp", "p", "N"], ["time", "comm_max"])
    base = {r["decomp"]: r["time"] for r in weak if r["p"] == 1}
    for r in weak:
        r["weak_efficiency"] = round(base[r["decomp"]] / r["time"], 3)
    write_csv("weak_scaling.csv", weak)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for decomp in ["1D", "2D"]:
        pts = [r for r in weak if r["decomp"] == decomp]
        ax.plot([r["p"] for r in pts], [r["time"] for r in pts], "o-", label=f"{decomp} non-blocking")
    ax.set_title("CS1 weak scaling (2048x2048 cells per process)")
    ax.set_xlabel("MPI processes p")
    ax.set_ylabel("time for 500 steps (s)")
    ax.set_xticks(sorted({r["p"] for r in weak}))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="median of 3 runs", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "weak_scaling.png"), dpi=150)

    # ---------------- K = 1 vs K = 50 ----------------
    kc = median_of([r for r in rows if r["exp"] == "kcheck"], ["N", "p", "K"], ["time", "allreduce"])
    write_csv("convergence_check.csv", kc)

    print("written to", RES)
    for name in ["strong_scaling.csv", "weak_scaling.csv", "convergence_check.csv"]:
        print("\n==", name)
        print(open(os.path.join(RES, name)).read())


if __name__ == "__main__":
    main()
