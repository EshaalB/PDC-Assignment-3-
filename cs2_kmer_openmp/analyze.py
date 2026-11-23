#!/usr/bin/env python3
"""
CS2 - turns results/raw.txt into tables (results/*.csv) and graphs (results/*.png).
Every time in raw.txt is already the median of 3 runs (the programs do that themselves).

Usage:  python3 analyze.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


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
            row["status"] = line.split()[-1]
            rows.append(row)
    return rows


def write_csv(name, rows, fields):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    rows = read_runs()
    serial = {(r["k"], r["skew"]): r["time_med"] for r in rows if r["exp"] == "serial"}

    # ---------------- strong scaling of A / B / C ----------------
    strong = [r for r in rows if r["exp"] == "strong"]
    for r in strong:
        t1 = serial[(r["k"], r["skew"])]
        r["T_serial"] = round(t1, 4)
        r["speedup"] = round(t1 / r["time_med"], 3)
        r["efficiency"] = round(t1 / r["time_med"] / r["threads"], 3)
    strong.sort(key=lambda r: (r["skew"], r["k"], r["version"], r["threads"]))
    write_csv("strong_scaling.csv", strong,
              ["version", "k", "skew", "threads", "time_med", "merge_med", "T_serial", "speedup",
               "efficiency", "Mkmers_per_s", "total", "expected", "lost", "mismatched_bins",
               "extra_MB", "status"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k in zip(axes, [8.0, 12.0]):
        pts_all = [r for r in strong if r["k"] == k and r["skew"] == 0]
        ts = sorted({r["threads"] for r in pts_all})
        ax.plot(ts, ts, "k--", label="ideal")
        for v, name in [("B", "B atomic"), ("C", "C private + merge")]:
            pts = [r for r in pts_all if r["version"] == v]
            ax.plot([r["threads"] for r in pts], [r["speedup"] for r in pts], "o-", label=name)
        ax.set_title(f"CS2 strong scaling, k={int(k)}  ({4 ** int(k):,} bins), L=1e8")
        ax.set_xlabel("OpenMP threads")
        ax.set_ylabel("speedup vs serial")
        ax.set_xticks(ts)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "uniform input; each point = median of 3 runs", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "strong_scaling.png"), dpi=150, bbox_inches="tight")

    # ---------------- false sharing ----------------
    fs = [r for r in rows if r["exp"] == "falseshare"]
    fs.sort(key=lambda r: (r["k"], r["threads"], r["version"]))
    table = []
    for k in sorted({r["k"] for r in fs}):
        for t in sorted({r["threads"] for r in fs}):
            u = next(r for r in fs if r["k"] == k and r["threads"] == t and r["version"] == "U")
            c = next(r for r in fs if r["k"] == k and r["threads"] == t and r["version"] == "C")
            table.append({"k": int(k), "threads": int(t),
                          "unpadded_time": u["time_med"], "padded_time": c["time_med"],
                          "slowdown_from_false_sharing": round(u["time_med"] / c["time_med"], 2)})
    write_csv("false_sharing.csv", table, list(table[0].keys()))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    k1 = [r for r in table if r["k"] == 1]
    ax.plot([r["threads"] for r in k1], [r["unpadded_time"] for r in k1], "o-", label="unpadded (false sharing)")
    ax.plot([r["threads"] for r in k1], [r["padded_time"] for r in k1], "o-", label="padded to 64 B")
    ax.set_title("CS2 false sharing, k=1 (4 counters per thread)")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("time (s)")
    ax.set_xticks([r["threads"] for r in k1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="median of 3 runs", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "false_sharing.png"), dpi=150)

    print("written to", RES)
    for name in ["strong_scaling.csv", "false_sharing.csv"]:
        print("\n==", name)
        print(open(os.path.join(RES, name)).read())


if __name__ == "__main__":
    main()
