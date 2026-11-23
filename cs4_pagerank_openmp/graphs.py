#!/usr/bin/env python3
"""
make_graphs_cs4_from_output.py
Draws all Case Study 4 (PageRank) graphs and tables from the numbers that were COPIED FROM THE
TERMINAL OUTPUT of your own runs (graph: R-MAT scale 20, n = 1,048,576, m = 16,084,838).
No experiment has to be re-run.

    python3 make_graphs_cs4_from_output.py
    python3 make_graphs_cs4_from_output.py --peak-bw 38.4     # optional: your DRAM bandwidth in GB/s

Needs: python3, numpy, matplotlib   (pip install numpy matplotlib)
Output: report/figures/cs4_*.png  and  report/tables/cs4_*.md  (next to this script)

OPTIONAL degree-distribution plot (takes 5 seconds, not the full experiment):
    ./serial -r 1 -D results/deg        # inside cs4_pagerank_openmp/ (create the folder 'results' first)
then pass the folder:   --deg cs4_pagerank_openmp/results
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, GREY, PURPLE = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#7f7f7f", "#9467bd"

# ======================================================================================
#  MEASURED DATA  (copied from your terminal output; every time = median of 3 runs)
# ======================================================================================
N, M, ITERS, REPS = 1_048_576, 16_084_838, 24, 3
GRAPH = f"R-MAT scale 20: n = {N:,}, m = {M:,} edges, {ITERS} iterations"

T_SERIAL = 1.4175          # serial median of 3 (min 1.3716)
EPS_SERIAL = 272.3         # M edges/s

# key -> dict(time, eps = M edges/s, gbps = model estimate, R = measured, Rmodel, threads = per-thread loop times)
RUN = {
    "pull_static_p4":  dict(p=4, algo="pull", label="static", time=0.7434, eps=519.3, gbps=7.86, R=1.736, Rm=2.200,
                            thr=[0.601, 0.245, 0.361, 0.178]),
    "push_static_p4":  dict(p=4, algo="push", label="static", time=3.0313, eps=127.4, gbps=3.14, R=1.757, Rm=2.201,
                            thr=[2.898, 1.519, 1.578, 0.602]),
    "pull_static_p8":  dict(p=8, algo="pull", label="static", time=0.7765, eps=497.2, gbps=7.52, R=2.178, Rm=3.286,
                            thr=[0.706, 0.446, 0.365, 0.217, 0.358, 0.209, 0.199, 0.095]),
    "pull_dyn256_p8":  dict(p=8, algo="pull", label="dynamic,256", time=0.3477, eps=1110.3, gbps=16.80, R=1.001, Rm=3.286,
                            thr=[0.282, 0.282, 0.282, 0.282, 0.283, 0.283, 0.282, 0.282]),
    "pull_guided64_p8": dict(p=8, algo="pull", label="guided,64", time=0.8784, eps=439.5, gbps=6.65, R=1.229, Rm=3.286,
                             thr=[0.445, 0.366, 0.421, 0.302, 0.311, 0.384, 0.299, 0.369]),
    "pull_edge_p8":    dict(p=8, algo="pull", label="edge-balanced static", time=0.5507, eps=701.0, gbps=10.61, R=1.179, Rm=3.286,
                            thr=[0.448, 0.367, 0.429, 0.528, 0.520, 0.360, 0.499, 0.434]),
    "pull_p1":         dict(p=1, algo="pull", label="static", time=1.5535, eps=248.5, gbps=3.76, R=1.0, Rm=1.0, thr=[1.421]),
    "pushplain_p1":    dict(p=1, algo="pushplain", label="static", time=1.9471, eps=198.3, gbps=4.90, R=1.0, Rm=1.0, thr=[1.746]),
    "push_p1":         dict(p=1, algo="push", label="static", time=4.3227, eps=89.3, gbps=2.21, R=1.0, Rm=1.0, thr=[4.385]),
    # second, separate experiment (original vs randomly relabelled vertex ids), pull, static, p = 8
    "pull_static_p8_rerun": dict(p=8, algo="pull", label="static (original ids, 2nd run)", time=0.6941, eps=556.2, gbps=8.41,
                                 R=2.113, Rm=3.286, thr=[0.621, 0.344, 0.394, 0.175, 0.369, 0.178, 0.185, 0.086]),
    "pull_static_p8_perm":  dict(p=8, algo="pull", label="static (shuffled ids)", time=0.6207, eps=621.9, gbps=9.41,
                                 R=1.237, Rm=1.041, thr=[0.336, 0.514, 0.310, 0.446, 0.460, 0.267, 0.477, 0.512]),
}


# ======================================================================================
def save(fig, name, outdir):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, HERE))


def write_md(tabdir, name, title, header, rows, note=None):
    path = os.path.join(tabdir, name)
    with open(path, "w") as f:
        f.write(f"**{title}**\n\n| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")
        if note:
            f.write(f"\n*{note}*\n")
    print("  wrote", os.path.relpath(path, HERE))


def S(t):
    return T_SERIAL / t


# ------------------------------------------------------------------ 1. strong scaling
def fig_strong(out, tab):
    pts = {"pull": [(1, RUN["pull_p1"]["time"]), (4, RUN["pull_static_p4"]["time"]), (8, RUN["pull_static_p8"]["time"])],
           "push": [(1, RUN["push_p1"]["time"]), (4, RUN["push_static_p4"]["time"])]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    ps = np.arange(1, 9)
    a1.plot(ps, ps, "k--", lw=1, label="ideal")
    rows = []
    for algo, col, mk in (("pull", BLUE, "o"), ("push", RED, "s")):
        p = np.array([x[0] for x in pts[algo]]); t = np.array([x[1] for x in pts[algo]])
        a1.plot(p, S(t), marker=mk, color=col, label=f"{algo}, schedule(static)")
        a2.plot(p, S(t) / p, marker=mk, color=col, label=f"{algo}, schedule(static)")
        for pi, ti in zip(p, t):
            rows.append([algo, pi, f"{ti:.3f}", f"{S(ti):.2f}", f"{S(ti) / pi:.2f}"])
    # best schedule at p = 8
    best = RUN["pull_dyn256_p8"]["time"]
    a1.plot([8], [S(best)], "*", ms=14, color=GREEN, label="pull, dynamic,256 (best)")
    a2.plot([8], [S(best) / 8], "*", ms=14, color=GREEN, label="pull, dynamic,256 (best)")
    rows.append(["pull (dynamic,256)", 8, f"{best:.3f}", f"{S(best):.2f}", f"{S(best) / 8:.2f}"])
    # second measurement of the same configuration, shows run-to-run noise
    r2 = RUN["pull_static_p8_rerun"]["time"]
    a1.plot([8], [S(r2)], "o", mfc="none", color=BLUE, label="pull static p=8, repeated run")
    a1.set_xlabel("threads p"); a1.set_ylabel("speedup  S(p) = T_serial / T_p")
    a2.set_xlabel("threads p"); a2.set_ylabel("efficiency  E(p) = S(p) / p")
    a2.axhline(1, ls="--", color="black", lw=1); a2.axhline(0.7, ls=":", color=GREY, lw=1); a2.set_ylim(0, 1.15)
    a1.legend(frameon=False, fontsize=8); a2.legend(frameon=False, fontsize=8)
    fig.suptitle(f"CS4 PageRank strong scaling - {GRAPH}\nserial T = {T_SERIAL:.3f} s; each point = median of {REPS} runs", fontsize=9)
    save(fig, "cs4_strong_scaling.png", out)
    write_md(tab, "cs4_strong.md", f"CS4 strong scaling ({GRAPH}); serial baseline {T_SERIAL:.4f} s, median of {REPS} runs",
             ["algorithm", "p", "T_p (s)", "S(p)", "E(p)"], rows,
             "Only the thread counts that were measured are listed. Speedup is relative to the serial program, "
             "not to the parallel program on one thread.")


# ------------------------------------------------------------------ 2. per-thread work (imbalance picture)
def fig_per_thread(out):
    panels = [("pull_static_p4", "static, p = 4"), ("pull_static_p8", "static, p = 8"),
              ("pull_dyn256_p8", "dynamic,256, p = 8"), ("pull_guided64_p8", "guided,64, p = 8"),
              ("pull_edge_p8", "edge-balanced static, p = 8"), ("pull_static_p8_perm", "static, shuffled vertex ids, p = 8")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    ymax = max(max(RUN[k]["thr"]) for k, _ in panels) * 1.18
    for ax, (k, title) in zip(axes.ravel(), panels):
        r = RUN[k]; thr = r["thr"]
        cols = [RED if t == max(thr) else BLUE for t in thr]
        ax.bar(range(len(thr)), thr, color=cols)
        ax.axhline(np.mean(thr), ls="--", color="black", lw=1)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{title}\nR = {max(thr) / np.mean(thr):.2f}   (run time {r['time']:.2f} s)", fontsize=9)
        ax.set_xlabel("thread id")
    for ax in axes[:, 0]:
        ax.set_ylabel("time in edge loop, summed over 24 iterations (s)")
    fig.suptitle("CS4 pull: how much work each thread did (red = slowest thread, dashed = mean). "
                 f"R = slowest / mean.  Median of {REPS} runs.\n"
                 "For static the same thread is always slowest, so R is exact; for dynamic/guided the slow thread changes "
                 "every iteration, so R understates the stalls - compare their run time instead.", fontsize=9)
    save(fig, "cs4_per_thread_load.png", out)


# ------------------------------------------------------------------ 3. imbalance: measured vs model, original vs shuffled
def fig_imbalance(out):
    groups = [("static\np = 4", "pull_static_p4"), ("static\np = 8", "pull_static_p8_rerun"),
              ("static, shuffled ids\np = 8", "pull_static_p8_perm")]
    x = np.arange(len(groups)); w = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(x - w / 2, [RUN[k]["R"] for _, k in groups], w, color=BLUE, label="measured R (slowest / mean thread time)")
    a1.bar(x + w / 2, [RUN[k]["Rm"] for _, k in groups], w, color=GREY, label="model R from edge counts of the static blocks")
    for i, (_, k) in enumerate(groups):
        a1.text(i - w / 2, RUN[k]["R"], f"{RUN[k]['R']:.2f}", ha="center", va="bottom", fontsize=8)
        a1.text(i + w / 2, RUN[k]["Rm"], f"{RUN[k]['Rm']:.2f}", ha="center", va="bottom", fontsize=8)
    a1.axhline(1, ls=":", color="black", lw=1)
    a1.set_xticks(x); a1.set_xticklabels([g for g, _ in groups]); a1.set_ylabel("load-imbalance ratio R")
    a1.set_title("Imbalance of schedule(static)", fontsize=9); a1.legend(frameon=False, fontsize=8)
    a1.set_ylim(0, 3.8)
    # right: time of original vs shuffled (same experiment)
    t = [RUN["pull_static_p8_rerun"]["time"], RUN["pull_static_p8_perm"]["time"]]
    a2.bar(["original ids", "shuffled ids"], t, color=[BLUE, GREEN])
    for i, v in enumerate(t):
        a2.text(i, v, f"{v:.3f} s", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("time (s), pull, static, p = 8"); a2.set_ylim(0, max(t) * 1.25)
    a2.set_title(f"Better balance, but only {100 * (1 - t[1] / t[0]):.0f}% faster", fontsize=9)
    fig.suptitle(f"CS4 effect of vertex numbering; median of {REPS} runs", fontsize=10)
    save(fig, "cs4_imbalance.png", out)


# ------------------------------------------------------------------ 4. schedules at p = 8
def fig_schedules(out, tab):
    keys = ["pull_static_p8", "pull_guided64_p8", "pull_edge_p8", "pull_dyn256_p8"]
    keys.sort(key=lambda k: -RUN[k]["time"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    best = min(RUN[k]["time"] for k in keys)
    labs = [RUN[k]["label"] + (f"   (R = {RUN[k]['R']:.2f})" if RUN[k]["label"] == "static" else "") for k in keys]
    cols = [GREEN if RUN[k]["time"] == best else BLUE for k in keys]
    ax.barh(labs, [RUN[k]["time"] for k in keys], color=cols)
    for i, k in enumerate(keys):
        ax.text(RUN[k]["time"], i, f"  {RUN[k]['time']:.3f} s  (speedup {S(RUN[k]['time']):.2f}x)", va="center", fontsize=9)
    ax.axvline(T_SERIAL, ls="--", color="black", lw=1)
    ax.text(T_SERIAL, len(keys) - 0.45, " serial", fontsize=8, va="bottom")
    ax.set_xlim(0, T_SERIAL * 1.25); ax.set_xlabel("time (s)")
    ax.set_title(f"CS4 pull, p = 8: schedule comparison (green = best); {GRAPH}; median of {REPS} runs", fontsize=8)
    save(fig, "cs4_schedules.png", out)
    rows = [[RUN[k]["label"], f"{RUN[k]['time']:.3f}", f"{S(RUN[k]['time']):.2f}",
             f"{RUN[k]['R']:.2f}" + ("" if RUN[k]["label"] == "static" else "*")] for k in sorted(keys, key=lambda k: RUN[k]["time"])]
    write_md(tab, "cs4_schedules.md", f"CS4 time vs schedule, pull, p = 8, median of {REPS} runs",
             ["schedule,chunk", "time (s)", "speedup vs serial", "measured R"], rows,
             "*R of dynamic/guided/edge is a sum over all iterations, so it can look balanced although each iteration waited "
             "for a straggler; judge these schedules by time. Only the chunk sizes you actually ran are shown "
             "(dynamic,256 and guided,64).")


# ------------------------------------------------------------------ 5. atomics
def fig_atomics(out, tab):
    d = {k: RUN[k]["time"] for k in ("pull_p1", "pushplain_p1", "push_p1")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    names = ["pull\n(no atomics)", "push, plain stores\n(1 thread, no atomics)", "push, atomic\n(1 thread)"]
    vals = [d["pull_p1"], d["pushplain_p1"], d["push_p1"]]
    a1.bar(names, vals, color=[BLUE, GREEN, RED])
    for i, v in enumerate(vals):
        a1.text(i, v, f"{v:.2f} s", ha="center", va="bottom", fontsize=9)
    ov = (d["push_p1"] / d["pushplain_p1"] - 1) * 100
    a1.set_ylabel("time (s), p = 1"); a1.set_ylim(0, max(vals) * 1.2)
    a1.set_title(f"Atomic instruction alone: +{ov:.0f}%  ({d['push_p1'] / d['pushplain_p1']:.1f}x)", fontsize=9)
    # right: push vs pull at p = 4 (contention)
    pu, ph = RUN["pull_static_p4"]["time"], RUN["push_static_p4"]["time"]
    a2.bar(["pull", "push (atomic)"], [pu, ph], color=[BLUE, RED])
    for i, v in enumerate([pu, ph]):
        a2.text(i, v, f"{v:.2f} s", ha="center", va="bottom", fontsize=9)
    a2.axhline(T_SERIAL, ls="--", color="black", lw=1); a2.text(0.5, T_SERIAL, " serial", fontsize=8, va="bottom", ha="center")
    a2.set_ylabel("time (s), schedule(static), p = 4"); a2.set_ylim(0, ph * 1.2)
    a2.set_title(f"With 4 threads: push is {ph / pu:.1f}x slower than pull", fontsize=9)
    fig.suptitle(f"CS4 cost of atomics; median of {REPS} runs", fontsize=10)
    save(fig, "cs4_atomics.png", out)
    write_md(tab, "cs4_atomics.md", f"CS4 pull vs push, median of {REPS} runs",
             ["variant", "threads", "time (s)", "relative"],
             [["pull", 1, f"{d['pull_p1']:.3f}", "1.00 (reference)"],
              ["push, plain stores (no atomics)", 1, f"{d['pushplain_p1']:.3f}", f"{d['pushplain_p1'] / d['pull_p1']:.2f}x pull"],
              ["push, atomic", 1, f"{d['push_p1']:.3f}", f"{d['push_p1'] / d['pull_p1']:.2f}x pull, {d['push_p1'] / d['pushplain_p1']:.2f}x plain push"],
              ["pull, static", 4, f"{pu:.3f}", "1.00 (reference)"],
              ["push, atomic, static", 4, f"{ph:.3f}", f"{ph / pu:.2f}x pull"]],
             f"Pure cost of the atomic instruction = +{ov:.0f}%. Extra slowdown at p = 4 comes from contention on hub vertices.")


# ------------------------------------------------------------------ 6. throughput / bandwidth
def fig_throughput(out, tab, peak):
    order = ["push_p1", "pushplain_p1", "pull_p1", "push_static_p4", "pull_static_p4", "pull_guided64_p8",
             "pull_static_p8", "pull_edge_p8", "pull_dyn256_p8"]
    nice = {"push_p1": "push atomic, p=1", "pushplain_p1": "push plain, p=1", "pull_p1": "pull static, p=1",
            "push_static_p4": "push atomic, p=4", "pull_static_p4": "pull static, p=4",
            "pull_guided64_p8": "pull guided,64, p=8", "pull_static_p8": "pull static, p=8",
            "pull_edge_p8": "pull edge-bal., p=8", "pull_dyn256_p8": "pull dynamic,256, p=8"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    y = np.arange(len(order))
    cols = [RED if "push" in k else BLUE for k in order]
    a1.barh(y, [RUN[k]["eps"] for k in order], color=cols)
    a1.axvline(EPS_SERIAL, ls="--", color="black", lw=1); a1.text(EPS_SERIAL, len(order) - 0.4, " serial", fontsize=8)
    a2.barh(y, [RUN[k]["gbps"] for k in order], color=cols)
    for i, k in enumerate(order):
        a1.text(RUN[k]["eps"], i, f" {RUN[k]['eps']:.0f}", va="center", fontsize=8)
        a2.text(RUN[k]["gbps"], i, f" {RUN[k]['gbps']:.1f}", va="center", fontsize=8)
    if peak:
        a2.axvline(peak, ls="--", color="black", lw=1.2); a2.text(peak, len(order) - 0.4, f" DRAM peak {peak:g} GB/s", fontsize=8)
        a2.set_xlim(0, max(peak, max(RUN[k]["gbps"] for k in order)) * 1.15)
    a1.set_yticks(y); a1.set_yticklabels([nice[k] for k in order])
    a1.set_xlabel("million edges processed per second"); a2.set_xlabel("effective memory traffic (GB/s, model estimate)")
    a1.set_xlim(0, 1300)
    fig.suptitle(f"CS4 throughput and memory traffic (blue = pull, red = push); {GRAPH}; median of {REPS} runs", fontsize=9)
    save(fig, "cs4_throughput_bandwidth.png", out)
    write_md(tab, "cs4_throughput.md", f"CS4 throughput (median of {REPS} runs)",
             ["configuration", "time (s)", "M edges/s", "GB/s (model)"],
             [[nice[k], f"{RUN[k]['time']:.3f}", f"{RUN[k]['eps']:.0f}", f"{RUN[k]['gbps']:.2f}"] for k in order],
             "GB/s = (12*m + 48*n bytes per iteration for pull, 20*m + 72*n for push) * iterations / time. It is a model, "
             "not a hardware counter reading.")


# ------------------------------------------------------------------ 7. degree distribution (needs 2 small csv files)
def fig_degree(out, degdir):
    pin, pout = os.path.join(degdir, "deg_in.csv"), os.path.join(degdir, "deg_out.csv")
    if not (os.path.exists(pin) and os.path.exists(pout)):
        print(f"  [skip] degree plot: {pin} / {pout} not found. Create them in 5 s with:  ./serial -r 1 -D results/deg")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, path, name, col in ((axes[0], pin, "in-degree", BLUE), (axes[1], pout, "out-degree", RED)):
        rows = list(csv.DictReader(open(path)))
        d = np.array([float(r["degree"]) for r in rows]); c = np.array([float(r["count"]) for r in rows])
        ax.loglog(d[d > 0], c[d > 0], ".", ms=4, color=col)
        mean = (d * c).sum() / c.sum(); sel = d >= 10
        alpha = 1 + c[sel].sum() / (c[sel] * np.log(d[sel] / 9.5)).sum()
        ax.set_xlabel(f"{name} (log scale)")
        ax.set_title(f"{name}: max = {int(d.max()):,}, mean = {mean:.1f}, zero-degree vertices = {int(c[d == 0].sum()):,}\n"
                     f"power-law exponent (degree >= 10) ~ {alpha:.2f}", fontsize=8)
    axes[0].set_ylabel("number of vertices (log scale)")
    fig.suptitle(f"CS4 degree distribution (log-log) - {GRAPH}", fontsize=10)
    save(fig, "cs4_degree_distribution.png", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "report", "figures"))
    ap.add_argument("--tables", default=os.path.join(HERE, "report", "tables"))
    ap.add_argument("--peak-bw", type=float, default=None, help="your machine's peak DRAM bandwidth in GB/s (optional)")
    ap.add_argument("--deg", default=os.path.join(HERE, "cs4_pagerank_openmp", "results"),
                    help="folder with deg_in.csv / deg_out.csv (optional)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.tables, exist_ok=True)
    print("Case Study 4 graphs (from the measured numbers pasted in this script)")
    fig_strong(a.out, a.tables)
    fig_per_thread(a.out)
    fig_imbalance(a.out)
    fig_schedules(a.out, a.tables)
    fig_atomics(a.out, a.tables)
    fig_throughput(a.out, a.tables, a.peak_bw)
    fig_degree(a.out, a.deg)
    print("Done.")


if __name__ == "__main__":
    main()
