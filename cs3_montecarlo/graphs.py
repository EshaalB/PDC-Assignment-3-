#!/usr/bin/env python3
"""
graphs.py - Case Study 3: Monte Carlo Option Pricing (OpenMP).

Uses only the measurements already collected from the terminal; it does not
rerun the Monte Carlo programs.

Run from this directory:
    python3 graphs.py

Dependencies:
    python3 -m pip install numpy matplotlib

Outputs:
    report/figures/cs3_rng_performance.png
    report/figures/cs3_price_accuracy.png
    report/figures/cs3_standard_error.png
    report/figures/cs3_strong_scaling.png
    report/tables/cs3_rng_comparison.md
    report/tables/cs3_reproducibility.md
    report/tables/cs3_strong_scaling.md
    report/tables/cs3_timed.md

Note:
The 2M/4M diagnostic runs document the RNG-mode comparison; the M = 100,000,000
strong-scaling and -m timed (-T 5) runs were measured on the Intel Core i7-1255U
(2 P-cores + 8 E-cores, Fedora, gcc -O3, OMP_PROC_BIND=close OMP_PLACES=cores).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "report", "figures")
TAB = os.path.join(HERE, "report", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False
})
BLUE, RED, GREEN, ORANGE, GREY, PURPLE = (
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#7f7f7f", "#9467bd"
)

# Option parameters from mc_common.h / README.
BS_PRICE = 10.45058357
S0, K, R, SIGMA, T = 100, 100, 0.05, 0.20, 1.0
SEED = 42

# Exact measurements copied from the supplied terminal output.
# The 2M-path runs are a diagnostic comparison of RNG modes.
RNG_RUNS = [
    dict(mode="perthread", M=2_000_000, p=4, time=0.0408, throughput=49.01,
         price=10.43233905, se=0.01039680, abs_error=0.01824453,
         z=1.75, reproducible="yes"),
    dict(mode="shared", M=2_000_000, p=4, time=0.2601, throughput=7.69,
         price=10.44913179, se=0.01040647, abs_error=0.00145178,
         z=0.14, reproducible="NO"),
    dict(mode="sameseed", M=2_000_000, p=4, time=0.0477, throughput=41.90,
         price=10.43211896, se=0.01038903, abs_error=0.01846461,
         z=1.78, reproducible="yes"),
]

# Separate 4M-path per-thread run; do not mix it into the 2M RNG timing chart.
PERTHREAD_4M = dict(mode="perthread", M=4_000_000, p=4, time=0.0586,
                    throughput=68.26, price=10.43616894, se=0.00735443,
                    abs_error=0.01441463, z=1.96, reproducible="yes")

# Reproducibility checks from --csv output. These hex values are preserved
# literally because equality is the point of this comparison.
REPRO = [
    ["serial, M=4M", "1", "per-thread serial stream",
     "0x1.4e26e7b5ac647p+3", "Matches parallel p=1"],
    ["parallel, M=4M", "1", "perthread",
     "0x1.4e26e7b5ac647p+3", "Bit-identical to serial"],
    ["parallel, M=4M", "1", "blocks",
     "0x1.4e44bc1954d83p+3", "Same blocks result as p=3 and p=7"],
    ["parallel, M=4M", "3", "blocks",
     "0x1.4e44bc1954d83p+3", "Bit-identical across thread counts"],
    ["parallel, M=4M", "7", "blocks",
     "0x1.4e44bc1954d83p+3", "Bit-identical across thread counts"],
    ["parallel, M=2M", "4", "shared",
     "varies between repetitions", "Not reproducible (reported NO)"],
    ["parallel, M=2M", "4", "sameseed",
     "0x1.4dd3eb261a4d3p+3", "Repeatable, but streams are correlated"],
]

# Strong scaling (M = 100,000,000, seed = 42, median of 3 runs,
# Intel Core i7-1255U, OMP_PROC_BIND=close OMP_PLACES=cores).
T_SERIAL_100M = 1.656829
STRONG_100M = [
    dict(p=1, time=1.590752, price=10.45047924, se=0.00147196, mpaths_s=100.0 / 1.590752),
    dict(p=2, time=0.957298, price=10.44942835, se=0.00147190, mpaths_s=100.0 / 0.957298),
    dict(p=4, time=0.767562, price=10.44985081, se=0.00147202, mpaths_s=100.0 / 0.767562),
    dict(p=8, time=0.383659, price=10.45185139, se=0.00147210, mpaths_s=100.0 / 0.383659),
]

# Fixed 5-second time budget (-m timed -T 5, seed = 42).
TIMED_5S = [
    dict(p=1, M=318_636_032, time=5.000918, price=10.45039541, se=0.00082457, abs_err=0.00018817, z=0.23),
    dict(p=2, M=544_604_160, time=5.000317, price=10.44996541, se=0.00063068, abs_err=0.00061817, z=0.98),
    dict(p=4, M=875_233_280, time=5.000973, price=10.45014605, se=0.00049753, abs_err=0.00043752, z=0.88),
    dict(p=8, M=1_519_124_480, time=5.001917, price=10.45037976, se=0.00037764, abs_err=0.00020381, z=0.54),
]


def save(fig, filename):
    path = os.path.join(FIG, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def write_md(filename, title, headers, rows, note=None):
    path = os.path.join(TAB, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"**{title}**\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
        if note:
            f.write("\n*" + note + "*\n")
    print("wrote", os.path.relpath(path, HERE))


def fig_rng_performance():
    names = ["perthread", "shared", "sameseed"]
    times = [x["time"] for x in RNG_RUNS]
    rates = [x["throughput"] for x in RNG_RUNS]
    colors = [BLUE, RED, ORANGE]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    bars = a1.bar(names, times, color=colors)
    for b, v in zip(bars, times):
        a1.text(b.get_x() + b.get_width()/2, v, f"{v:.4f} s",
                ha="center", va="bottom", fontsize=9)
    a1.set_ylabel("median execution time (s)")
    a1.set_title("Runtime by RNG mode (2M paths, 4 threads)")
    a1.set_ylim(0, max(times) * 1.22)

    bars = a2.bar(names, rates, color=colors)
    for b, v in zip(bars, rates):
        a2.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("throughput (million paths/s)")
    a2.set_title("Monte Carlo throughput (2M paths, 4 threads)")
    a2.set_ylim(0, max(rates) * 1.22)

    fig.suptitle("CS3 RNG-mode comparison — same workload (M = 2,000,000)")
    save(fig, "cs3_rng_performance.png")

    rows = []
    for x in RNG_RUNS:
        rows.append([
            x["mode"], f'{x["time"]:.4f}', f'{x["throughput"]:.2f}',
            f'{x["price"]:.8f}', f'{x["se"]:.8f}',
            f'{x["abs_error"]:.8f}', f'{x["z"]:.2f}', x["reproducible"]
        ])
    write_md(
        "cs3_rng_comparison.md",
        "CS3 RNG modes (M = 2,000,000, p = 4, seed = 42)",
        ["RNG mode", "time (s)", "Mpaths/s", "MC price", "SE",
         "|MC − BS|", "error / SE", "repeatable"],
        rows,
        "The shared generator is protected by a lock, so its runtime is much "
        "higher. The same-seed mode repeats correlated samples and therefore "
        "does not provide independent worker streams. A small error in one "
        "run does not make that RNG mode statistically valid."
    )


def fig_price_accuracy():
    labels = ["perthread\n2M", "shared\n2M", "sameseed\n2M", "perthread\n4M"]
    runs = RNG_RUNS + [PERTHREAD_4M]
    prices = [x["price"] for x in runs]
    errors = [x["se"] for x in runs]
    x = np.arange(len(runs))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(BS_PRICE, color="black", linestyle="--", linewidth=1.3,
               label=f"Black–Scholes = {BS_PRICE:.8f}")
    ax.errorbar(x, prices, yerr=[np.array(errors) * 1.96,
                                 np.array(errors) * 1.96],
                fmt="o", capsize=5, color=BLUE,
                label="Monte Carlo estimate ± 1.96 SE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("option price")
    ax.set_title("Monte Carlo estimates against Black–Scholes")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "cs3_price_accuracy.png")


def fig_standard_error():
    # Two measured perthread observations, shown as observations rather than
    # a fitted curve. The theoretical M^-1/2 relationship is a reference line.
    Ms = np.array([2_000_000, 4_000_000], dtype=float)
    ses = np.array([RNG_RUNS[0]["se"], PERTHREAD_4M["se"]], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(Ms, ses, "o-", color=BLUE, label="measured perthread SE")
    reference = ses[0] * np.sqrt(Ms[0] / Ms)
    ax.loglog(Ms, reference, "--", color=GREY,
              label=r"reference: SE $\propto 1/\sqrt{M}$")
    ax.set_xlabel("number of Monte Carlo paths (M)")
    ax.set_ylabel("standard error")
    ax.set_title("Standard error versus number of paths")
    ax.legend(frameon=False)
    save(fig, "cs3_standard_error.png")


def fig_strong_scaling():
    ps = np.array([x["p"] for x in STRONG_100M], dtype=float)
    ts = np.array([x["time"] for x in STRONG_100M], dtype=float)
    speedup = T_SERIAL_100M / ts
    eff = speedup / ps

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    p_ideal = np.arange(1, 9)
    a1.plot(p_ideal, p_ideal, "k--", lw=1, label="ideal")
    a1.plot(ps, speedup, "o-", color=BLUE, label="perthread (static split)")
    for pi, si in zip(ps, speedup):
        if int(pi) == 2:
            a1.text(pi + 0.15, si - 0.55, f"{si:.2f}x", ha="left", va="top", fontsize=8)
        else:
            a1.text(pi, si + 0.2, f"{si:.2f}x", ha="center", va="bottom", fontsize=8)
    a1.set_xlabel("OpenMP threads p")
    a1.set_ylabel("speedup  S(p) = T_serial / T_p")
    a1.set_title("Strong scaling (M = 100,000,000)")
    a1.set_xticks([1, 2, 4, 8])
    a1.set_ylim(0, 8.6)
    a1.legend(frameon=False, fontsize=8)

    a2.axhline(1.0, ls="--", color="black", lw=1, label="ideal")
    a2.axhline(0.7, ls=":", color=GREY, lw=1)
    a2.plot(ps, eff, "o-", color=BLUE, label="perthread (static split)")
    for pi, ei in zip(ps, eff):
        a2.text(pi, ei + 0.03, f"{ei:.2f}", ha="center", va="bottom", fontsize=8)
    a2.set_xlabel("OpenMP threads p")
    a2.set_ylabel("efficiency  E(p) = S(p) / p")
    a2.set_title("Parallel efficiency (M = 100,000,000)")
    a2.set_xticks([1, 2, 4, 8])
    a2.set_ylim(0, 1.18)
    a2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"CS3 Monte Carlo strong scaling (M = 100,000,000, serial T = {T_SERIAL_100M:.3f} s; median of 3 runs)")
    save(fig, "cs3_strong_scaling.png")

    rows = [["serial", "1", f"{T_SERIAL_100M:.4f}", f"{100.0 / T_SERIAL_100M:.2f}", "1.00", "1.00"]]
    for x in STRONG_100M:
        s = T_SERIAL_100M / x["time"]
        rows.append([
            "perthread", str(x["p"]), f'{x["time"]:.4f}', f'{x["mpaths_s"]:.2f}',
            f"{s:.2f}", f'{s / x["p"]:.2f}'
        ])
    write_md(
        "cs3_strong_scaling.md",
        "CS3 strong scaling (M = 100,000,000, seed = 42, median of 3 runs, Intel Core i7-1255U)",
        ["mode", "p", "T (s)", "Mpaths/s", "S(p)", "E(p)"],
        rows,
        "Measured with OMP_PROC_BIND=close OMP_PLACES=cores on Intel Core i7-1255U (2 P-cores + 8 E-cores). "
        "At p = 4 and p = 8, fast P-cores wait for slower E-cores under static equal splitting."
    )

    trows = []
    for x in TIMED_5S:
        rate = x["M"] / x["time"] / 1e6
        trows.append([
            str(x["p"]), f'{x["M"]:,}', f"{rate:.2f}", f'{x["price"]:.6f}',
            f'{x["se"]:.6f}', f'{x["abs_err"]:.6f}', f'{x["z"]:.2f}'
        ])
    write_md(
        "cs3_timed.md",
        "CS3 fixed time budget (-m timed -T 5, seed = 42, Intel Core i7-1255U)",
        ["p", "paths M", "Mpaths/s", "MC price", "SE", "|MC - BS|", "error / SE"],
        trows,
        "Increasing threads from 1 to 8 raises simulated paths 4.77x in 5 s and lowers SE by 2.18x (matching sqrt(4.77))."
    )


def main():
    print("CS3 Monte Carlo plots (using supplied terminal measurements)")
    print(f"Black–Scholes reference price: {BS_PRICE:.8f}")
    fig_rng_performance()
    fig_price_accuracy()
    fig_standard_error()
    fig_strong_scaling()
    write_md(
        "cs3_reproducibility.md",
        "CS3 reproducibility observations",
        ["run", "threads", "mode", "hex-float price", "observation"],
        REPRO,
        "The blocks mode is reproducible across p = 1, 3, and 7 because each "
        "fixed block owns a deterministic RNG stream and results are combined "
        "in a fixed order. Per-thread mode assigns streams according to p, so "
        "changing p changes the samples and can change the final estimate. "
        "Floating-point addition is also non-associative."
    )
    print("\nDone. Figures and Markdown tables are in report/.")


if __name__ == "__main__":
    main()
