/* parallel.c - Case Study 4: parallel PageRank with OpenMP.
 *
 * Usage: ./parallel [-t threads] [-a pull|push|pushplain] [-s static|dynamic|guided|edge]
 *                   [-c chunk] [-v] [common graph options, see pr_common.h]
 *
 * Algorithms (-a)
 *   pull       every thread OWNS a set of destination vertices v and reads the contributions of
 *              the in-neighbours of v:   r'(v) = base + d * sum_{u->v} contrib(u).
 *              Reads are random, writes are private -> no synchronisation, no atomics.
 *   push       every thread owns a set of SOURCE vertices u and adds contrib(u) to each
 *              out-neighbour: acc[v] += contrib(u).  Two threads may hit the same v, so every
 *              update is `#pragma omp atomic`.
 *   pushplain  the push loop WITHOUT atomics. Only correct with ONE thread; it exists to isolate
 *              the pure cost of the atomic instruction (compare with push at -t 1).
 *
 * Schedules (-s) for the main vertex loop
 *   static [-c c]   schedule(static[,c])  contiguous equal-VERTEX blocks (default) or round-robin chunks
 *   dynamic -c c    schedule(dynamic,c)   chunks handed out on demand
 *   guided  -c c    schedule(guided,c)    shrinking chunks, minimum size c
 *   edge            static partition balanced by EDGE count (extra experiment: shows that the
 *                   imbalance of `static` comes from unequal work per vertex, not from OpenMP)
 *
 * -v verifies the result against the serial implementation (L1 and max-abs difference).
 */
#include "pr_common.h"

#define PAD 8    /* per-thread timers padded to one 64-byte cache line (avoid false sharing) */

typedef enum { A_PULL, A_PUSH, A_PUSHPLAIN } Algo;
typedef enum { S_STATIC, S_DYNAMIC, S_GUIDED, S_EDGE } Sched;

static const char *algo_name(Algo a)  { return a == A_PULL ? "pull" : a == A_PUSH ? "push" : "pushplain"; }
static const char *sched_name(Sched s){ return s == S_STATIC ? "static" : s == S_DYNAMIC ? "dynamic" : s == S_GUIDED ? "guided" : "edge"; }

/* Boundaries so that each of p blocks gets ~equal (edges + vertices) of the given CSR. */
static uint32_t *edge_balanced_bounds(const CSR *c, int p)
{
    uint32_t *b = (uint32_t *)malloc(((size_t)p + 1) * sizeof(uint32_t));
    uint64_t total = c->m + c->n;                 /* +1 per vertex: loop overhead */
    uint32_t v = 0;
    b[0] = 0;
    for (int t = 1; t < p; t++) {
        uint64_t target = total * (uint64_t)t / (uint64_t)p;
        while (v < c->n && (c->off[v] + v) < target) v++;
        b[t] = v;
    }
    b[p] = c->n;
    return b;
}

/* Machine-independent imbalance of `schedule(static)`: edges that each thread would process when the
 * vertex range is cut into p equal blocks. Returns max/mean. */
static double static_work_ratio(const CSR *c, int p, double *max_share)
{
    double mx = 0, tot = 0;
    for (int t = 0; t < p; t++) {
        uint64_t lo = (uint64_t)c->n * t / p, hi = (uint64_t)c->n * (t + 1) / p;
        double w = (double)(c->off[hi] - c->off[lo]) + (double)(hi - lo);
        if (w > mx) mx = w;
        tot += w;
    }
    if (max_share) *max_share = mx / tot;
    return mx / (tot / p);
}

/* One full PageRank run. tth[t*PAD] receives the time thread t spent in the main edge loop,
 * summed over all iterations (used for the load-imbalance ratio R). */
static int pr_parallel(const Graph *g, Algo algo, Sched sch, int chunk, int p,
                       double d, double eps, int maxit, double *r_out, double *tth,
                       const uint32_t *ebound)
{
    const uint32_t n = g->n;
    double *cur = (double *)malloc((size_t)n * sizeof(double));
    double *nxt = (double *)malloc((size_t)n * sizeof(double));
    double *contrib = (double *)malloc((size_t)n * sizeof(double));
    for (int t = 0; t < p; t++) tth[(size_t)t * PAD] = 0.0;

    switch (sch) {
    case S_STATIC:  omp_set_schedule(omp_sched_static,  chunk); break;
    case S_DYNAMIC: omp_set_schedule(omp_sched_dynamic, chunk > 0 ? chunk : 1); break;
    case S_GUIDED:  omp_set_schedule(omp_sched_guided,  chunk > 0 ? chunk : 1); break;
    case S_EDGE:    break;
    }

    /* first-touch initialisation in parallel (helps NUMA machines) */
    #pragma omp parallel for num_threads(p) schedule(static)
    for (uint32_t v = 0; v < n; v++) { cur[v] = 1.0 / n; nxt[v] = 0.0; contrib[v] = 0.0; }

    const uint64_t *ooff = g->out.off;
    const uint32_t *oadj = g->out.adj;
    const uint64_t *ioff = g->in.off;
    const uint32_t *iadj = g->in.adj;

    int it = 0;
    while (it < maxit) {
        /* ---- phase 1: contribution of every vertex + total dangling mass ---- */
        double dangling = 0.0;
        #pragma omp parallel for num_threads(p) schedule(static) reduction(+:dangling)
        for (uint32_t u = 0; u < n; u++) {
            uint64_t deg = ooff[u + 1] - ooff[u];
            if (deg) contrib[u] = cur[u] / (double)deg;
            else   { contrib[u] = 0.0; dangling += cur[u]; }
        }
        const double base = (1.0 - d) / n + d * dangling / n;
        double diff = 0.0;

        if (algo == A_PULL) {
            /* ---- phase 2 (pull): thread owns destination vertices, reads in-neighbours ---- */
            if (sch == S_EDGE) {
                #pragma omp parallel num_threads(p) reduction(+:diff)
                {
                    int t = omp_get_thread_num();
                    double t0 = wtime();
                    for (uint32_t v = ebound[t]; v < ebound[t + 1]; v++) {
                        double s = 0.0;
                        for (uint64_t e = ioff[v]; e < ioff[v + 1]; e++) s += contrib[iadj[e]];
                        double x = base + d * s;
                        diff += fabs(x - cur[v]);
                        nxt[v] = x;
                    }
                    tth[(size_t)t * PAD] += wtime() - t0;
                }
            } else {
                #pragma omp parallel num_threads(p) reduction(+:diff)
                {
                    double t0 = wtime();
                    #pragma omp for schedule(runtime) nowait
                    for (uint32_t v = 0; v < n; v++) {
                        double s = 0.0;
                        for (uint64_t e = ioff[v]; e < ioff[v + 1]; e++) s += contrib[iadj[e]];
                        double x = base + d * s;
                        diff += fabs(x - cur[v]);
                        nxt[v] = x;
                    }
                    tth[(size_t)omp_get_thread_num() * PAD] += wtime() - t0;
                }
            }
        } else {
            /* ---- phase 2a (push): clear the accumulator ---- */
            #pragma omp parallel for num_threads(p) schedule(static)
            for (uint32_t v = 0; v < n; v++) nxt[v] = 0.0;

            /* ---- phase 2b (push): thread owns source vertices, scatters into out-neighbours ---- */
            if (algo == A_PUSHPLAIN) {                       /* single thread, no atomics */
                double t0 = wtime();
                for (uint32_t u = 0; u < n; u++) {
                    double c = contrib[u];
                    for (uint64_t e = ooff[u]; e < ooff[u + 1]; e++) nxt[oadj[e]] += c;
                }
                tth[0] += wtime() - t0;
            } else if (sch == S_EDGE) {
                #pragma omp parallel num_threads(p)
                {
                    int t = omp_get_thread_num();
                    double t0 = wtime();
                    for (uint32_t u = ebound[t]; u < ebound[t + 1]; u++) {
                        double c = contrib[u];
                        for (uint64_t e = ooff[u]; e < ooff[u + 1]; e++) {
                            #pragma omp atomic
                            nxt[oadj[e]] += c;
                        }
                    }
                    tth[(size_t)t * PAD] += wtime() - t0;
                }
            } else {
                #pragma omp parallel num_threads(p)
                {
                    double t0 = wtime();
                    #pragma omp for schedule(runtime) nowait
                    for (uint32_t u = 0; u < n; u++) {
                        double c = contrib[u];
                        for (uint64_t e = ooff[u]; e < ooff[u + 1]; e++) {
                            #pragma omp atomic
                            nxt[oadj[e]] += c;
                        }
                    }
                    tth[(size_t)omp_get_thread_num() * PAD] += wtime() - t0;
                }
            }
            /* ---- phase 2c (push): apply damping + convergence metric ---- */
            #pragma omp parallel for num_threads(p) schedule(static) reduction(+:diff)
            for (uint32_t v = 0; v < n; v++) {
                double x = base + d * nxt[v];
                diff += fabs(x - cur[v]);
                nxt[v] = x;
            }
        }

        double *tmp = cur; cur = nxt; nxt = tmp;
        it++;
        if (diff < eps) break;
    }
    memcpy(r_out, cur, (size_t)n * sizeof(double));
    free(cur); free(nxt); free(contrib);
    return it;
}

int main(int argc, char **argv)
{
    Opts o; opts_default(&o);
    int p = omp_get_max_threads(), chunk = 0, verify = 0;
    Algo algo = A_PULL; Sched sch = S_STATIC;

    for (int i = 1; i < argc; i++) {
        if (parse_common(&o, argc, argv, &i)) continue;
        if      (!strcmp(argv[i], "-t") && i + 1 < argc) p = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verify = 1;
        else if (!strcmp(argv[i], "-a") && i + 1 < argc) {
            const char *a = argv[++i];
            if      (!strcmp(a, "pull"))      algo = A_PULL;
            else if (!strcmp(a, "push"))      algo = A_PUSH;
            else if (!strcmp(a, "pushplain")) algo = A_PUSHPLAIN;
            else { fprintf(stderr, "unknown algorithm %s\n", a); return 1; }
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            const char *s = argv[++i];
            if      (!strcmp(s, "static"))  sch = S_STATIC;
            else if (!strcmp(s, "dynamic")) sch = S_DYNAMIC;
            else if (!strcmp(s, "guided"))  sch = S_GUIDED;
            else if (!strcmp(s, "edge"))    sch = S_EDGE;
            else { fprintf(stderr, "unknown schedule %s\n", s); return 1; }
        } else {
            fprintf(stderr, "usage: %s [-t p] [-a pull|push|pushplain] [-s static|dynamic|guided|edge] "
                            "[-c chunk] [-v] " COMMON_USAGE "\n", argv[0]);
            return 1;
        }
    }
    if (p < 1) p = 1;
    if (o.reps < 1) o.reps = 1;
    if (algo == A_PUSHPLAIN && p != 1) {
        fprintf(stderr, "note: pushplain is only correct with 1 thread; forcing -t 1\n");
        p = 1;
    }
    omp_set_dynamic(0);

    Graph g;
    load_graph(&o, &g);
    if (o.degdump) dump_degree_hist(&g, o.degdump);

    uint32_t *eb = NULL;
    if (sch == S_EDGE) eb = edge_balanced_bounds(algo == A_PULL ? &g.in : &g.out, p);

    double *r = (double *)malloc((size_t)g.n * sizeof(double));
    double *tth = (double *)calloc((size_t)p * PAD, sizeof(double));
    double *times = (double *)malloc((size_t)o.reps * sizeof(double));
    double R_imb = 1.0;
    int iters = 0;

    for (int rep = 0; rep < o.reps; rep++) {
        double t0 = wtime();
        iters = pr_parallel(&g, algo, sch, chunk, p, o.damping, o.eps, o.maxit, r, tth, eb);
        times[rep] = wtime() - t0;
        double mx = 0, sum = 0;
        for (int t = 0; t < p; t++) { double x = tth[(size_t)t * PAD]; if (x > mx) mx = x; sum += x; }
        R_imb = (sum > 0) ? mx / (sum / p) : 1.0;            /* slowest thread / mean thread */
    }
    double tmed = median_d(times, o.reps), tmin = min_d(times, o.reps);
    double eps_s = (double)g.m * iters / tmed;
    double bytes_it = (algo == A_PULL) ? (double)(12ULL * g.m + 48ULL * g.n)    /* see report, Q4 */
                                       : (double)(20ULL * g.m + 72ULL * g.n);
    double gbps = bytes_it * iters / tmed / 1e9;
    double work_R = static_work_ratio(algo == A_PULL ? &g.in : &g.out, p, NULL);

    double l1 = 0, linf = 0, sum_r = 0;
    for (uint32_t v = 0; v < g.n; v++) sum_r += r[v];
    if (verify) {
        double *rs = (double *)malloc((size_t)g.n * sizeof(double));
        int it_s = pr_serial(&g, o.damping, o.eps, o.maxit, rs);
        for (uint32_t v = 0; v < g.n; v++) {
            double df = fabs(r[v] - rs[v]);
            l1 += df; if (df > linf) linf = df;
        }
        if (!o.csv)
            printf("Verification vs serial: serial iters=%d  L1 diff=%.3e  max|diff|=%.3e  -> %s (tol 1e-7)\n",
                   it_s, l1, linf, l1 < 1e-7 ? "PASS" : "FAIL");
        free(rs);
    }

    if (o.csv) {
        /* algo,sched,chunk,threads,n,m,iters,t_median,t_min,edges_per_s,gbps_model,imbalance_R,work_R,verify_L1 */
        printf("CSV,%s,%s,%d,%d,%u,%llu,%d,%.6f,%.6f,%.0f,%.3f,%.3f,%.3f,%.3e\n",
               algo_name(algo), sched_name(sch), chunk, p, g.n, (unsigned long long)g.m, iters,
               tmed, tmin, eps_s, gbps, R_imb, work_R, l1);
        return 0;
    }

    printf("\n===== PARALLEL PAGERANK (OpenMP) =====\n");
    printf("Algorithm / schedule  : %s / %s%s  chunk=%d\n", algo_name(algo), sched_name(sch),
           sch == S_STATIC && chunk == 0 ? " (block)" : "", chunk);
    printf("Threads               : %d\n", p);
    printf("Iterations            : %d (eps=%g)   sum of ranks=%.12f\n", iters, o.eps, sum_r);
    printf("Time, median of %d    : %.4f s  (min %.4f s)\n", o.reps, tmed, tmin);
    printf("Throughput            : %.1f M edges/s   ~%.2f GB/s (model estimate)\n", eps_s / 1e6, gbps);
    printf("Load imbalance R      : %.3f measured (slowest/mean thread time), %.3f from static-work model\n",
           R_imb, work_R);
    printf("Per-thread loop time  :");
    for (int t = 0; t < p; t++) printf(" %.3f", tth[(size_t)t * PAD]);
    printf(" s\n");
    free(r); free(tth); free(times); free(eb);
    return 0;
}
