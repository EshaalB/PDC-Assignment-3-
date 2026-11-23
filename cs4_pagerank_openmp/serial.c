/* serial.c - Case Study 4 serial baseline: PageRank on a CSR graph (pull formulation).
 *
 * Usage: ./serial [-scale S] [-ef F] [-seed X] [-perm] [-f snap.txt] [-r reps] [-e eps]
 *                 [-i maxit] [-D degprefix] [-o ranks.bin] [--csv]
 * Only the PageRank iterations are timed (graph construction is excluded).
 */
#include "pr_common.h"

int main(int argc, char **argv)
{
    Opts o; opts_default(&o);
    const char *outfile = NULL;
    for (int i = 1; i < argc; i++) {
        if (parse_common(&o, argc, argv, &i)) continue;
        if (!strcmp(argv[i], "-o") && i + 1 < argc) outfile = argv[++i];
        else { fprintf(stderr, "usage: %s " COMMON_USAGE " [-o ranks.bin]\n", argv[0]); return 1; }
    }
    if (o.reps < 1) o.reps = 1;

    Graph g;
    load_graph(&o, &g);
    if (o.degdump) dump_degree_hist(&g, o.degdump);

    double *r = (double *)malloc((size_t)g.n * sizeof(double));
    double *times = (double *)malloc((size_t)o.reps * sizeof(double));
    int iters = 0;
    for (int rep = 0; rep < o.reps; rep++) {
        double t0 = wtime();
        iters = pr_serial(&g, o.damping, o.eps, o.maxit, r);
        times[rep] = wtime() - t0;
    }
    double tmed = median_d(times, o.reps), tmin = min_d(times, o.reps);
    double sum = 0; for (uint32_t v = 0; v < g.n; v++) sum += r[v];
    double eps_s = (double)g.m * iters / tmed;

    if (outfile) { FILE *f = fopen(outfile, "wb"); if (f) { fwrite(r, sizeof(double), g.n, f); fclose(f); } }

    if (o.csv) {
        /* algo,sched,chunk,threads,n,m,iters,t_median,t_min,edges_per_s,gbps_model,imbalance_R,work_R,verify_L1 */
        double gbps = ((double)(12ULL * g.m + 48ULL * g.n) * iters) / tmed / 1e9;
        printf("CSV,serial,-,0,1,%u,%llu,%d,%.6f,%.6f,%.0f,%.3f,1.000,1.000,0\n",
               g.n, (unsigned long long)g.m, iters, tmed, tmin, eps_s, gbps);
        return 0;
    }
    printf("\n===== SERIAL PAGERANK =====\n");
    printf("Iterations            : %d (eps=%g)\n", iters, o.eps);
    printf("Sum of ranks          : %.12f (should be 1)\n", sum);
    printf("Time, median of %d    : %.4f s  (min %.4f s)\n", o.reps, tmed, tmin);
    printf("Throughput            : %.1f M edges/s\n", eps_s / 1e6);
    return 0;
}
