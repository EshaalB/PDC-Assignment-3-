/* parallel.c - Case Study 3: parallel Monte Carlo European call pricing (OpenMP).
 *
 * Usage: ./parallel [-n M] [-t threads] [-s seed] [-r reps] [-m mode]
 *                   [-B blocks] [-T seconds] [--csv]
 *
 * Modes (-m):
 *   perthread  (default, THE solution) one PCG32 stream per thread, static split of the
 *              paths, partial sums combined in a FIXED order (thread 0,1,2,...).
 *              Reproducible for the same (seed, p).
 *   blocks     the paths are cut into B fixed blocks (default 4096); block b always uses
 *              stream b, partial sums combined in block order. The result is
 *              BIT-IDENTICAL for every thread count p.
 *   timed      like perthread but every thread simulates until -T seconds have elapsed
 *              (fixed-TIME experiment: how much accuracy do p threads buy per second?).
 *   shared     BAD ON PURPOSE: one global generator protected by a critical section.
 *              Statistically valid, but the lock serialises the program.
 *   sameseed   BAD ON PURPOSE: every thread uses the SAME (seed, stream). Threads simulate
 *              identical paths; the reported standard error is wrong (too small).
 *
 * Output: human-readable by default, one "CSV,..." line with --csv.
 */
#include "mc_common.h"
#include <omp.h>

#define PAD 8   /* 8 doubles = 64 bytes: one cache line per thread slot (no false sharing) */

typedef enum { M_PERTHREAD, M_BLOCKS, M_TIMED, M_SHARED, M_SAMESEED } Mode;

static const char *mode_name(Mode m)
{
    switch (m) {
    case M_PERTHREAD: return "perthread";
    case M_BLOCKS:    return "blocks";
    case M_TIMED:     return "timed";
    case M_SHARED:    return "shared";
    default:          return "sameseed";
    }
}

/* Fixed-order combination of per-worker partial results: this is what removes the
 * run-to-run non-determinism of an `omp reduction` (whose combination order is
 * unspecified). Floating-point addition is not associative, so ORDER MATTERS. */
static void ordered_sum(const double *s, const double *q, int n, int stride,
                        double *sum, double *sumsq)
{
    double a = 0.0, b = 0.0;
    for (int i = 0; i < n; i++) { a += s[(size_t)i * stride]; b += q[(size_t)i * stride]; }
    *sum = a; *sumsq = b;
}

/* ---- Mode: perthread --------------------------------------------------- */
static void run_perthread(uint64_t M, uint64_t seed, int p, double *sum, double *sumsq)
{
    uint64_t npairs = M / 2;
    double *ps = (double *)calloc((size_t)p * PAD, sizeof(double));
    double *pq = (double *)calloc((size_t)p * PAD, sizeof(double));
    int nt_used = p;

    #pragma omp parallel num_threads(p)
    {
        int t  = omp_get_thread_num();
        int nt = omp_get_num_threads();
        #pragma omp single
        nt_used = nt;

        uint64_t lo = npairs * (uint64_t)t / (uint64_t)nt;
        uint64_t hi = npairs * (uint64_t)(t + 1) / (uint64_t)nt;

        PCG32 rng;
        pcg32_seed(&rng, seed, (uint64_t)t + 1ULL);   /* independent stream per thread */
        mc_accumulate(&rng, hi - lo, &ps[(size_t)t * PAD], &pq[(size_t)t * PAD]);
    }
    ordered_sum(ps, pq, nt_used, PAD, sum, sumsq);
    free(ps); free(pq);
}

/* ---- Mode: blocks (result independent of p) ---------------------------- */
static void run_blocks(uint64_t M, uint64_t seed, int p, int B, double *sum, double *sumsq)
{
    uint64_t npairs = M / 2;
    double *bs = (double *)malloc((size_t)B * sizeof(double));
    double *bq = (double *)malloc((size_t)B * sizeof(double));

    #pragma omp parallel for num_threads(p) schedule(static)
    for (int b = 0; b < B; b++) {
        uint64_t lo = npairs * (uint64_t)b / (uint64_t)B;
        uint64_t hi = npairs * (uint64_t)(b + 1) / (uint64_t)B;
        PCG32 rng;
        pcg32_seed(&rng, seed, (uint64_t)b + 1ULL);   /* stream depends on BLOCK, not thread */
        mc_accumulate(&rng, hi - lo, &bs[b], &bq[b]);
    }
    ordered_sum(bs, bq, B, 1, sum, sumsq);
    free(bs); free(bq);
}

/* ---- Mode: shared (single generator + lock): demonstrates serialisation -- */
static void run_shared(uint64_t M, uint64_t seed, int p, double *sum, double *sumsq)
{
    uint64_t npairs = M / 2;
    const double drift = (R - 0.5 * SIGMA * SIGMA) * T;
    const double vol   = SIGMA * sqrt(T);
    double *ps = (double *)calloc((size_t)p * PAD, sizeof(double));
    double *pq = (double *)calloc((size_t)p * PAD, sizeof(double));
    int nt_used = p;
    PCG32 shared_rng;
    pcg32_seed(&shared_rng, seed, 1ULL);

    #pragma omp parallel num_threads(p)
    {
        int t  = omp_get_thread_num();
        int nt = omp_get_num_threads();
        #pragma omp single
        nt_used = nt;
        uint64_t lo = npairs * (uint64_t)t / (uint64_t)nt;
        uint64_t hi = npairs * (uint64_t)(t + 1) / (uint64_t)nt;
        double s = 0.0, q = 0.0;
        for (uint64_t i = lo; i < hi; i++) {
            double z0, z1;
            #pragma omp critical(rng_lock)
            { normal_pair(&shared_rng, &z0, &z1); }
            double p0 = payoff_from_z(z0, drift, vol);
            double p1 = payoff_from_z(z1, drift, vol);
            s += p0 + p1;
            q += p0 * p0 + p1 * p1;
        }
        ps[(size_t)t * PAD] = s;
        pq[(size_t)t * PAD] = q;
    }
    ordered_sum(ps, pq, nt_used, PAD, sum, sumsq);
    free(ps); free(pq);
}

/* ---- Mode: sameseed (correlated streams): demonstrates wrong std. error -- */
static void run_sameseed(uint64_t M, uint64_t seed, int p, double *sum, double *sumsq)
{
    uint64_t npairs = M / 2;
    double *ps = (double *)calloc((size_t)p * PAD, sizeof(double));
    double *pq = (double *)calloc((size_t)p * PAD, sizeof(double));
    int nt_used = p;

    #pragma omp parallel num_threads(p)
    {
        int t  = omp_get_thread_num();
        int nt = omp_get_num_threads();
        #pragma omp single
        nt_used = nt;
        uint64_t lo = npairs * (uint64_t)t / (uint64_t)nt;
        uint64_t hi = npairs * (uint64_t)(t + 1) / (uint64_t)nt;
        PCG32 rng;
        pcg32_seed(&rng, seed, 1ULL);                 /* SAME stream in every thread (the bug) */
        mc_accumulate(&rng, hi - lo, &ps[(size_t)t * PAD], &pq[(size_t)t * PAD]);
    }
    ordered_sum(ps, pq, nt_used, PAD, sum, sumsq);
    free(ps); free(pq);
}

/* ---- Mode: timed (fixed wall-clock budget) ----------------------------- */
static uint64_t run_timed(double seconds, uint64_t seed, int p, double *sum, double *sumsq)
{
    const uint64_t BATCH = 1ULL << 15;                 /* pairs per time check */
    double *ps = (double *)calloc((size_t)p * PAD, sizeof(double));
    double *pq = (double *)calloc((size_t)p * PAD, sizeof(double));
    uint64_t *cnt = (uint64_t *)calloc((size_t)p * PAD, sizeof(uint64_t));
    int nt_used = p;
    double t_start = omp_get_wtime();

    #pragma omp parallel num_threads(p)
    {
        int t = omp_get_thread_num();
        #pragma omp single
        nt_used = omp_get_num_threads();
        PCG32 rng;
        pcg32_seed(&rng, seed, (uint64_t)t + 1ULL);
        double s = 0.0, q = 0.0;
        uint64_t pairs = 0;
        while (omp_get_wtime() - t_start < seconds) {
            double bs, bq;
            mc_accumulate(&rng, BATCH, &bs, &bq);
            s += bs; q += bq; pairs += BATCH;
        }
        ps[(size_t)t * PAD] = s;
        pq[(size_t)t * PAD] = q;
        cnt[(size_t)t * PAD] = pairs;
    }
    ordered_sum(ps, pq, nt_used, PAD, sum, sumsq);
    uint64_t total_pairs = 0;
    for (int i = 0; i < nt_used; i++) total_pairs += cnt[(size_t)i * PAD];
    free(ps); free(pq); free(cnt);
    return 2 * total_pairs;
}

int main(int argc, char **argv)
{
    uint64_t M = DEFAULT_M, seed = DEFAULT_SEED;
    int p = omp_get_max_threads(), reps = 3, csv = 0, B = 4096;
    double tbudget = 5.0;
    Mode mode = M_PERTHREAD;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc)      M = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) p = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B") && i + 1 < argc) B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) tbudget = atof(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            const char *m = argv[++i];
            if      (!strcmp(m, "perthread")) mode = M_PERTHREAD;
            else if (!strcmp(m, "blocks"))    mode = M_BLOCKS;
            else if (!strcmp(m, "timed"))     mode = M_TIMED;
            else if (!strcmp(m, "shared"))    mode = M_SHARED;
            else if (!strcmp(m, "sameseed"))  mode = M_SAMESEED;
            else { fprintf(stderr, "unknown mode %s\n", m); return 1; }
        }
        else if (!strcmp(argv[i], "--csv")) csv = 1;
        else {
            fprintf(stderr, "usage: %s [-n M] [-t p] [-s seed] [-r reps] "
                            "[-m perthread|blocks|timed|shared|sameseed] [-B blocks] [-T sec] [--csv]\n",
                    argv[0]);
            return 1;
        }
    }
    if (p < 1) p = 1;
    if (reps < 1) reps = 1;
    if (mode == M_TIMED) reps = 1;
    if (B < 1) B = 1;
    M &= ~1ULL;
    omp_set_dynamic(0);                      /* give us exactly p threads */

    double *times = (double *)malloc((size_t)reps * sizeof(double));
    double price = 0, se = 0, first_price = 0;
    int reproducible = 1;
    uint64_t M_actual = M;

    for (int rep = 0; rep < reps; rep++) {
        double sum = 0, sumsq = 0;
        double t0 = wall_time();
        switch (mode) {
        case M_PERTHREAD: run_perthread(M, seed, p, &sum, &sumsq); break;
        case M_BLOCKS:    run_blocks(M, seed, p, B, &sum, &sumsq); break;
        case M_SHARED:    run_shared(M, seed, p, &sum, &sumsq);    break;
        case M_SAMESEED:  run_sameseed(M, seed, p, &sum, &sumsq);  break;
        case M_TIMED:     M_actual = run_timed(tbudget, seed, p, &sum, &sumsq); break;
        }
        times[rep] = wall_time() - t0;
        mc_finalize(sum, sumsq, M_actual, &price, &se);
        if (rep == 0) first_price = price;
        else if (memcmp(&price, &first_price, sizeof(double)) != 0) reproducible = 0;
    }

    double bs = black_scholes_price();
    double err = fabs(price - bs);
    double tmed = median_of(times, reps), tmin = min_of(times, reps);

    if (csv) {
        /* mode,threads,M,seed,reps,t_median,t_min,price,se,bs,abs_err,z,repro,hexprice */
        printf("CSV,%s,%d,%llu,%llu,%d,%.6f,%.6f,%.10f,%.10f,%.10f,%.10f,%.4f,%d,%a\n",
               mode_name(mode), p, (unsigned long long)M_actual, (unsigned long long)seed,
               reps, tmed, tmin, price, se, bs, err, err / se, reproducible, price);
        return 0;
    }

    printf("\n===== PARALLEL MONTE CARLO OPTION PRICING (OpenMP) =====\n");
    printf("Mode                  : %s\n", mode_name(mode));
    printf("Threads               : %d\n", p);
    printf("Paths (M)             : %llu%s\n", (unsigned long long)M_actual,
           mode == M_TIMED ? "  (as many as fit in the time budget)" : "");
    printf("Seed                  : %llu\n", (unsigned long long)seed);
    printf("Monte Carlo price     : %.8f\n", price);
    printf("Hex-float price       : %a   (compare bit-for-bit across runs)\n", price);
    printf("Standard error        : %.8f\n", se);
    printf("Black-Scholes price   : %.8f\n", bs);
    printf("Absolute error        : %.8f  (%.2f standard errors)\n", err, err / se);
    printf("Time, median of %d    : %.4f s   (min %.4f s)\n", reps, tmed, tmin);
    printf("Throughput            : %.2f Mpaths/s\n", (double)M_actual / tmed / 1e6);
    printf("Bit-identical reps    : %s\n", reproducible ? "yes" : "NO");
    free(times);
    return 0;
}
