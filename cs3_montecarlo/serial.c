/* serial.c - Case Study 3 serial baseline: Monte Carlo European call pricing.
 *
 * Usage: ./serial [-n M] [-s seed] [-r reps] [--csv]
 *   -n  number of simulated paths (default 1e8; rounded down to an even number)
 *   -s  seed (default 42)
 *   -r  number of repetitions; the MEDIAN time is reported (default 3)
 */
#include "mc_common.h"

int main(int argc, char **argv)
{
    uint64_t M = DEFAULT_M, seed = DEFAULT_SEED;
    int reps = 3, csv = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc)      M = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv"))              csv = 1;
        else { fprintf(stderr, "usage: %s [-n M] [-s seed] [-r reps] [--csv]\n", argv[0]); return 1; }
    }
    M &= ~1ULL;                              /* Box-Muller consumes paths in pairs */
    if (reps < 1) reps = 1;

    double *times = (double *)malloc((size_t)reps * sizeof(double));
    double price = 0, se = 0;

    for (int rep = 0; rep < reps; rep++) {
        PCG32 rng;
        pcg32_seed(&rng, seed, 1ULL);        /* stream 1 == thread 0 of parallel.c */
        double sum, sumsq;
        double t0 = wall_time();
        mc_accumulate(&rng, M / 2, &sum, &sumsq);
        times[rep] = wall_time() - t0;
        mc_finalize(sum, sumsq, M, &price, &se);
    }

    double bs  = black_scholes_price();
    double err = fabs(price - bs);
    double tmed = median_of(times, reps), tmin = min_of(times, reps);

    if (csv) {
        /* mode,threads,M,seed,reps,t_median,t_min,price,se,bs,abs_err,z,repro,hexprice */
        printf("CSV,serial,1,%llu,%llu,%d,%.6f,%.6f,%.10f,%.10f,%.10f,%.10f,%.4f,1,%a\n",
               (unsigned long long)M, (unsigned long long)seed, reps, tmed, tmin,
               price, se, bs, err, err / se, price);
        return 0;
    }

    printf("\n===== SERIAL MONTE CARLO OPTION PRICING =====\n");
    printf("Paths (M)             : %llu\n", (unsigned long long)M);
    printf("Seed                  : %llu\n", (unsigned long long)seed);
    printf("Monte Carlo price     : %.8f\n", price);
    printf("Standard error        : %.8f\n", se);
    printf("95%% CI                : [%.6f, %.6f]\n", price - 1.96 * se, price + 1.96 * se);
    printf("Black-Scholes price   : %.8f\n", bs);
    printf("Absolute error        : %.8f  (%.2f standard errors)\n", err, err / se);
    printf("Time, median of %d    : %.4f s   (min %.4f s)\n", reps, tmed, tmin);
    printf("Throughput            : %.2f Mpaths/s\n", (double)M / tmed / 1e6);
    free(times);
    return 0;
}
