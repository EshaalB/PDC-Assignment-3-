/* mc_common.h - shared pieces for Case Study 3 (Monte Carlo option pricing).
 * Used by BOTH serial.c and parallel.c so the two programs run exactly the same
 * numerical kernel (this is what makes "parallel with p=1 == serial" bit-exact).
 */
#ifndef MC_COMMON_H
#define MC_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---------- Model parameters (European call, geometric Brownian motion) ---- */
#define S0    100.0   /* initial stock price      */
#define K     100.0   /* strike                   */
#define R     0.05    /* risk-free rate           */
#define SIGMA 0.20    /* volatility               */
#define T     1.0     /* maturity (years)         */

#define DEFAULT_M    100000000ULL   /* 1e8 paths (assignment minimum) */
#define DEFAULT_SEED 42ULL

/* ---------- PCG32 generator (one independent stream per worker) ----------- */
typedef struct { uint64_t state, inc; } PCG32;

static inline uint32_t pcg32_random(PCG32 *rng)
{
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/* SplitMix64: used only to turn (seed, stream) into a well-mixed initial state,
 * so that streams with neighbouring ids do not start in correlated states. */
static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* Different `stream` => different odd increment => a different full-period
 * sequence. Same (seed, stream) => identical sequence (used on purpose in the
 * "same seed" experiment). */
static inline void pcg32_seed(PCG32 *rng, uint64_t seed, uint64_t stream)
{
    rng->state = 0;
    rng->inc   = (stream << 1u) | 1u;
    pcg32_random(rng);
    rng->state += splitmix64(seed ^ (stream * 0xD1B54A32D192ED03ULL));
    pcg32_random(rng);
}

static inline double random_uniform(PCG32 *rng)       /* in (0,1), never 0 */
{
    return ((double)pcg32_random(rng) + 0.5) / 4294967296.0;
}

/* Box-Muller: two uniforms -> TWO independent N(0,1) (cos AND sin branch). */
static inline void normal_pair(PCG32 *rng, double *z0, double *z1)
{
    double u1  = random_uniform(rng);
    double u2  = random_uniform(rng);
    double rad = sqrt(-2.0 * log(u1));
    double th  = 2.0 * M_PI * u2;
    *z0 = rad * cos(th);
    *z1 = rad * sin(th);
}

/* ---------- The compute kernel -------------------------------------------- */
static inline double payoff_from_z(double z, double drift, double vol)
{
    double st = S0 * exp(drift + vol * z);
    return st > K ? st - K : 0.0;
}

/* Simulates 2*npairs paths with the given generator; returns sum of payoffs and
 * sum of squared payoffs (needed for the standard error). */
static inline void mc_accumulate(PCG32 *rng, uint64_t npairs,
                                 double *sum_out, double *sumsq_out)
{
    const double drift = (R - 0.5 * SIGMA * SIGMA) * T;
    const double vol   = SIGMA * sqrt(T);
    double s = 0.0, q = 0.0;
    for (uint64_t i = 0; i < npairs; i++) {
        double z0, z1;
        normal_pair(rng, &z0, &z1);
        double p0 = payoff_from_z(z0, drift, vol);
        double p1 = payoff_from_z(z1, drift, vol);
        s += p0; s += p1;
        q += p0 * p0 + p1 * p1;
    }
    *sum_out = s;
    *sumsq_out = q;
}

/* ---------- Estimator, closed form, statistics ----------------------------- */
static inline void mc_finalize(double sum, double sumsq, uint64_t M,
                               double *price, double *stderr_out)
{
    double disc = exp(-R * T);
    double mean = sum / (double)M;
    double var  = (sumsq - sum * sum / (double)M) / (double)(M - 1);
    if (var < 0.0) var = 0.0;
    *price      = disc * mean;
    *stderr_out = disc * sqrt(var / (double)M);
}

static inline double black_scholes_price(void)
{
    double d1 = (log(S0 / K) + (R + 0.5 * SIGMA * SIGMA) * T) / (SIGMA * sqrt(T));
    double d2 = d1 - SIGMA * sqrt(T);
    double n1 = 0.5 * erfc(-d1 / sqrt(2.0));
    double n2 = 0.5 * erfc(-d2 / sqrt(2.0));
    return S0 * n1 - K * exp(-R * T) * n2;
}

/* ---------- Timing / small utilities --------------------------------------- */
static inline double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static inline double median_of(const double *v, int n)
{
    double *c = (double *)malloc((size_t)n * sizeof(double));
    memcpy(c, v, (size_t)n * sizeof(double));
    qsort(c, (size_t)n, sizeof(double), cmp_double);
    double m = (n % 2) ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

static inline double min_of(const double *v, int n)
{
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] < m) m = v[i];
    return m;
}

#endif /* MC_COMMON_H */
