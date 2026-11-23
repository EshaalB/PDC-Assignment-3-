/* CS2 - stuff used by both serial.c and parallel.c
 *
 * k-mer -> number: each base is 2 bits (A=0 C=1 G=2 T=3), so a k-mer fits
 * in 2k bits. e.g. "GAT" = 10 00 11 = 35, so we just do hist[35]++.
 * Moving the window one letter right:
 *     code = ((code << 2) | new_base) & mask
 * (the mask drops the base that left the window)
 */
#ifndef KMER_COMMON_H
#define KMER_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static inline double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// splitmix64-style hash, random_at(seed, i) is always the same for the same i
// so it doesn't matter which thread makes which letter
static inline uint64_t random_at(uint64_t seed, uint64_t i)
{
    uint64_t x = seed ^ (i * 0xD1B54A32D192ED03ULL);
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// random DNA. skewed=0: 25% each, skewed=1: A 45% C 5% G 5% T 45%
// (not timed, the pragma is just so generating 1e8 letters is quicker)
static void make_dna(char *dna, size_t L, uint64_t seed, int skewed)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < L; i++) {
        uint64_t r = random_at(seed, i);
        if (!skewed) {
            dna[i] = "ACGT"[r >> 62];                 // top 2 bits
        } else {
            unsigned pct = (unsigned)(r >> 32) % 100;
            dna[i] = pct < 45 ? 'A' : pct < 50 ? 'C' : pct < 55 ? 'G' : 'T';
        }
    }
}

static unsigned char BASE_CODE[256];
static void setup_base_codes(void)
{
    BASE_CODE['A'] = 0; BASE_CODE['C'] = 1; BASE_CODE['G'] = 2; BASE_CODE['T'] = 3;
}

// total count + a fingerprint so two histograms can be compared quickly
static uint64_t histogram_summary(const uint32_t *hist, size_t bins, uint64_t *total)
{
    uint64_t fingerprint = 0, sum = 0;
    for (size_t i = 0; i < bins; i++) {
        sum += hist[i];
        fingerprint += random_at(7, i) * (uint64_t)hist[i];
    }
    *total = sum;
    return fingerprint;
}

#endif
