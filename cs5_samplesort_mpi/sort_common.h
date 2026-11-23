/* CS5 - stuff shared by serial.c and parallel.c
 *
 * make_key(seed, g, dist) = key at global position g. It only depends on
 * (seed, g), so rank r can make keys r*N/p .. (r+1)*N/p - 1 itself and the
 * whole input is the same no matter what p is -> we can compare with serial.
 *
 * dist: 0 uniform 32-bit, 1 zipf-like (P(j) ~ 1/j for j = 1..1e6, about 5%
 * of keys are 1), 2 dup16 (only 16 different values)
 *
 * checksum: sum of keys (nothing lost) and sum of key*(2*pos+1) (same order)
 */
#ifndef SORT_COMMON_H
#define SORT_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline uint64_t random_at(uint64_t seed, uint64_t g)
{
    uint64_t x = seed * 0x2545F4914F6CDD1DULL + g;
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline uint32_t make_key(uint64_t seed, uint64_t g, int dist)
{
    uint64_t r = random_at(seed, g);
    if (dist == 1) {                                          // zipf
        double u = (double)(r >> 11) / 9007199254740992.0;    // [0,1)
        uint32_t key = (uint32_t)floor(exp(u * log(1.0e6)));  // 1..1e6
        return key < 1 ? 1 : key;
    }
    if (dist == 2)                                            // 0, 1000, ..., 15000
        return (uint32_t)(r >> 60) * 1000u;
    return (uint32_t)(r >> 32);
}

static const char *dist_name(int dist)
{
    return dist == 1 ? "zipf" : dist == 2 ? "dup16" : "uniform";
}

static int compare_keys(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

// first_pos = global position of keys[0]
static inline void checksum(const uint32_t *keys, size_t n, uint64_t first_pos,
                            uint64_t *sum, uint64_t *hash)
{
    for (size_t i = 0; i < n; i++) {
        *sum  += keys[i];
        *hash += (uint64_t)keys[i] * (2 * (first_pos + i) + 1);
    }
}

#endif
