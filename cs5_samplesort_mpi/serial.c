/* CS5 - serial baseline: make all N keys and qsort them.
 * Used for the speedup and to check the parallel answer (same sum/hash).
 *
 * run: ./serial [N] [dist] [seed]    defaults 1e8, 0 (uniform), 12345
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sort_common.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    size_t N      = (argc > 1) ? strtoull(argv[1], NULL, 10) : 100000000ULL;
    int dist      = (argc > 2) ? atoi(argv[2]) : 0;
    uint64_t seed = (argc > 3) ? strtoull(argv[3], NULL, 10) : 12345;

    uint32_t *keys = malloc(N * sizeof(uint32_t));
    if (!keys) { printf("out of memory\n"); return 1; }
    for (size_t g = 0; g < N; g++)
        keys[g] = make_key(seed, g, dist);

    // only the sort is timed
    double start = now();
    qsort(keys, N, sizeof(uint32_t), compare_keys);
    double elapsed = now() - start;

    int sorted = 1;
    for (size_t i = 1; i < N; i++)
        if (keys[i - 1] > keys[i]) { sorted = 0; break; }

    uint64_t sum = 0, hash = 0;
    checksum(keys, N, 0, &sum, &hash);
    printf("RESULT impl=serial N=%zu dist=%s time=%.6f sorted=%s sum=%016llx hash=%016llx\n",
           N, dist_name(dist), elapsed, sorted ? "yes" : "NO",
           (unsigned long long)sum, (unsigned long long)hash);

    free(keys);
    return 0;
}
