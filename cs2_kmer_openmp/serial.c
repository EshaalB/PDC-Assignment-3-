/* CS2 - k-mer counting, serial version (baseline)
 *
 * k-mer = substring of length k. "ACGTA" has the 3-mers ACG, CGT, GTA.
 * A string of length L has L-k+1 k-mers so the counts must add up to that
 * (easy way to catch a broken parallel version).
 *
 * We slide a window over the string and do hist[code]++ (see kmer_common.h).
 *
 * run: ./serial [L] [k] [skewed] [seed] [repeats]
 *   defaults L=1e8 k=12 skewed=0 seed=42 repeats=3, prints the median time
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kmer_common.h"

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    size_t L      = (argc > 1) ? strtoull(argv[1], NULL, 10) : 100000000ULL;
    int k         = (argc > 2) ? atoi(argv[2]) : 12;
    int skewed    = (argc > 3) ? atoi(argv[3]) : 0;
    uint64_t seed = (argc > 4) ? strtoull(argv[4], NULL, 10) : 42;
    int repeats   = (argc > 5) ? atoi(argv[5]) : 3;
    if (k < 1 || k > 15 || L < (size_t)k) { printf("need 1 <= k <= 15 and L >= k\n"); return 1; }
    if (repeats < 1) repeats = 1;
    if (repeats > 50) repeats = 50;

    setup_base_codes();
    size_t bins   = (size_t)1 << (2 * k);       // 4^k
    uint32_t mask = (uint32_t)(bins - 1);
    size_t n_kmers = L - k + 1;

    char *dna      = malloc(L);
    uint32_t *hist = malloc(bins * sizeof(uint32_t));
    if (!dna || !hist) { printf("out of memory\n"); return 1; }
    make_dna(dna, L, seed, skewed);

    double times[50];
    for (int rep = 0; rep < repeats; rep++) {
        memset(hist, 0, bins * sizeof(uint32_t));
        double start = now();

        // first k-1 letters, window not full yet
        uint32_t code = 0;
        for (int j = 0; j < k - 1; j++)
            code = (code << 2) | BASE_CODE[(unsigned char)dna[j]];

        for (size_t pos = 0; pos < n_kmers; pos++) {
            char new_letter = dna[pos + k - 1];
            code = ((code << 2) | BASE_CODE[(unsigned char)new_letter]) & mask;
            hist[code]++;
        }
        times[rep] = now() - start;
    }
    qsort(times, repeats, sizeof(double), compare_doubles);
    double median = times[repeats / 2];

    uint64_t total, fingerprint = histogram_summary(hist, bins, &total);
    printf("RESULT impl=serial L=%zu k=%d skew=%d reps=%d time_med=%.6f total=%llu expected=%zu "
           "checksum=%016llx Mkmers_per_s=%.1f\n",
           L, k, skewed, repeats, median, (unsigned long long)total, n_kmers,
           (unsigned long long)fingerprint, n_kmers / median / 1e6);

    free(dna);
    free(hist);
    return 0;
}
