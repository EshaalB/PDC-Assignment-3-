/* CS2 - k-mer counting, OpenMP versions
 *
 * Start positions 0..L-k are split into blocks of BLOCK and the blocks are
 * given to threads with omp for. A k-mer near the end of a block reads a few
 * letters from the next block but that's only reading, and each start
 * position is in exactly one block so nothing is counted twice or missed.
 *
 * Versions (-v):
 *  A  shared hist, plain hist[code]++  -> data race, loses counts (on purpose)
 *  B  shared hist with omp atomic      -> correct but every ++ is atomic
 *  C  private hist per thread, merged at the end. each copy is padded to a
 *     64 byte cache line
 *  U  same as C but copies packed together (no padding). with k=1 a copy is
 *     only 16 bytes so 4 threads share one cache line -> false sharing
 *
 * Every version is checked bin by bin against a serial count, prints
 * CORRECT/WRONG.
 *
 * run: OMP_NUM_THREADS=8 ./parallel -L 100000000 -k 12 -v ABC
 *   -L length, -k k (1..15), -s 0/1 skewed, -S seed, -r repeats, -v versions
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include "kmer_common.h"

#define BLOCK 65536
#define CACHE_LINE_INTS 16   // 64 bytes / 4

// set once in main
static const char *dna;
static int k;
static size_t n_kmers, bins, n_blocks;
static uint32_t mask;

// code of the k-1 letters before the first k-mer at pos
static inline uint32_t start_code(size_t pos)
{
    uint32_t code = 0;
    for (int j = 0; j < k - 1; j++)
        code = (code << 2) | BASE_CODE[(unsigned char)dna[pos + j]];
    return code;
}

static inline size_t block_begin(size_t b) { return b * BLOCK; }
static inline size_t block_end(size_t b)   { size_t e = (b + 1) * BLOCK; return e < n_kmers ? e : n_kmers; }

// A: no protection (race)
static void count_version_A(uint32_t *hist)
{
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < n_blocks; b++) {
        uint32_t code = start_code(block_begin(b));
        for (size_t pos = block_begin(b); pos < block_end(b); pos++) {
            code = ((code << 2) | BASE_CODE[(unsigned char)dna[pos + k - 1]]) & mask;
            hist[code]++;          // read-add-write, two threads can clash here
        }
    }
}

// B: atomic increment
static void count_version_B(uint32_t *hist)
{
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < n_blocks; b++) {
        uint32_t code = start_code(block_begin(b));
        for (size_t pos = block_begin(b); pos < block_end(b); pos++) {
            code = ((code << 2) | BASE_CODE[(unsigned char)dna[pos + k - 1]]) & mask;
            #pragma omp atomic
            hist[code]++;
        }
    }
}

// C and U: private copies then merge.
// copy_size = distance between two threads' copies (C rounds up to a cache
// line, U is just bins). merge_time is returned separately for the report.
static void count_private(uint32_t *hist, uint32_t *copies, size_t copy_size, double *merge_time)
{
    double merge_start = 0.0;

    #pragma omp parallel
    {
        int me = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        uint32_t *my_hist = copies + (size_t)me * copy_size;

        memset(my_hist, 0, copy_size * sizeof(uint32_t));

        #pragma omp for schedule(static)
        for (size_t b = 0; b < n_blocks; b++) {
            uint32_t code = start_code(block_begin(b));
            for (size_t pos = block_begin(b); pos < block_end(b); pos++) {
                code = ((code << 2) | BASE_CODE[(unsigned char)dna[pos + k - 1]]) & mask;
                my_hist[code]++;
            }
        }
        // implicit barrier after omp for, so all counting is done here

        #pragma omp single
        merge_start = omp_get_wtime();

        // merge in parallel: each thread sums a range of bins over all copies
        #pragma omp for schedule(static)
        for (size_t bin = 0; bin < bins; bin++) {
            uint32_t sum = 0;
            for (int t = 0; t < n_threads; t++)
                sum += copies[(size_t)t * copy_size + bin];
            hist[bin] = sum;
        }
    }
    *merge_time = omp_get_wtime() - merge_start;
}

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    size_t L = 100000000ULL;
    int skewed = 0, repeats = 3, opt;
    uint64_t seed = 42;
    const char *versions = "ABC";
    k = 12;
    while ((opt = getopt(argc, argv, "L:k:s:S:r:v:")) != -1) {
        switch (opt) {
        case 'L': L = strtoull(optarg, NULL, 10); break;
        case 'k': k = atoi(optarg); break;
        case 's': skewed = atoi(optarg); break;
        case 'S': seed = strtoull(optarg, NULL, 10); break;
        case 'r': repeats = atoi(optarg); break;
        case 'v': versions = optarg; break;
        default:
            fprintf(stderr, "usage: %s [-L len] [-k k] [-s skew] [-S seed] [-r repeats] [-v ABCU]\n", argv[0]);
            return 1;
        }
    }
    if (k < 1 || k > 15 || L < (size_t)k) { printf("need 1 <= k <= 15 and L >= k\n"); return 1; }
    if (repeats < 1) repeats = 1;
    if (repeats > 50) repeats = 50;

    setup_base_codes();
    bins     = (size_t)1 << (2 * k);
    mask     = (uint32_t)(bins - 1);
    n_kmers  = L - k + 1;
    n_blocks = (n_kmers + BLOCK - 1) / BLOCK;

    char *buffer       = malloc(L);
    uint32_t *correct  = calloc(bins, sizeof(uint32_t));   // serial answer
    uint32_t *hist     = malloc(bins * sizeof(uint32_t));
    if (!buffer || !correct || !hist) { printf("out of memory\n"); return 1; }
    make_dna(buffer, L, seed, skewed);
    dna = buffer;

    // reference count (serial)
    uint32_t code = start_code(0);
    for (size_t pos = 0; pos < n_kmers; pos++) {
        code = ((code << 2) | BASE_CODE[(unsigned char)dna[pos + k - 1]]) & mask;
        correct[code]++;
    }
    uint64_t correct_total, correct_fp = histogram_summary(correct, bins, &correct_total);
    int n_threads = omp_get_max_threads();
    printf("# L=%zu k=%d bins=%zu skew=%d threads=%d reps=%d reference_total=%llu checksum=%016llx\n",
           L, k, bins, skewed, n_threads, repeats,
           (unsigned long long)correct_total, (unsigned long long)correct_fp);

    for (const char *v = versions; *v; v++) {
        char version = *v;
        if (version != 'A' && version != 'B' && version != 'C' && version != 'U') continue;

        uint32_t *copies = NULL;
        size_t copy_size = 0;
        if (version == 'C' || version == 'U') {
            if (version == 'C')   // pad to whole cache lines
                copy_size = (bins + CACHE_LINE_INTS - 1) / CACHE_LINE_INTS * CACHE_LINE_INTS;
            else
                copy_size = bins;
            size_t bytes = (size_t)n_threads * copy_size * sizeof(uint32_t);
            bytes = (bytes + 63) / 64 * 64;        // aligned_alloc wants a multiple of 64
            copies = aligned_alloc(64, bytes);
            if (!copies) { printf("out of memory for private histograms\n"); return 1; }
        }

        double times[50], merges[50];
        for (int rep = 0; rep < repeats; rep++) {
            memset(hist, 0, bins * sizeof(uint32_t));
            merges[rep] = 0.0;
            double start = omp_get_wtime();
            if (version == 'A')      count_version_A(hist);
            else if (version == 'B') count_version_B(hist);
            else                     count_private(hist, copies, copy_size, &merges[rep]);
            times[rep] = omp_get_wtime() - start;
        }
        qsort(times, repeats, sizeof(double), compare_doubles);
        qsort(merges, repeats, sizeof(double), compare_doubles);
        double median = times[repeats / 2];

        size_t wrong_bins = 0;
        for (size_t i = 0; i < bins; i++)
            if (hist[i] != correct[i]) wrong_bins++;
        uint64_t total, fp = histogram_summary(hist, bins, &total);
        double extra_mb = copies ? (double)n_threads * copy_size * sizeof(uint32_t) / (1024.0 * 1024.0) : 0.0;

        printf("RESULT version=%c k=%d L=%zu skew=%d threads=%d time_med=%.6f merge_med=%.6f "
               "total=%llu expected=%zu lost=%lld mismatched_bins=%zu checksum=%016llx "
               "extra_MB=%.1f Mkmers_per_s=%.1f %s\n",
               version, k, L, skewed, n_threads, median, merges[repeats / 2],
               (unsigned long long)total, n_kmers, (long long)n_kmers - (long long)total,
               wrong_bins, (unsigned long long)fp, extra_mb, n_kmers / median / 1e6,
               wrong_bins == 0 ? "CORRECT" : "WRONG");
        free(copies);
    }

    free(buffer);
    free(correct);
    free(hist);
    return 0;
}
