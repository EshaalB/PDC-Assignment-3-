/* CS5 - sample sort with MPI
 *
 * Each rank starts with N/p keys. At the end every rank has a sorted piece
 * and rank 0's keys <= rank 1's keys <= ... No rank holds all N keys.
 *
 * Steps:
 *  1. local sort      qsort my own keys
 *  2. splitters       every rank sends evenly spaced samples to rank 0
 *                     (Gather), rank 0 picks p-1 splitters and Bcasts them
 *  3. partition       cut my sorted keys at the splitters (binary search),
 *                     slice j goes to rank j
 *  4. exchange        Alltoall for the counts, Alltoallv for the keys
 *  5. merge           I now have p sorted runs, merge them with a min-heap
 *
 * If there are lots of equal keys (zipf, dup16) all copies of a value end
 * up on the same rank, so the load can get uneven. We print max/mean count.
 * -o = oversampling, takes os*p-1 samples per rank instead of p-1.
 *
 * Each step has a barrier before it and is timed separately.
 *
 * run: mpirun -np 4 ./parallel -n 100000000 -d 0 -o 1 -c
 *   -n N, -d dist (0 uniform 1 zipf 2 dup16), -s seed, -o oversampling,
 *   -c print how many keys each rank ends up with
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "sort_common.h"

static int rank, p;

// step 2: pick p-1 splitters (same on all ranks after the Bcast)
//
// each rank takes m = os*p-1 samples, which cut its keys into os*p equal
// parts, so sample os*(j+1)-1 is at the border between bucket j and j+1.
// after rank 0 sorts all p*m samples, the p copies of sample i (one per rank)
// are around position i*p .. i*p+p-1, and we take the middle one.
// (first version just took every p-th sample and oversampling made the
//  balance worse, this fixed it)
static void choose_splitters(const uint32_t *keys, size_t n, int os, uint32_t *splitters)
{
    int m = os * p - 1;

    uint32_t *my_samples = malloc(m * sizeof(uint32_t));
    for (int i = 0; i < m; i++)
        my_samples[i] = (n > 0) ? keys[(size_t)(i + 1) * n / (m + 1)] : UINT32_MAX;

    uint32_t *all_samples = NULL;
    if (rank == 0) all_samples = malloc((size_t)m * p * sizeof(uint32_t));
    MPI_Gather(my_samples, m, MPI_UINT32_T, all_samples, m, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        qsort(all_samples, (size_t)m * p, sizeof(uint32_t), compare_keys);
        for (int j = 0; j < p - 1; j++) {
            size_t sample = (size_t)os * (j + 1) - 1;
            splitters[j] = all_samples[sample * p + p / 2];
        }
        free(all_samples);
    }

    MPI_Bcast(splitters, p - 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    free(my_samples);
}

// binary search: first index with keys[i] > value
static size_t first_bigger(const uint32_t *keys, size_t n, uint32_t value)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (keys[mid] <= value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// step 3: part j = keys with splitters[j-1] < key <= splitters[j].
// keys are sorted so every part is one slice (send_start, send_count)
static void partition(const uint32_t *keys, size_t n, const uint32_t *splitters,
                      int *send_count, int *send_start)
{
    size_t from = 0;
    for (int j = 0; j < p; j++) {
        size_t to = (j < p - 1) ? first_bigger(keys, n, splitters[j]) : n;
        if (to < from) to = from;            // only if two splitters are equal
        send_start[j] = (int)from;
        send_count[j] = (int)(to - from);
        from = to;
    }
}

// step 5: merge p sorted runs with a min-heap of the front key of each run,
// about n*log2(p) comparisons
typedef struct { uint32_t key; int run; } HeapItem;

static void heap_fix_down(HeapItem *heap, int size, int i)
{
    while (1) {
        int left = 2 * i + 1, right = 2 * i + 2, smallest = i;
        if (left < size && heap[left].key < heap[smallest].key) smallest = left;
        if (right < size && heap[right].key < heap[smallest].key) smallest = right;
        if (smallest == i) return;
        HeapItem tmp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = tmp;
        i = smallest;
    }
}

static void merge_runs(const uint32_t *in, const int *count, const int *start, uint32_t *out)
{
    HeapItem *heap = malloc(p * sizeof(HeapItem));
    size_t *next = malloc(p * sizeof(size_t));   // next unread index of each run
    size_t *end  = malloc(p * sizeof(size_t));
    int size = 0;

    for (int r = 0; r < p; r++) {
        next[r] = start[r];
        end[r]  = (size_t)start[r] + count[r];
        if (count[r] > 0) {
            heap[size].key = in[next[r]];
            heap[size].run = r;
            size++;
        }
    }
    for (int i = size / 2 - 1; i >= 0; i--)
        heap_fix_down(heap, size, i);

    size_t written = 0;
    while (size > 0) {
        int r = heap[0].run;
        out[written++] = heap[0].key;
        next[r]++;
        if (next[r] < end[r])
            heap[0].key = in[next[r]];
        else
            heap[0] = heap[--size];            // run r is finished
        heap_fix_down(heap, size, 0);
    }
    free(heap); free(next); free(end);
}

// checks + printing (rank 0)
static void check_and_report(const uint32_t *keys, size_t n, size_t N, int dist, int os,
                             const double step_time[6], int show_counts)
{
    // is my piece sorted
    int my_ok = 1;
    for (size_t i = 1; i < n; i++)
        if (keys[i - 1] > keys[i]) { my_ok = 0; break; }
    int all_ok;
    MPI_Reduce(&my_ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

    // count, min, max of every rank -> rank 0
    unsigned long long mine[3] = { n, n ? keys[0] : 0, n ? keys[n - 1] : 0 };
    unsigned long long *info = (rank == 0) ? malloc(3 * p * sizeof(unsigned long long)) : NULL;
    MPI_Gather(mine, 3, MPI_UNSIGNED_LONG_LONG, info, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // global position of my first key = keys on the ranks before me (Exscan)
    unsigned long long n_ull = n, keys_before_me = 0;
    MPI_Exscan(&n_ull, &keys_before_me, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) keys_before_me = 0;      // undefined on rank 0
    uint64_t sum = 0, hash = 0, total_sum = 0, total_hash = 0;
    checksum(keys, n, keys_before_me, &sum, &hash);
    MPI_Reduce(&sum, &total_sum, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hash, &total_hash, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    // slowest rank per step
    double t[6];
    MPI_Reduce(step_time, t, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank != 0) return;

    int borders_ok = 1, seen_one = 0;
    unsigned long long total = 0, biggest_count = 0, prev_max = 0;
    for (int r = 0; r < p; r++) {
        unsigned long long count = info[3 * r], lo = info[3 * r + 1], hi = info[3 * r + 2];
        total += count;
        if (count > biggest_count) biggest_count = count;
        if (count > 0) {                       // skip empty ranks
            if (seen_one && prev_max > lo) borders_ok = 0;
            prev_max = hi;
            seen_one = 1;
        }
    }
    double mean = (double)total / p;
    double total_time = t[1] + t[2] + t[3] + t[4] + t[5];

    printf("RESULT impl=mpi N=%zu p=%d dist=%s oversample=%d time=%.6f "
           "t_localsort=%.6f t_sampling=%.6f t_partition=%.6f t_alltoallv=%.6f t_merge=%.6f "
           "max_count=%llu mean_count=%.1f imbalance=%.4f sum=%016llx hash=%016llx\n",
           N, p, dist_name(dist), os, total_time, t[1], t[2], t[3], t[4], t[5],
           biggest_count, mean, biggest_count / mean,
           (unsigned long long)total_sum, (unsigned long long)total_hash);
    printf("VERIFY local_sorted=%s borders=%s count=%llu/%zu %s\n",
           all_ok ? "yes" : "NO", borders_ok ? "yes" : "NO", total, N,
           (all_ok && borders_ok && total == N) ? "PASS" : "FAIL");
    if (show_counts) {
        printf("COUNTS");
        for (int r = 0; r < p; r++) printf(" %llu", info[3 * r]);
        printf("\n");
    }
    free(info);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t N = 100000000ULL;
    int dist = 0, os = 1, show_counts = 0, opt;
    uint64_t seed = 12345;
    while ((opt = getopt(argc, argv, "n:d:s:o:c")) != -1) {
        switch (opt) {
        case 'n': N = strtoull(optarg, NULL, 10); break;
        case 'd': dist = atoi(optarg); break;
        case 's': seed = strtoull(optarg, NULL, 10); break;
        case 'o': os = atoi(optarg); break;
        case 'c': show_counts = 1; break;
        default:
            if (rank == 0) fprintf(stderr, "usage: %s [-n N] [-d dist] [-s seed] [-o oversample] [-c]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (os < 1) os = 1;

    double step_time[6] = {0};   // index 1..5 = steps
    double t;

    // make my N/p keys (not timed)
    size_t first = N * rank / p, last = N * (rank + 1) / p;
    size_t n = last - first;
    uint32_t *keys = malloc((n ? n : 1) * sizeof(uint32_t));
    if (!keys) { fprintf(stderr, "rank %d: out of memory\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (size_t i = 0; i < n; i++)
        keys[i] = make_key(seed, first + i, dist);

    // 1. local sort
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    qsort(keys, n, sizeof(uint32_t), compare_keys);
    step_time[1] = MPI_Wtime() - t;

    // 2. splitters
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    uint32_t *splitters = malloc((p > 1 ? p - 1 : 1) * sizeof(uint32_t));
    if (p > 1)
        choose_splitters(keys, n, os, splitters);
    step_time[2] = MPI_Wtime() - t;

    // 3. partition
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    int *send_count = malloc(p * sizeof(int)), *send_start = malloc(p * sizeof(int));
    int *recv_count = malloc(p * sizeof(int)), *recv_start = malloc(p * sizeof(int));
    partition(keys, n, splitters, send_count, send_start);
    step_time[3] = MPI_Wtime() - t;

    // 4. exchange: counts first, then the keys
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, MPI_COMM_WORLD);
    size_t n_received = 0;
    for (int r = 0; r < p; r++) {
        recv_start[r] = (int)n_received;
        n_received += recv_count[r];
    }
    uint32_t *received = malloc((n_received ? n_received : 1) * sizeof(uint32_t));
    if (!received) { fprintf(stderr, "rank %d: out of memory\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Alltoallv(keys, send_count, send_start, MPI_UINT32_T,
                  received, recv_count, recv_start, MPI_UINT32_T, MPI_COMM_WORLD);
    step_time[4] = MPI_Wtime() - t;
    free(keys);

    // 5. merge
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime();
    uint32_t *result = malloc((n_received ? n_received : 1) * sizeof(uint32_t));
    merge_runs(received, recv_count, recv_start, result);
    step_time[5] = MPI_Wtime() - t;
    free(received);

    check_and_report(result, n_received, N, dist, os, step_time, show_counts);

    free(result); free(splitters);
    free(send_count); free(send_start); free(recv_count); free(recv_start);
    MPI_Finalize();
    return 0;
}
