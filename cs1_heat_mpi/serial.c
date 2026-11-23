/* CS1 - heat diffusion, serial version (baseline)
 *
 * N x N plate, every step each cell moves towards the average of its
 * 4 neighbours:
 *     new = old + ALPHA * (up + down + left + right - 4*old)
 *
 * Edges are fixed: top = 100, left = 50, right and bottom = 0.
 * We store (N+2) x (N+2) so the outer ring holds the edge values and the
 * inner N x N is what gets updated. Two grids (cur/next) that we swap,
 * otherwise we'd be reading values we already overwrote.
 *
 * Every K steps we also compute maxdiff (biggest change of any cell).
 *
 * run: ./serial [N] [T] [K] [ref_out.bin]
 *   defaults N=1024 T=500 K=50
 *   if ref_out.bin is given the final grid is saved so the MPI version can
 *   check it gets the same thing
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ALPHA 0.1   // has to be <= 0.25 or it blows up

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double edge_temperature(int row, int col)
{
    if (row == 0) return 100.0;   // top
    if (col == 0) return 50.0;    // left
    return 0.0;                   // bottom, right
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int T = (argc > 2) ? atoi(argv[2]) : 500;
    int K = (argc > 3) ? atoi(argv[3]) : 50;
    const char *ref_file = (argc > 4) ? argv[4] : NULL;
    if (K < 1) K = 1;

    int W = N + 2;                                   // width incl. edges
    double *cur  = calloc((size_t)W * W, sizeof(double));
    double *next = calloc((size_t)W * W, sizeof(double));
    if (!cur || !next) { printf("out of memory\n"); return 1; }

#define CELL(grid, row, col) grid[(size_t)(row) * W + (col)]

    // edges in both grids
    for (int r = 0; r < W; r++)
        for (int c = 0; c < W; c++)
            if (r == 0 || c == 0 || r == W - 1 || c == W - 1)
                CELL(cur, r, c) = CELL(next, r, c) = edge_temperature(r, c);

    double maxdiff = 0.0;
    double start = now();

    for (int step = 1; step <= T; step++) {
        int check_now = (step % K == 0);
        double biggest_change = 0.0;

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double old = CELL(cur, i, j);
                double neighbours = CELL(cur, i - 1, j) + CELL(cur, i + 1, j)
                                  + CELL(cur, i, j - 1) + CELL(cur, i, j + 1);
                double updated = old + ALPHA * (neighbours - 4.0 * old);
                CELL(next, i, j) = updated;

                if (check_now && fabs(updated - old) > biggest_change)
                    biggest_change = fabs(updated - old);
            }
        }
        if (check_now) maxdiff = biggest_change;

        double *tmp = cur; cur = next; next = tmp;
    }

    double elapsed = now() - start;

    double checksum = 0.0;   // sum of inner cells, to compare with mpi
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            checksum += CELL(cur, i, j);

    printf("RESULT impl=serial N=%d T=%d K=%d time=%.6f checksum=%.12e maxdiff=%.6e\n",
           N, T, K, elapsed, checksum, maxdiff);

    if (ref_file) {
        FILE *f = fopen(ref_file, "wb");
        if (!f) { perror(ref_file); return 1; }
        for (int i = 1; i <= N; i++)
            fwrite(&CELL(cur, i, 1), sizeof(double), N, f);
        fclose(f);
    }

    free(cur);
    free(next);
    return 0;
}
