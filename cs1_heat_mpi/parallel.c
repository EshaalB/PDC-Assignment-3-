/* CS1 - heat diffusion, MPI version
 *
 * Same update as serial.c, but the plate is split between the ranks and each
 * rank only stores its own block plus a 1-cell halo around it.
 *
 * -d 1 : 1D, horizontal strips (only up/down neighbours)
 * -d 2 : 2D, tiles made with MPI_Cart_create (up/down/left/right)
 *
 * Ranks on the plate edge get MPI_PROC_NULL as neighbour, so send/recv to
 * them just does nothing and we don't need special cases. 1D is really just
 * a p x 1 cartesian grid so it uses the same code.
 *
 * Halo layout of one block:
 *
 *         halo row   (last row of the rank above)
 *       +----------+
 *     h | my cells | h    (columns from left/right ranks)
 *       +----------+
 *         halo row   (first row of the rank below)
 *
 * -m 0 : blocking, MPI_Sendrecv all halos then update
 * -m 1 : non-blocking, Isend/Irecv, update the inside of the block while
 *        the messages are going, Waitall, then update the outer ring
 *
 * Every K steps we do MPI_Allreduce(MAX) on the biggest change (convergence).
 *
 * run:  mpirun -np 4 ./parallel -n 1024 -t 500 -k 50 -d 2 -m 1
 *   -n grid size, -t steps, -k allreduce every k steps, -d 1|2, -m 0|1,
 *   -e eps (stop early if maxdiff < eps, 0 = off),
 *   -r file (compare with the grid that ./serial saved)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#define ALPHA 0.1

// info about the block this rank owns
typedef struct {
    int rows, cols;            // block size without halo
    int first_row, first_col;  // where the block starts in the full plate
    int width;                 // cols + 2 (row length incl. halo)
    int up, down, left, right; // neighbours, MPI_PROC_NULL if none
    MPI_Datatype column;       // one column of the block (strided)
    MPI_Comm comm;             // cartesian comm
} Block;

// rows/cols 0 and last are the halo
#define CELL(b, grid, r, c) (grid)[(size_t)(r) * (b)->width + (c)]

static double edge_temperature(int row, int col)
{
    if (row == 0) return 100.0;   // same as serial.c
    if (col == 0) return 50.0;
    return 0.0;
}

// split n things over P parts as evenly as possible, e.g. 10 over 3 -> 4,3,3
static void share(int n, int P, int which, int *count, int *start)
{
    int base = n / P, extra = n % P;
    *count = base + (which < extra ? 1 : 0);
    *start = which * base + (which < extra ? which : extra);
}

// update rows r0..r1, cols c0..c1 of the block (same formula as serial)
// returns the biggest change if check is set
static double update_cells(const Block *b, const double *cur, double *next,
                           int r0, int r1, int c0, int c1, int check)
{
    double biggest = 0.0;
    for (int i = r0; i <= r1; i++) {
        for (int j = c0; j <= c1; j++) {
            double old = CELL(b, cur, i, j);
            double neighbours = CELL(b, cur, i - 1, j) + CELL(b, cur, i + 1, j)
                              + CELL(b, cur, i, j - 1) + CELL(b, cur, i, j + 1);
            double updated = old + ALPHA * (neighbours - 4.0 * old);
            CELL(b, next, i, j) = updated;
            if (check && fabs(updated - old) > biggest)
                biggest = fabs(updated - old);
        }
    }
    return biggest;
}

// blocking halo exchange with 4 Sendrecv's.
// rows are contiguous so we just send C doubles, columns are not so we use
// the vector type (1 double, then skip a whole row, ...)
static void exchange_halos_blocking(const Block *b, double *g)
{
    int R = b->rows, C = b->cols;
    // top row up, bottom halo from below
    MPI_Sendrecv(&CELL(b, g, 1, 1), C, MPI_DOUBLE, b->up, 0,
                 &CELL(b, g, R + 1, 1), C, MPI_DOUBLE, b->down, 0, b->comm, MPI_STATUS_IGNORE);
    // bottom row down, top halo from above
    MPI_Sendrecv(&CELL(b, g, R, 1), C, MPI_DOUBLE, b->down, 1,
                 &CELL(b, g, 0, 1), C, MPI_DOUBLE, b->up, 1, b->comm, MPI_STATUS_IGNORE);
    // left col to the left, right halo from the right
    MPI_Sendrecv(&CELL(b, g, 1, 1), 1, b->column, b->left, 2,
                 &CELL(b, g, 1, C + 1), 1, b->column, b->right, 2, b->comm, MPI_STATUS_IGNORE);
    // right col to the right, left halo from the left
    MPI_Sendrecv(&CELL(b, g, 1, C), 1, b->column, b->right, 3,
                 &CELL(b, g, 1, 0), 1, b->column, b->left, 3, b->comm, MPI_STATUS_IGNORE);
}

// non-blocking version: just posts the 4 Irecv + 4 Isend and returns.
// need MPI_Waitall(8, req) before the halo can be used. same tags as above.
static void start_halo_exchange(const Block *b, double *g, MPI_Request req[8])
{
    int R = b->rows, C = b->cols;
    MPI_Irecv(&CELL(b, g, 0, 1),     C, MPI_DOUBLE, b->up,    1, b->comm, &req[0]);
    MPI_Irecv(&CELL(b, g, R + 1, 1), C, MPI_DOUBLE, b->down,  0, b->comm, &req[1]);
    MPI_Irecv(&CELL(b, g, 1, 0),     1, b->column,  b->left,  3, b->comm, &req[2]);
    MPI_Irecv(&CELL(b, g, 1, C + 1), 1, b->column,  b->right, 2, b->comm, &req[3]);
    MPI_Isend(&CELL(b, g, 1, 1),     C, MPI_DOUBLE, b->up,    0, b->comm, &req[4]);
    MPI_Isend(&CELL(b, g, R, 1),     C, MPI_DOUBLE, b->down,  1, b->comm, &req[5]);
    MPI_Isend(&CELL(b, g, 1, 1),     1, b->column,  b->left,  2, b->comm, &req[6]);
    MPI_Isend(&CELL(b, g, 1, C),     1, b->column,  b->right, 3, b->comm, &req[7]);
}

// for testing: rank 0 collects all blocks and compares with the serial output file
static void verify_against_serial(const Block *b, const double *g, int N,
                                  const int dims[2], const char *ref_file)
{
    int rank, p;
    MPI_Comm_rank(b->comm, &rank);
    MPI_Comm_size(b->comm, &p);

    // copy block without the halo
    double *mine = malloc((size_t)b->rows * b->cols * sizeof(double));
    for (int i = 0; i < b->rows; i++)
        memcpy(&mine[(size_t)i * b->cols], &CELL(b, g, i + 1, 1), b->cols * sizeof(double));

    if (rank != 0) {
        MPI_Send(mine, b->rows * b->cols, MPI_DOUBLE, 0, 99, b->comm);
        free(mine);
        return;
    }

    // rank 0 reads the serial answer
    double *serial = malloc((size_t)N * N * sizeof(double));
    FILE *f = fopen(ref_file, "rb");
    if (!f || fread(serial, sizeof(double), (size_t)N * N, f) != (size_t)N * N) {
        fprintf(stderr, "cannot read reference file %s\n", ref_file);
        MPI_Abort(b->comm, 1);
    }
    fclose(f);

    double max_error = 0.0;
    for (int r = 0; r < p; r++) {
        // which part of the plate does rank r have
        int coords[2], rows, cols, row0, col0;
        MPI_Cart_coords(b->comm, r, 2, coords);
        share(N, dims[0], coords[0], &rows, &row0);
        share(N, dims[1], coords[1], &cols, &col0);

        double *blk = mine;
        if (r != 0) {
            blk = malloc((size_t)rows * cols * sizeof(double));
            MPI_Recv(blk, rows * cols, MPI_DOUBLE, r, 99, b->comm, MPI_STATUS_IGNORE);
        }
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                double err = fabs(blk[(size_t)i * cols + j] - serial[(size_t)(row0 + i) * N + (col0 + j)]);
                if (err > max_error) max_error = err;
            }
        if (r != 0) free(blk);
    }
    printf("VERIFY max_abs_error=%.3e tolerance=1e-12 %s\n",
           max_error, max_error <= 1e-12 ? "PASS" : "FAIL");
    free(serial);
    free(mine);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int p, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // options
    int N = 1024, T = 500, K = 50, decomp = 2, nonblocking = 1, opt;
    double eps = 0.0;
    const char *ref_file = NULL;
    while ((opt = getopt(argc, argv, "n:t:k:d:m:e:r:")) != -1) {
        switch (opt) {
        case 'n': N = atoi(optarg); break;
        case 't': T = atoi(optarg); break;
        case 'k': K = atoi(optarg); break;
        case 'd': decomp = atoi(optarg); break;
        case 'm': nonblocking = atoi(optarg); break;
        case 'e': eps = atof(optarg); break;
        case 'r': ref_file = optarg; break;
        default:
            if (world_rank == 0)
                fprintf(stderr, "usage: %s [-n N] [-t T] [-k K] [-d 1|2] [-m 0|1] [-e eps] [-r ref.bin]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    if (K < 1) K = 1;

    // process grid
    int dims[2] = {0, 0};                 // rows x cols of ranks
    if (decomp == 1) { dims[0] = p; dims[1] = 1; }       // 1D: p strips
    else MPI_Dims_create(p, 2, dims);                    // 2D: 4 -> 2x2, 8 -> 4x2
    int no_wraparound[2] = {0, 0};

    Block b;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, no_wraparound, 0, &b.comm);
    int rank, my_coords[2];
    MPI_Comm_rank(b.comm, &rank);
    MPI_Cart_coords(b.comm, rank, 2, my_coords);
    MPI_Cart_shift(b.comm, 0, 1, &b.up, &b.down);
    MPI_Cart_shift(b.comm, 1, 1, &b.left, &b.right);

    // my rows and cols of the plate
    share(N, dims[0], my_coords[0], &b.rows, &b.first_row);
    share(N, dims[1], my_coords[1], &b.cols, &b.first_col);
    b.width = b.cols + 2;

    // column = b.rows doubles, b.width apart
    MPI_Type_vector(b.rows, 1, b.width, MPI_DOUBLE, &b.column);
    MPI_Type_commit(&b.column);

    // allocate block + halo, set fixed edges
    size_t ncells = (size_t)(b.rows + 2) * b.width;
    double *cur  = calloc(ncells, sizeof(double));
    double *next = calloc(ncells, sizeof(double));
    if (!cur || !next) { fprintf(stderr, "rank %d: out of memory\n", rank); MPI_Abort(b.comm, 1); }

    for (int i = 0; i < b.rows + 2; i++)
        for (int j = 0; j < b.width; j++) {
            int gr = b.first_row + i, gc = b.first_col + j;     // global row/col
            if (gr == 0 || gc == 0 || gr == N + 1 || gc == N + 1)
                CELL(&b, cur, i, j) = CELL(&b, next, i, j) = edge_temperature(gr, gc);
        }

    // main loop
    double comm_time = 0.0, allreduce_time = 0.0, maxdiff = 0.0;
    int steps_done = 0, R = b.rows, C = b.cols;

    MPI_Barrier(b.comm);
    double start = MPI_Wtime();

    for (int step = 1; step <= T; step++) {
        int check_now = (step % K == 0);
        double my_change = 0.0, ch;

        if (!nonblocking) {
            // blocking: halos first, then the whole block
            double t = MPI_Wtime();
            exchange_halos_blocking(&b, cur);
            comm_time += MPI_Wtime() - t;

            my_change = update_cells(&b, cur, next, 1, R, 1, C, check_now);
        } else {
            // non-blocking: start the exchange, do the inside while waiting
            MPI_Request req[8];
            double t = MPI_Wtime();
            start_halo_exchange(&b, cur, req);
            comm_time += MPI_Wtime() - t;

            // rows 2..R-1, cols 2..C-1 don't need the halo
            if (R > 2 && C > 2)
                my_change = update_cells(&b, cur, next, 2, R - 1, 2, C - 1, check_now);

            t = MPI_Wtime();
            MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
            comm_time += MPI_Wtime() - t;

            // outer ring now that halos are here
            ch = update_cells(&b, cur, next, 1, 1, 1, C, check_now);              // top
            if (ch > my_change) my_change = ch;
            if (R > 1) {
                ch = update_cells(&b, cur, next, R, R, 1, C, check_now);          // bottom
                if (ch > my_change) my_change = ch;
            }
            if (R > 2) {
                ch = update_cells(&b, cur, next, 2, R - 1, 1, 1, check_now);      // left
                if (ch > my_change) my_change = ch;
                if (C > 1) {
                    ch = update_cells(&b, cur, next, 2, R - 1, C, C, check_now);  // right
                    if (ch > my_change) my_change = ch;
                }
            }
        }

        double *tmp = cur; cur = next; next = tmp;
        steps_done = step;

        // convergence check every K steps
        if (check_now) {
            double t = MPI_Wtime();
            MPI_Allreduce(&my_change, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, b.comm);
            allreduce_time += MPI_Wtime() - t;
            if (eps > 0.0 && maxdiff < eps) break;
        }
    }
    double my_time = MPI_Wtime() - start;

    // checksum + timings to rank 0
    double my_sum = 0.0, checksum = 0.0;
    for (int i = 1; i <= R; i++)
        for (int j = 1; j <= C; j++)
            my_sum += CELL(&b, cur, i, j);
    MPI_Reduce(&my_sum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, b.comm);

    double time_max, comm_max, comm_total, allreduce_max;
    MPI_Reduce(&my_time, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, b.comm);       // slowest rank
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, b.comm);
    MPI_Reduce(&comm_time, &comm_total, 1, MPI_DOUBLE, MPI_SUM, 0, b.comm);
    MPI_Reduce(&allreduce_time, &allreduce_max, 1, MPI_DOUBLE, MPI_MAX, 0, b.comm);

    if (rank == 0)
        printf("RESULT impl=mpi N=%d T=%d K=%d p=%d dims=%dx%d decomp=%dD mode=%s steps=%d "
               "time=%.6f comm_max=%.6f comm_avg=%.6f allreduce=%.6f checksum=%.12e maxdiff=%.6e\n",
               N, T, K, p, dims[0], dims[1], decomp, nonblocking ? "nonblocking" : "blocking",
               steps_done, time_max, comm_max, comm_total / p, allreduce_max, checksum, maxdiff);

    // optional check against serial
    if (ref_file)
        verify_against_serial(&b, cur, N, dims, ref_file);

    MPI_Type_free(&b.column);
    free(cur);
    free(next);
    MPI_Comm_free(&b.comm);
    MPI_Finalize();
    return 0;
}
