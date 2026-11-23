/* CS5 - ping-pong test to get alpha (latency) and beta (time per byte)
 * for the model  T(m) = alpha + beta*m
 *
 * rank 0 sends m bytes to rank 1 and gets them back, one-way = round trip / 2.
 * m = 1, 4, 16, ... up to 16 MiB.
 * alpha = one-way time for 1 byte, beta = slope between the two largest sizes
 *
 * Then for the sample sort Alltoallv:
 *   T ~ (p-1)*alpha + beta * (p-1)/p * (N/p) * 4
 *
 * run: mpirun -np 2 ./pingpong [reps]
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BYTES (16 << 20)

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    if (p < 2) {
        if (rank == 0) fprintf(stderr, "run with 2 ranks: mpirun -np 2 ./pingpong\n");
        MPI_Finalize();
        return 1;
    }
    int reps = (argc > 1) ? atoi(argv[1]) : 200;
    char *buffer = calloc(MAX_BYTES, 1);

    double t_small = 0, t_prev = 0, t_last = 0;
    int m_prev = 0, m_last = 0;
    if (rank == 0) printf("# bytes one_way_us bandwidth_MBps\n");

    for (int m = 1; m <= MAX_BYTES; m *= 4) {
        int r = (m >= (1 << 20)) ? 20 : reps;        // big ones take long, fewer reps
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        for (int i = 0; i < r; i++) {
            if (rank == 0) {
                MPI_Send(buffer, m, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, m, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(buffer, m, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, m, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
        double one_way = (MPI_Wtime() - start) / (2.0 * r);

        if (rank == 0) printf("%d %.3f %.1f\n", m, one_way * 1e6, m / one_way / 1e6);
        if (m == 1) t_small = one_way;
        t_prev = t_last; m_prev = m_last;
        t_last = one_way; m_last = m;
    }

    if (rank == 0) {
        double alpha = t_small;
        double beta  = (t_last - t_prev) / (double)(m_last - m_prev);
        printf("FIT alpha=%.3f us  beta=%.4f ns/byte  (bandwidth ~%.0f MB/s)\n",
               alpha * 1e6, beta * 1e9, 1e-6 / beta);
    }
    free(buffer);
    MPI_Finalize();
    return 0;
}
