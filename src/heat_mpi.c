/* 2D heat conduction, explicit Jacobi, 1D row decomposition with MPI.
 *
 * The measurement is the loop, not the setup: timing starts after the halo
 * exchange pattern has run once, so the first (cold) exchange is not in the sample.
 *
 * Domain: N x N global grid, rank r owns a contiguous band of rows.
 * Each rank keeps a (local + 2) x N buffer: two halo rows plus its own band.
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ALPHA 0.20 /* diffusion number, must stay below 0.25 for 2D explicit */

static void init_field(double *u, int local_rows, int n, int rank) {
    /* u[i][j] with i in [0, local_rows+2), j in [0, n) */
    const int stride = n;
    for (int i = 0; i < local_rows + 2; ++i) {
        const int gi = rank * local_rows + (i - 1); /* global row index, halos included */
        for (int j = 0; j < n; ++j) {
            double v = 0.0;
            if (gi > 0 && gi < n - 1 && j > 0 && j < n - 1) {
                const double x = (double)gi / (double)(n - 1);
                const double y = (double)j / (double)(n - 1);
                v = sin(M_PI * x) * sin(M_PI * y);
            }
            u[i * stride + j] = v;
        }
    }
}

static void exchange_halos(double *u, int local_rows, int n, int rank, int size, MPI_Comm comm) {
    const int stride = n;
    MPI_Status status;
    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    /* send my first row up, receive the row below into the bottom halo */
    MPI_Sendrecv(&u[1 * stride], n, MPI_DOUBLE, up, 0, &u[(local_rows + 1) * stride], n, MPI_DOUBLE,
                 down, 0, comm, &status);

    /* send my last row down, receive from above into the top halo */
    MPI_Sendrecv(&u[local_rows * stride], n, MPI_DOUBLE, down, 1, &u[0 * stride], n, MPI_DOUBLE, up,
                 1, comm, &status);
}

static void step(double *u, double *unew, int local_rows, int n) {
    const int stride = n;
    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            unew[i * stride + j] = u[i * stride + j] +
                                   ALPHA * (u[(i - 1) * stride + j] + u[(i + 1) * stride + j] +
                                            u[i * stride + j - 1] + u[i * stride + j + 1] -
                                            4.0 * u[i * stride + j]);
        }
    }
    /* boundaries stay at zero */
    for (int i = 1; i <= local_rows; ++i) {
        unew[i * stride + 0] = 0.0;
        unew[i * stride + (n - 1)] = 0.0;
    }
}

int main(int argc, char **argv) {
    int n = 1024, steps = 100, reps = 3, rows_per_rank = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) rows_per_rank = atoi(argv[++i]);
    }

    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rows_per_rank > 0) n = rows_per_rank * size; /* weak scaling: problem grows with ranks */

    if (n % size != 0) {
        if (rank == 0) fprintf(stderr, "n (%d) must be divisible by ranks (%d)\n", n, size);
        MPI_Finalize();
        return 2;
    }

    const int local_rows = n / size;
    const int stride = n;
    double *u = calloc((size_t)(local_rows + 2) * stride, sizeof(double));
    double *unew = calloc((size_t)(local_rows + 2) * stride, sizeof(double));
    if (!u || !unew) {
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    init_field(u, local_rows, n, rank);
    memcpy(unew, u, (size_t)(local_rows + 2) * stride * sizeof(double));

    /* warm the exchange path once, untimed */
    exchange_halos(u, local_rows, n, rank, size, MPI_COMM_WORLD);

    double best = 1e30, median = 0.0;
    double samples[16];
    if (reps < 1) reps = 1;
    if (reps > 16) reps = 16;

    for (int r = 0; r < reps; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        for (int s = 0; s < steps; ++s) {
            exchange_halos(u, local_rows, n, rank, size, MPI_COMM_WORLD);
            step(u, unew, local_rows, n);
            double *tmp = u;
            u = unew;
            unew = tmp;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double dt = MPI_Wtime() - t0;
        samples[r] = dt;
        if (dt < best) best = dt;
    }

    /* median of the samples */
    for (int a = 1; a < reps; ++a) {
        double key = samples[a];
        int b = a - 1;
        while (b >= 0 && samples[b] > key) {
            samples[b + 1] = samples[b];
            --b;
        }
        samples[b + 1] = key;
    }
    median = samples[reps / 2];

    /* deterministic checksum so different rank counts can be compared */
    double local_sum = 0.0;
    for (int i = 1; i <= local_rows; ++i)
        for (int j = 0; j < n; ++j) local_sum += u[i * stride + j];
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double updates = (double)n * (double)n * (double)steps;
        printf("%s,%d,%d,%d,%.6f,%.3f,%.2f,%.6f\n", rows_per_rank > 0 ? "weak" : "strong", n, steps,
               size, median, median / steps * 1e3, updates / median / 1e6, global_sum);
        fflush(stdout);
    }

    free(u);
    free(unew);
    MPI_Finalize();
    return 0;
}
