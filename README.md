# mpi-heat

2D heat conduction, explicit Jacobi, row-decomposed across MPI ranks. Two scaling curves, and the reason
they disagree.

The point of the repo is not "MPI is fast". It is that the **same code** scales 6.4× on one grid size and
3.3× on another on the same 8 cores, and the explanation has nothing to do with MPI.

## The kernel

- 5-point Jacobi stencil, `alpha = 0.20` (below the 0.25 stability limit for 2D explicit)
- global `n × n` grid, split into contiguous row bands; every rank keeps its band plus two halo rows
- halo exchange with two `MPI_Sendrecv` calls per step (`MPI_PROC_NULL` at the boundaries, so no special case)
- timing covers the loop only: one untimed warmup exchange, then 3 timed repetitions, median reported
- **checksum invariant**: the summed field is identical for every rank count, which is how a decomposition bug
  would show up (it would change the answer, not just the speed)

## Build and run

```bash
make build                       # mpicc -O3 -march=native
mpirun -np 4 ./build/heat_mpi -n 1024 -t 200 -r 3
make bench                       # strong + weak scaling -> results/*.csv
```

Needs `mpicc` / `mpirun` (OpenMPI or MPICH).

## Results

Machine: WSL2 Ubuntu 24.04 on an `i7-14650HX`, **12 threads**, gcc 16.2.0 + OpenMPI 5.0.11 (conda-forge).
Median of 3 timed runs, 100–200 steps each.

**Strong scaling, n = 1024 (cache-friendly, ~8 MB per array)**

| ranks | seconds | ms/step | MLUPS | speedup | efficiency |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0653 | 0.326 | 3,212 | 1.00× | 100% |
| 2 | 0.0346 | 0.173 | 6,063 | 1.89× | 94% |
| 4 | 0.0205 | 0.103 | 10,216 | 3.18× | 80% |
| 8 | 0.0102 | 0.051 | 20,652 | **6.43×** | 80% |

**Strong scaling, n = 2048 (streaming, ~32 MB per array, past L3)**

| ranks | seconds | ms/step | MLUPS | speedup | efficiency |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.3501 | 3.501 | 1,198 | 1.00× | 100% |
| 2 | 0.2015 | 2.015 | 2,081 | 1.74× | 87% |
| 4 | 0.1346 | 1.346 | 3,116 | 2.60× | 65% |
| 8 | 0.1077 | 1.077 | 3,894 | **3.25×** | 41% |

**Weak scaling, 512 rows per rank (n = 512, 1024, 2048, 4096)**

| ranks | n | ms/step | MLUPS |
| --- | --- | --- | --- |
| 1 | 512 | 0.088 | 2,983 |
| 2 | 1,024 | 0.217 | 4,827 |
| 4 | 2,048 | 1.243 | 3,374 |
| 8 | 4,096 | 5.445 | 3,081 |

## Reading the two curves honestly

1. **Small grid: near-ideal scaling.** 6.43× on 8 ranks for n=1024. The band each rank touches stays in cache,
   the halo is two rows, and the computation per byte is high enough that adding ranks adds throughput.
2. **Large grid: efficiency falls to 41%.** n=2048 gives 3.25× on the same 8 cores. The stencil reads 5 values
   and writes 1 per cell — roughly **5 reads + 1 write per FLOP**, so the loop is memory-bandwidth-bound.
   Eight ranks share one memory bus; eight times the compute does not mean eight times the bandwidth.
3. **Weak scaling is a trap if the baseline is cache-resident.** With 512 rows/rank, the p=1 case is a
   512×512 grid that fits in cache and runs at 2,983 MLUPS. As the grid grows past cache, per-step time
   goes up 62× for 8× the work. The number is correct; the *framing* would be dishonest if I reported only
   the small rank counts. This is the same failure mode as quoting "4 minutes to train a quadruped" without
   the GPU model.
4. **Where MPI actually costs.** The halo exchange moves `2 × n` doubles per rank per step. At n=1024 that is
   16 KB — small enough that the two `MPI_Sendrecv` calls are invisible next to the stencil. At n=4096 it is
   64 KB per exchange and the message starts to matter, but memory bandwidth dominates first.

The checksum is identical across every run (`423230.037735` at n=1024 for 1/2/4/8 ranks): the decomposition
changes the schedule, not the answer.

## Why CI numbers are not in the tables above

GitHub's `ubuntu-latest` runner has **2 vCPUs**. Running 4 ranks there is oversubscription, and the CI log shows
exactly that: 4,095 MLUPS at 1 rank, 8,230 at 2, 5,007 at 4. It is a perfectly good *regression guard* — the
checksum test and the build both run there — but it cannot produce a scaling curve. The curves above come from
a 12-thread machine, and the runner is named next to the numbers that come from it.

## What I would do next

- [ ] 2D decomposition (rows × columns) so the rank count is not limited by `n % ranks == 0`
- [ ] non-blocking halo exchange (`MPI_Isend/Irecv`) with the interior computed while the halo is in flight
- [ ] pack halos into one contiguous buffer instead of two strided sends
- [ ] rotate the loop order / use `#pragma omp simd` inside each rank — the inner loop is where the bytes go
- [ ] rerun on a real multi-node cluster, where the interconnect rather than the memory bus is the limit

## License

MIT
