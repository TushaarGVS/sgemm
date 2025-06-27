/**
 * @file sgemm_naive.cuh
 * @brief Naive SGEMM kernel.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-06-25
 */

#include <cuda_runtime.h>

namespace sgemm::kernels {
__global__ void sgemm_naive(
    const float *A,  // M x K
    const float *B,  // K x N
    float *C,        // M x N
    float alpha,
    float beta,
    uint M,
    uint K,
    uint N
) {
    // Think: Single-thread perspective!!
    // Expected A, B, C to be in row-major format.

    // Move `blockIdx` number of blocks and then move `threadIdx` number of threads
    // within that block.
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute: acc = A[x]*B[:,y].
    // Accessing row-x: move over `x` times, each time, by K elements.
    // Accession col-y: move in steps of N (0, N, 2N, ...), offset by `y` each time.
    if (x < M && y < N) {
        float acc = 0.0f;  // typecast aside: https://stackoverflow.com/a/5199515
        for (int i = 0; i < K; i++) {
            acc += A[x * K + i] * B[y + i * N];
        }

        // Compute: alpha * acc + beta * C[x][y].
        // Same as before, since C is M x N, get to the x-th row by moving `N * x`
        // elements and then, move to the right column by moving by `y` elements.
        C[x * N + y] = alpha * acc + beta * C[x * N + y];
    }
}
}  // namespace sgemm::kernels
