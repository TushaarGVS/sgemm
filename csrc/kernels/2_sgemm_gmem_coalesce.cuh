/**
 * @file sgemm_gmem_coalesce.cuh
 * @brief SGEMM kernel that coalesces global memory accesses.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-06-30
 */

#include <cuda_runtime.h>

namespace sgemm::kernels {
__global__ void sgemm_gmem_coalesce(
    const float *A,  // M x K
    const float *B,  // K x N
    float *C,        // M x N
    float alpha,
    float beta,
    uint M,
    uint K,
    uint N
) {
    // Computes C[y][x].
    // Unlike in the naive kernel, the threads in a warp need to access the same
    // row of A and different columns of B for us to coalesce global memory accesses.

    // Move to the correct block, then move to the right thread.
    const uint x = blockDim.x * blockIdx.x + threadIdx.x;
    const uint y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < M && y < N) {
        // Compute: A[y]@B[:,x].
        // To ensure global memory coalescing, we will compute A[:,x]@B[y] as opposed
        // to A[y]@B[:,x] in the naive kernel. This means, threads (0, 0) and (1, 0),
        // in the same warp, will load different columns of B and the same row of A.
        float acc = 0.0f;
        for (int i = 0; i < K; i++) {
            acc += A[y * K + i] * B[x + i * N];
        }
        acc *= alpha;

        // Compute: C[y][x] := acc + beta * C[y][x].
        C[N * y + x] = acc + beta * C[N * y + x];
    }
}
}  // namespace sgemm::kernels
