/**
 * @file 3_sgemm_smem_tiling.cuh
 * @brief SGEMM kernel with matrix tiling for SMEM store/load.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-07-04
 */

#pragma once

namespace sgemm::kernels {
template <const unsigned tileDim>
__global__ void sgemm_smem_tiling(
    const float *A,  // M x K
    const float *B,  // K x N
    float *C,        // M x N
    float alpha,
    float beta,
    uint M,
    uint K,
    uint N
) {
    // Each block computes a single tileDim x tileDim tile of C. To do so, we slide
    // tileDim x tileDim tiles of A left-to-right, tileDim x tileDim tiles of B
    // top-to-bottom.
    const unsigned cRow = blockIdx.y;
    const unsigned cCol = blockIdx.x;

    // The kernel assumes a block of <tileDim, tileDim> threads, and as many blocks
    // as needed. This is to say that each thread loads a single fl32 of A tile and a
    // single fl32 of B tile into SMEM.
    // Move A, B, C pointers to the right locations.
    A += cRow * tileDim * K;                   // row=cRow, col=0
    B += cCol * tileDim;                       // row=0, col=cCol
    C += cRow * tileDim * K + cCol * tileDim;  // row=cRow, col=cCol

    // Allocate shared memory for the current block.
    __shared__ float tileA[tileDim * tileDim];
    __shared__ float tileB[tileDim * tileDim];

    // Slide A, B tiles across the K-th dimension, each time computing a matmul
    // between the A and B tiles.
    float acc = 0.0f;  // accumulator
    for (int i = 0; i < K; i += tileDim) {
        // Load A and B tiles into SMEM.
        // Observe that the memory locations are contiguous, both for A and B, hence a
        // warp-wide LOAD instruction is used to fetch 32 fl32s at a time.
        // Additionally, there are no bank conflicts, since the memory locations stored
        // by the warp will be in different SMEM banks.
        tileA[threadIdx.y * tileDim + threadIdx.x] = A[threadIdx.y * K + threadIdx.x];
        tileB[threadIdx.y * tileDim + threadIdx.x] = B[threadIdx.y * N + threadIdx.x];

        // NOTE: `__syncthreads()` is a block-level (not grid-level) sync barrier.
        __syncthreads();  // wait for all threads to load data to SMEM
        // Advance A and B pointers to the next tile.
        A += tileDim;
        B += tileDim * N;

        // Compute: tileA@tileB.
        // Each thread only computes a single entry of the C tile.
        for (int j = 0; j < tileDim; j++) {
            // NOTE: This kernel assumes that M, N, and K are divisible by tileDim.
            // There are no explicit guardrails for this, so we assert it at the time
            // of function invocation (on the host).
            acc += tileA[threadIdx.y * tileDim + j] * tileB[threadIdx.x + tileDim * j];
        }

        __syncthreads();  // to avoid premature loading of the next block into SMEM
        // NOTE: We don't need to multiply by `alpha` here, since we will need to
        // accumulate the sum across all `i` before multiplying with `alpha`.
    }
    C[threadIdx.y * N + threadIdx.x] =
        alpha * acc + beta * C[threadIdx.y * N + threadIdx.x];
}
}  // namespace sgemm::kernels
