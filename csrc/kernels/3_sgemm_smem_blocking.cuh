/**
 * @file 3_sgemm_smem_blocking.cuh
 * @brief SGEMM kernel with shared memory blocking.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-07-02
 */

#pragma once

namespace sgemm::kernels {
__global__ void sgemm_smem_blocking(
    const float *A,  // M x K
    const float *B,  // K x N
    float *C,        // M x N
    float alpha,
    float beta,
    uint M,
    uint K,
    uint N
) {}
}  // namespace sgemm::kernels
