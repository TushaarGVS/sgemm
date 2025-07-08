/**
 * @file runner.cuh
 * @brief Runner for SGEMM kernels.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-06-25
 */

#pragma once

#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fmt/format.h>

namespace sgemm::utils {
__forceinline__ constexpr uint CEIL_DIV(uint a, uint b) {
    return (a + b - 1) / b;
}

void cudaCheck(cudaError_t err, const char *file, int line);

void printCudaDeviceInfo();

/**
 * Adapted from: https://github.com/NVIDIA/nvbench/blob/main/nvbench/detail/l2flush.cuh.
 */
void l2Flush();

/**
 * Adapted from: https://github.com/siboehm/SGEMM_CUDA/blob/master/src/runner.cu.
 */
__inline__ void initRandom(float *mat, size_t size) {
    srand(time(nullptr));  // seed
    for (size_t i = 0; i < size; i++) {
        // Random floats as ({0, 1, ..., 3} + {0, 0.01, ..., 0.03}) * {-1, 1}.
        float tmp = (float)(rand() % 4) + 0.01 * (float)(rand() % 4);
        tmp = tmp * (rand() % 2 ? 1 : -1);
        mat[i] = tmp;
    }
}

void runKernel(
    int kernelNum,
    const float *A,
    const float *B,
    float *C,
    float alpha,
    float beta,
    uint M,
    uint K,
    uint N,
    cublasHandle_t handle = nullptr
);

__inline__ bool
allClose(const float *mat, const float *matRef, size_t size, float atol = 1e-2f) {
    float diff = 0.0;
    for (uint i = 0; i < size; i++) {
        diff = std::fabs(matRef[i] - mat[i]);
        if (diff > atol) {
            fmt::println(
                "Error at index={} with atol={} (ref={}, mat={}, diff={})",
                i,
                atol,
                matRef[i],
                mat[i],
                diff
            );
            return false;
        }
    }
    return true;
}

/**
 * Adapted from: https://github.com/siboehm/SGEMM_CUDA/blob/master/src/runner.cu.
 */
__inline__ void
printMatrix(const float *mat, size_t nRows, size_t nCols, const char *name = "MATRIX") {
    fmt::print("{} [{} x {}]:\n[", name, nRows, nCols);
    for (size_t i = 0; i < nRows; ++i) {
        for (size_t j = 0; j < nCols; ++j) {
            fmt::print("{:>7.2f}", mat[i * nCols + j]);
            if (j + 1 < nCols)
                fmt::print(", ");
        }
        if (i + 1 < nRows)
            fmt::print(";\n ");
    }
    fmt::print("]\n");
}
}  // namespace sgemm::utils
