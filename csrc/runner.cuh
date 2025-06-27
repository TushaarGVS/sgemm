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

namespace sgemm {
constexpr uint CEIL_DIV(uint a, uint b) {
    return (a + b - 1) / b;
}

namespace utils {
void cudaCheck(cudaError_t err, const char *file, int line);

void printCudaDeviceInfo();

/**
 * Taken from: https://github.com/siboehm/SGEMM_CUDA/blob/master/src/runner.cu.
 */
void randomizeMatrix(float *mat, size_t size);

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

bool allClose(const float *mat, const float *matRef, size_t size, float atol = 1e-2f);

/**
 * Adapted from: https://github.com/siboehm/SGEMM_CUDA/blob/master/src/runner.cu.
 */
void printMatrix(
    const float *mat, size_t nRows, size_t nCols, const char *name = "MATRIX"
);
}  // namespace utils
}  // namespace sgemm
