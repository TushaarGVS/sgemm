/**
 * @file runner.cu
 * @brief Runner for the SGEMM kernels.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-06-25
 */

#include "kernels/1_sgemm_naive.cuh"
#include "kernels/2_sgemm_gmem_coalesce.cuh"
#include "kernels/3_sgemm_smem_tiling.cuh"
#include "runner.cuh"

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fmt/base.h>
#include <fmt/format.h>

#define CUDA_CHECK(err) sgemm::utils::cudaCheck(err, __FILE__, __LINE__)

namespace sgemm::utils {
void cudaCheck(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fmt::println(
            "[CUDA ERROR] file={} line={}\n{}", file, line, cudaGetErrorString(err)
        );
        exit(EXIT_FAILURE);
    }
}

void printCudaDeviceInfo() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    // clang-format off
	fmt::println(
		R"(-------- DEVICE PROPERTIES --------
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {}.{}
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {}
+ {:<21} : {} B
+ {:<21} : {} GB
+ {:<21} : {} KB
+ {:<21} : {} KB
+ {:<21} : {} KB
+ {:<21} : {}
------------------------------------)",
		"deviceId", deviceId,
		"name", prop.name,
		"computeCapability", prop.major, prop.minor,
		"numSms", prop.multiProcessorCount,
		"maxThreadsPerBlock", prop.maxThreadsPerBlock,
		"maxThreadsPerSm", prop.maxThreadsPerMultiProcessor,
        "threadsPerWarp", prop.warpSize,
        "registersPerBlock", prop.regsPerBlock,
        "registersPerSm", prop.regsPerMultiprocessor,
        "numWarpsPerSm", prop.maxThreadsPerMultiProcessor / prop.warpSize,
        "memoryBusWidth", prop.memoryBusWidth,
		"totalGlobalMem", prop.totalGlobalMem / 1024 / 1024 / 1024,
		"sharedMemPerBlock", prop.sharedMemPerBlock / 1024,
		"sharedMemPerSm", prop.sharedMemPerMultiprocessor / 1024,
		"totalConstMem", prop.totalConstMem / 1024,
		"warpSize", prop.warpSize
	);
    // clang-format on
}

void l2Flush() {
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    int l2CacheSize;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, deviceId));
    int *buf;  // we will use this to flush out CUDA L2 cache
    if (l2CacheSize > 0) {
        CUDA_CHECK(cudaMalloc(&buf, l2CacheSize));
        CUDA_CHECK(cudaMemsetAsync(buf, 0, l2CacheSize));  // flush the cache
        CUDA_CHECK(cudaFree(buf));
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
    cublasHandle_t handle
) {
    switch (kernelNum) {
        case 0: {
            // SGEMM cuBLAS reference in FP32.
            // NOTE: cuBLAS uses column-major order, but A, B, and C are row-major.
            // However, if we multiply B and A (in that order) in column-major order,
            // i.e., B.T@A.T = A@B, which is the same as multiplying A and B in
            // row-major order.
            // For reference, see: https://docs.nvidia.com/cuda/cublas/#cublasgemmex.
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,  // don't transpose the first matrix (= B)
                CUBLAS_OP_N,  // don't transpose the second matrix (= A)
                N,
                M,
                K,
                &alpha,      // alpha
                B,           // first matrix
                CUDA_R_32F,  // fp32 (all matrices are fp32)
                // Stride to reach the next element in the first matrix; since our
                // matrices are row-major, we need to move a whole row of N elements to
                // get to the next element to make it seem like column-major.
                N,
                A,           // second matrix
                CUDA_R_32F,  // fp32 (all matrices are fp32)
                // Stride to reach the next element in the second matrix; again, since
                // our matrices are row-major, we need to move a whole row of K elements
                // to get to the next element to make it seem like column-major.
                K,
                &beta,       // beta
                C,           // output matrix
                CUDA_R_32F,  // fp32 (all matrices are fp32)
                // Stride to reach the next element in the output matrix; same as
                // before, move a whole row of N elements.
                N,
                CUBLAS_COMPUTE_32F,            // internal compute precision, also fp32
                CUBLAS_GEMM_DEFAULT_TENSOR_OP  // use tensor core ops, if available
            );
            break;
        }
        case 1: {
            // Naive SGEMM kernel.
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            dim3 blockDim(32, 32);
            sgemm::kernels::sgemm_naive<<<gridDim, blockDim>>>(
                A, B, C, alpha, beta, M, K, N
            );
            break;
        }
        case 2: {
            // SGEMM kernel with coalesced global memory access.
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            dim3 blockDim(32, 32);
            sgemm::kernels::sgemm_gmem_coalesce<<<gridDim, blockDim>>>(
                A, B, C, alpha, beta, M, K, N
            );
            break;
        }
        case 3: {
            // SGEMM kernel with shared memory blocking.
            // NOTE: This kernel assumes square matrices; guards for non-square matrices
            // are not implemented.
            assert(M == N && N == K);
            // NOTE: This kernel assumes that M, N, and K are divisible by 32. There
            // are no explicit guardrails for this, so we assert it here.
            assert(M % 32 == 0 && N % 32 == 0 && K % 32 == 0);

            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            dim3 blockDim(32, 32);
            // In this kernel, we don't use L1 cache (only SMEM is used). Hence, we
            // "carve out" all of the L1 to SMEM.
            cudaFuncSetAttribute(
                sgemm::kernels::sgemm_smem_tiling<32>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                cudaSharedmemCarveoutMaxShared
            );
            sgemm::kernels::sgemm_smem_tiling<32>
                <<<gridDim, blockDim>>>(A, B, C, alpha, beta, M, K, N);
            break;
        }
        default: {
            fmt::println("Kernel-{} not implemented", kernelNum);
            exit(EXIT_FAILURE);
        }
    }
}
}  // namespace sgemm::utils
