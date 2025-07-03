/**
 * @file main.cu
 * @brief CLI for calling SGEMM kernels.
 * @copyright Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
 * @date 2025-06-25
 */

#include "runner.cuh"

#include <cstdlib>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(err) sgemm::utils::cudaCheck(err, __FILE__, __LINE__)

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Select a kernel (int from 0-12, 0 for cublas)" << std::endl;
        exit(EXIT_FAILURE);
    }
    int kernelNum = std::atoi(argv[1]);  // aside: https://stackoverflow.com/a/20585402
    if (kernelNum < 0 || kernelNum > 12) {
        std::cerr << "Invalid kernel number, must be between 0-12" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set the device based on the `DEVICE` environment variable (if passed).
    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL) {
        deviceIdx = std::atoi(getenv("DEVICE"));
    }
    CUDA_CHECK(cudaSetDevice(deviceIdx));
    fmt::println("Running kernel-{} on device-{} ...", kernelNum, deviceIdx);
    sgemm::utils::printCudaDeviceInfo();

    // Create a cuBLAS handle (this will only be used for the cuBLAS reference).
    cublasHandle_t handle;
    // NOTE: If successful, `cublasCreate` will return 0 (of type `cublasStatus_t`).
    if (cublasCreate(&handle)) {
        fmt::println("Error: Failed to create cuBLAS handle");
        exit(EXIT_FAILURE);
    }

    // --- RUN CONFIG ---
    // We will run kernels with square matrices of the following sizes.
    // On A100 GPU with 108 SMs, for 100% util, we need 108*32=3456 matrix size. This
    // means that at 3456x3456 (one wave) we have 100% util, at 6912x6912 (two waves)
    // we have 100% util, and at 10368x10368 (three waves) we have 100% util.
    int min_matsize = 1024, increment = 512, num_matsizes = 20;
    float alpha = 0.5f, beta = 0.3f;
    fmt::println(
        "Running `C := {}AB + {}C` with square matrices of sizes: [{}, {}, ..., {}].",
        alpha,
        beta,
        min_matsize,
        min_matsize + increment,
        min_matsize + (num_matsizes - 1) * increment
    );

    // Use CUDA events to time the kernel.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- KERNEL EXECUTION ---
    // Run the kernel for each size.
    int M, K, N;
    for (int i = 0; i < num_matsizes; i++) {
        int matsize = min_matsize + i * increment;

        // --- MEMORY ALLOCATION ---
        // Allocate memory for the host matrices.
        int numel = matsize * matsize;
        int numbytes = numel * sizeof(float);
        float *A = (float *)std::malloc(numbytes);
        float *B = (float *)std::malloc(numbytes);
        float *C = (float *)std::malloc(numbytes);

        // Allocate memory for the cuBLAS reference which is dummy for now, but will be
        // used to copy device-to-host later.
        float *CRef = nullptr;
        if (kernelNum != 0) {
            // NOTE: We don't need to allocate memory for `CRef` if `kernelNum == 0`,
            // since kernel-0 is the cuBLAS reference.
            CRef = (float *)std::malloc(numbytes);
        }

        // Generate random matrices.
        sgemm::utils::randomizeMatrix(A, numel);
        sgemm::utils::randomizeMatrix(B, numel);
        sgemm::utils::randomizeMatrix(C, numel);

        // Allocate memory for the device matrices.
        // Convention: `_d` -> device.
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        float *d_CRef = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, numbytes));
        CUDA_CHECK(cudaMalloc(&d_B, numbytes));
        CUDA_CHECK(cudaMalloc(&d_C, numbytes));
        if (kernelNum != 0) {
            CUDA_CHECK(cudaMalloc(&d_CRef, numbytes));
        }
        // Copy the host matrices to the device.
        CUDA_CHECK(cudaMemcpy(d_A, A, numbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, numbytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, C, numbytes, cudaMemcpyHostToDevice));
        // NOTE: We use `C` while copying to `d_CRef` and NOT `CRef`, which is currently
        // a dummy memory placeholder.
        if (kernelNum != 0) {
            CUDA_CHECK(cudaMemcpy(d_CRef, C, numbytes, cudaMemcpyHostToDevice));
        }

        M = N = K = matsize;
        // nRepeats: https://github.com/salykova/sgemm.cu/blob/master/benchmark.cu.
        int nRepeats = std::max((int)(1000 * exp((min_matsize - matsize) / 3100.0)), 4);
        std::vector<float> elapsedTimeMs(nRepeats, 0.0f);

        // Run the custom SGEMM kernel and the cuBLAS kernel to verify the correctness
        // of the custom kernel. We will run the custom kernel `nRepeats` times later
        // and time that, to avoid possible cold-start overhead.
        // As a side note, we are validating against cuBLAS reference our main goal
        // is to reach cuBLAS-level performance. It is also possible to just validate
        // using three for-loops. For reference, see:
        // https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/H100/matmul.cu.
        if (kernelNum != 0) {
            sgemm::utils::runKernel(0, d_A, d_B, d_CRef, alpha, beta, M, K, N, handle);
            sgemm::utils::runKernel(kernelNum, d_A, d_B, d_C, alpha, beta, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());  // check for any async errors
            CUDA_CHECK(cudaMemcpy(C, d_C, numbytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(CRef, d_CRef, numbytes, cudaMemcpyDeviceToHost));
            // Verify the correctness of the custom kernel.
            if (!sgemm::utils::allClose(C, CRef, numel)) {
                fmt::println("Error: Custom kernel did not match cuBLAS reference");
                if (matsize <= 32) {
                    sgemm::utils::printMatrix(A, matsize, matsize, "A");
                    sgemm::utils::printMatrix(B, matsize, matsize, "B");
                    sgemm::utils::printMatrix(C, matsize, matsize, "C");
                    sgemm::utils::printMatrix(CRef, matsize, matsize, "CRef");
                }
                exit(EXIT_FAILURE);
            }
        }

        // We will now run the custom kernel `nRepeats` times and time it.
        for (int i = 0; i < nRepeats; i++) {
            // Start the timer.
            CUDA_CHECK(cudaEventRecord(start));

            // NOTE: `d_C` is editing in place; however, since we are only running
            // for runtime estimates, we don't reset `d_C` between runs.
            sgemm::utils::runKernel(kernelNum, d_A, d_B, d_C, alpha, beta, M, K, N);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            // NOTE: `cudaEventElapsedTime` returns the time in milliseconds. For
            // reference, see:
            // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html.
            CUDA_CHECK(cudaEventElapsedTime(&elapsedTimeMs[i], start, stop));
            // Flush the L2 cache.
            sgemm::utils::l2Flush();
        }

        // Compute and log runtime and FLOPs. For details, see:
        // https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#matrix-multiplication-performance.
        // For using mid-point+ time records, see: https://salykova.github.io/sgemm-gpu
        // (esp. section on benchmarking CUDA code).
        int midpointIterIdx = nRepeats / 2;
        float avgElapsedTimeMs = 0.0f;
        for (int i = midpointIterIdx; i < nRepeats; i++) {
            avgElapsedTimeMs += elapsedTimeMs[i];
        }
        avgElapsedTimeMs = avgElapsedTimeMs / (nRepeats - midpointIterIdx);
        // GEMM FLOPs: 2MKN (for D := alpha * AB) + 2MN (for C := D + beta*C).
        long flops = 2 * (M * K * N + M * N);
        double tflopsPerSec = flops * 1e-12f * 1e3f / avgElapsedTimeMs;
        // GEMM memory access: MK + KN + MN reads and MN writes.
        long memAccess = (M * K + K * N + 2 * M * N) * sizeof(float);
        double gbPerSec = memAccess * 1e-9f * 1e3f / avgElapsedTimeMs;
        // clang-format off
            fmt::println(
                R"(--- PERFORMANCE ---
+ {:<15} : {}
+ {:<15} : {}
+ {:<15} : {} ms
+ {:<15} : {} TFLOPs/s
+ {:<15} : {} GB/s
-------------------)",
                "matsize", matsize,
                "nRepeats", nRepeats - midpointIterIdx,
                "runtime", avgElapsedTimeMs,
                "flopsThroughput", tflopsPerSec,
                "memThroughput", gbPerSec
            );
        // clang-format on

        // --- CLEANUP ---
        fflush(stdout);

        // Reset the device memory to be the same as `d_CRef`.
        // Currently, `d_C` and `d_CRef` are not the same, which will cause problems
        // in the next run (with the next `size` in `sizes`).
        if (kernelNum != 0) {
            CUDA_CHECK(cudaMemcpy(d_C, d_CRef, numbytes, cudaMemcpyDeviceToDevice));
        }

        // Free the host memory.
        std::free(A), std::free(B), std::free(C);
        if (kernelNum != 0) {
            std::free(CRef);
        }

        // Free the device memory.
        CUDA_CHECK(cudaFree(d_A)), CUDA_CHECK(cudaFree(d_B)), CUDA_CHECK(cudaFree(d_C));
        if (kernelNum != 0) {
            CUDA_CHECK(cudaFree(d_CRef));
        }
    }

    // --- CLEANUP ---
    // Destroy the cuBLAS handle.
    cublasDestroy(handle);

    // Destroy the CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
