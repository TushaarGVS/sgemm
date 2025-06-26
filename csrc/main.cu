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
	int kernelNum = std::atoi(argv[1]);	 // aside: https://stackoverflow.com/a/20585402
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

	// Use CUDA events to time the kernel.
	float elapsedTimeInSecs = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// --- RUN CONFIG ---
	// We will run kernels with square matrices of the following sizes.
	std::vector<uint> sizes = {2, 128, 256, 512, 1024, 2048, 4096, 8192};
	// float alpha = 0.5f, beta = 0.3f;
	float alpha = 1.0f, beta = 0.0f;
	fmt::println(
		"Running `C := {}AB + {}C` with square matrices of sizes: {}",
		alpha,
		beta,
		sizes
	);

	// --- MEMORY ALLOCATION ---
	// Allocate memory for the host matrices.
	uint max_size = sizes[sizes.size() - 1];
	uint numel = max_size * max_size;
	uint numbytes = numel * sizeof(float);
	float *A = (float *)std::malloc(numbytes);
	float *B = (float *)std::malloc(numbytes);
	float *C = (float *)std::malloc(numbytes);
	// Allocate memory for the cuBLAS reference which is dummy for now, but will be
	// used to copy device-to-host later.
	float *CRef = (float *)std::malloc(numbytes);

	// Generate random matrices.
	sgemm::utils::randomizeMatrix(A, numel);
	sgemm::utils::randomizeMatrix(B, numel);
	sgemm::utils::randomizeMatrix(C, numel);

	// Allocate memory for the device matrices.
	// Convention: `_d` -> device.
	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_CRef = nullptr;
	CUDA_CHECK(cudaMalloc(&d_A, numbytes));
	CUDA_CHECK(cudaMalloc(&d_B, numbytes));
	CUDA_CHECK(cudaMalloc(&d_C, numbytes));
	CUDA_CHECK(cudaMalloc(&d_CRef, numbytes));
	// Copy the host matrices to the device.
	CUDA_CHECK(cudaMemcpy(d_A, A, numbytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, numbytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_C, C, numbytes, cudaMemcpyHostToDevice));
	// NOTE: We use `C` while copying to `d_CRef` and NOT `CRef`, which is currently
	// a dummy memory placeholder.
	CUDA_CHECK(cudaMemcpy(d_CRef, C, numbytes, cudaMemcpyHostToDevice));

	// --- KERNEL EXECUTION ---
	// Run the kernel for each size.
	uint8_t nRepeats = 50;
	uint M, K, N;
	for (uint size : sizes) {
		M = K = N = size;

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
			CUDA_CHECK(cudaGetLastError());	 // check for any async errors
			CUDA_CHECK(cudaMemcpy(C, d_C, numbytes, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(CRef, d_CRef, numbytes, cudaMemcpyDeviceToHost));
			// Verify the correctness of the custom kernel.
			if (!sgemm::utils::allClose(C, CRef, numel)) {
				fmt::println("Error: Custom kernel did not match cuBLAS reference");
				if (size <= 32) {
					sgemm::utils::printMatrix(A, size, size, "A");
					sgemm::utils::printMatrix(B, size, size, "B");
					sgemm::utils::printMatrix(C, size, size, "C");
					sgemm::utils::printMatrix(CRef, size, size, "CRef");
				}
			}
		}

		// We will now run the custom kernel `nRepeats` times and time it.
		cudaEventRecord(start);
		for (int i = 0; i < nRepeats; i++) {
			// NOTE: `d_C` is editing in place; however, since we are only running
			// for runtime estimates, we don't reset `d_C` between runs.
			sgemm::utils::runKernel(kernelNum, d_A, d_B, d_C, alpha, beta, M, K, N);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTimeInSecs, start, stop);

		// Compute and log runtime and FLOPs. For details, see:
		// https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#matrix-multiplication-performance.
		elapsedTimeInSecs = elapsedTimeInSecs / 1000.0f;
		float totFLOPs = (nRepeats * 2.0f * M * K * N * 1e-9f);
		float totMemAccess = nRepeats * (M * K + K * N + M * N) * sizeof(float) * 1e-9f;
		// clang-format off
		fmt::println(
			R"(--- PERFORMANCE ---
+ {:<15}: {}
+ {:<15}: {}s
+ {:<15}: {}GFLOPS
+ {:<15}: {}GB/s
-------------------)",
			"Size", size,
			"Average runtime", elapsedTimeInSecs / nRepeats,
			"Performance", totFLOPs / elapsedTimeInSecs,
			"Bandwidth", totMemAccess / elapsedTimeInSecs
		);
		// clang-format on

		fflush(stdout);
		// Reset the device memory to be the same as `d_CRef`.
		// Currently, `d_C` and `d_CRef` are not the same, which will cause problems
		// in the next run (with the next `size` in `sizes`).
		CUDA_CHECK(cudaMemcpy(d_C, d_CRef, numbytes, cudaMemcpyDeviceToDevice));
	}

	// --- CLEANUP ---
	// Free the host memory.
	std::free(A);
	std::free(B);
	std::free(C);
	std::free(CRef);

	// Free the device memory.
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	CUDA_CHECK(cudaFree(d_CRef));

	// Destroy the CUDA events.
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
