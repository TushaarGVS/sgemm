# Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for details.

# For details about phony targets, see: https://stackoverflow.com/a/2145605.
.PHONY: all build debug clean realclean profile bench 

CMAKE := cmake
BUILD_DIR := build
PROFILE_DIR := profile
BENCHMARK_DIR := benchmark
PLOTS_DIR := plots

# Clean, then build in release mode and benchmark.
# Usage: `make all`.
all: clean build bench

# Build in release mode.
# Usage: `make build`.
# Aside: Don't indent comments within target rules. See: 
# https://stackoverflow.com/a/18363477.
build:
	@echo "Building to \`$(BUILD_DIR)\` dir ..."
# Create a build dir if it doesn't already exist. 
# The `-p` flag ensures that no error is thrown if the directory already exists. 
# (Think of this as the `exists_ok=True` parameter in Python's `os.makedirs`.)
	@mkdir -p $(BUILD_DIR)
# Go to the build dir and run `cmake ..`, which configures the project whose 
# `CMakeLists.txt` is in the parent directory.
# The `-DCMAKE_BUILD_TYPE=Release` flag tells CMake to build the project in
# release mode, i.e., with optimizations and no debug symbols.
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
# Run `make` in the build dir to build the project.
# The `-C` flag tells `make` to run in the build dir.
	@$(MAKE) -C $(BUILD_DIR)
	@echo "Build complete."

# Build in debug mode.
# Usage: `make debug`.
debug:
	@echo "Building in *debug* mode to \`$(BUILD_DIR)\` dir ..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug ..
	@$(MAKE) -C $(BUILD_DIR)
	@echo "Debug build complete."

# Clean up the build dir.
# Usage: `make clean`.
clean:
	@echo "Removing \`$(BUILD_DIR)\` dir ..."
	@rm -rf $(BUILD_DIR)
	@echo "Done."

# Remove the build, profile, benchmark, and plots dirs.
# Usage: `make realclean`.
realclean:
	@echo "Removing ..."
	@echo " + \`$(BUILD_DIR)\`"
	@echo " + \`$(PROFILE_DIR)\`"
	@echo " + \`$(BENCHMARK_DIR)\`"
	@echo " + \`$(PLOTS_DIR)\`"
	@rm -rf $(BUILD_DIR)
	@rm -rf $(PROFILE_DIR)
	@rm -rf $(BENCHMARK_DIR)
	@rm -rf $(PLOTS_DIR)
	@echo "Done."

# Profiling a kernel using NVIDIA Nsight Compute.
# Once built, `./sgemm` will be the executable (see `add_executable` in CMakeLists.txt).
# Usage: `make profile KERNEL=<kernel-num>`.
profile:
	@echo "Profiling kernel-$(KERNEL) ..."
	@mkdir -p $(PROFILE_DIR)
# --set full: use the full metric set, i.e., memory throughput, SM util, etc.
	@ncu \
		--set full \
		--export $(PROFILE_DIR)/$(KERNEL) \
		--force-overwrite \
		./sgemm $(KERNEL)
	@ncu-ui $(PROFILE_DIR)/$(KERNEL).ncu-rep

# Benchmarking the kernel.
# Usage: `make bench [KERNEL=<kernel-num1>,<kernel-num2>,...]`.
bench:
	@echo "Benchmarking kernels to \`$(BENCHMARK_DIR)\` dir ..."
	@mkdir -p $(BENCHMARK_DIR)
	@count=$(shell ls ./csrc/kernels | wc -l); \
	if [ -z "$(KERNEL)" ]; then \
		KERNEL_LIST=$$(seq 0 $$count); \
	else \
		KERNEL_LIST=$$(echo "$$KERNEL" | tr ',' '\n'); \
	fi; \
	for kernel in $$KERNEL_LIST; do \
		echo "Benchmarking kernel-$$kernel ..." && \
		./sgemm $$kernel | \
		awk '\
			/--- PERFORMANCE ---/ {capture=1; next} \
			capture && /^\+\s/ { \
				key = $$2; \
				val = $$4; \
				gsub("s|GFLOPs/s|TFLOPs/s|GB/s|TB/s", "", val); \
				metrics[key] = val; \
				next \
			} \
			/-------------------/ && capture { \
				printf("{"); \
				i = 0; \
				for (m in metrics) { \
					if (i++ > 0) printf(", "); \
					printf("\"%s\": %s", m, metrics[m]); \
				} \
				printf("}\n"); \
				capture = 0; \
				delete metrics; \
			} \
		' > $(BENCHMARK_DIR)/$$kernel.jsonl; \
		sleep 2; \
	done
	@echo "Benchmarking complete, plotting results to \`$(PLOTS_DIR)\` dir ..."
	@python3 plot_perf.py
	@echo "Done."
