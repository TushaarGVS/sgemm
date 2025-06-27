# Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for details.

# For details about phony targets, see: https://stackoverflow.com/a/2145605.
.PHONY: all build debug clean profile bench 

CMAKE := cmake
BUILD_DIR := build
PROFILE_DIR := profile
BENCHMARK_DIR := benchmark

# Build in release mode and benchmark.
# Usage: `make all`
all: build bench

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
# Remove the build dir.
	@echo "Removing \`$(BUILD_DIR)\` dir ..."
	@rm -rf $(BUILD_DIR)
	@echo "\`$(BUILD_DIR)\` dir removed."

# Profiling a kernel using NVIDIA Nsight Compute.
# Once built, `./sgemm` will be the executable (see `add_executable` in CMakeLists.txt).
# Usage: `make profile KERNEL=<int>`.
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
# Usage: `make bench`.
bench:
	@echo "Benchmarking to \`$(BENCHMARK_DIR)\` dir ..."
	@mkdir -p $(BENCHMARK_DIR)
	@bash benchmark.sh
