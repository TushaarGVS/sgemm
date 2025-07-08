# Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for details.

# Use CMake 3.24+ for `CMAKE_CUDA_ARCHITECTURES="native"`.
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# Utility functions.
include(utils.cmake)

# Set the name, version, and languages of the project.
project(
    sgemm
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# --- 

# Look for a valid CUDA installation (driver, compiler, etc.) and setup variables
# such as `CUDA_FOUND`, `CUDA_INCLUDE_DIRS`, `CUDA_LIBRARIES`, etc.
# Ensure that the CUDA package is found, if not, throw an error.
find_package(CUDAToolkit REQUIRED)

# ---

# Enable the exporting of all compile commands used to build the project to 
# `compile_commands.json`. This is equivalent to `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 
# flag in CMake.
# NOTE: It is not needed to create a symlink to the `compile_commands.json` file in 
# the source directory. We will handle this explicitly in the `.clangd` file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ compiler global flags (apply to all targets).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA compiler global flags (apply to all targets).
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# Automatically set the CUDA compute capability to the native architecture of the 
# GPU. For reference, see: https://developer.nvidia.com/cuda-gpus.
set(CMAKE_CUDA_ARCHITECTURES "native")
# NOTE: Each CUDA version has a maximum supported GCC version. Check the compatibility
# here: https://stackoverflow.com/a/46380601. You can check the GCC version by running:
# `gcc --version`.
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
# ONLY if running in debug mode, remove most of the optimizations.
add_compile_options(
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g;>"
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g;-G;--source-in-ptx;>"
)

message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
# Print the compute capability of the GPU. Adapted from: 
# https://github.com/osayamenja/Kleos/blob/main/csrc/CMakeLists.txt.
# CMAKE_CUDA_ARCHITECTURES_NATIVE prints `xx-real`, so we extract the `xx` part.
string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY)
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU compute capability: ${COMPUTE_CAPABILITY}")

# ---

# Target-specific flags.

# Target-specific C++ flags (will be added later). Adapted from: 
# https://github.com/osayamenja/Kleos/blob/main/csrc/CMakeLists.txt.
set(
    SGEMM_CPP_FLAGS
    -fPIE  # position independent executable: https://stackoverflow.com/q/71761087
    -Wall  # warn about all warnings
    -Wextra  # more warnings
    -Wformat=2  # printf/scanf format warnings
    -Wno-psabi  # ignore ABI warnings
    -fno-strict-aliasing  # no strict aliasing
    -v  # verbose compiler output
)
# When using Clang to compile CUDA, pass CUDA path directly. Adapted from:
# https://github.com/NVIDIA/DALI/blob/main/CMakeLists.txt.
CUDA_get_toolkit_from_compiler(CUDA_TOOLKIT_PATH_VAR)
message(STATUS "Used CUDA toolkit: ${CUDA_TOOLKIT_PATH_VAR}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND SGEMM_CPP_FLAGS "--cuda-path=${CUDA_TOOLKIT_PATH_VAR}")
endif()

# Target-specific CUDA flags (will be added later).
set(
    SGEMM_CUDA_FLAGS
    -Xcudafe=--display_error_number  # display error numbers (flag to cuda frontend)
    -Xfatbin=-compress-all  # no need to generate executables for different GPU archs
    -Xnvlink=--verbose
    -Xptxas=--verbose  # verbose PTX assembly output
    -Xptxas=--warn-on-spills
    -lineinfo
    --use_fast_math
    --expt-extended-lambda
    --expt-relaxed-constexpr
    -gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}
)
# Pass the C++ flags to the CUDA compiler.
foreach(FLAG ${SGEMM_CPP_FLAGS})
    list(APPEND SGEMM_CUDA_FLAGS "-Xcompiler=${FLAG}")
endforeach()

# --- 

# Add the `sgemm` executable.
add_executable(sgemm)
target_sources(
    sgemm 
    PRIVATE 
    csrc/main.cu
    csrc/runner.cu
    csrc/runner.cuh
    csrc/kernels/1_sgemm_naive.cuh
    csrc/kernels/2_sgemm_gmem_coalesce.cuh
    csrc/kernels/3_sgemm_smem_tiling.cuh
)
target_link_libraries(
    sgemm 
    PRIVATE 
    pthread  # enable threading
    dl  # load dynamic libraries
    rt  # load real-time libraries
    CUDA::cudart 
    CUDA::cublas 
)
# Set the output directory to the source directory (it is otherwise set to the 
# `build` directory by default).
set_target_properties(
    sgemm 
    PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add target-specific C++ and CUDA flags, and define macros.
target_compile_options(
    sgemm 
    PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${SGEMM_CPP_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${SGEMM_CUDA_FLAGS}>
)
target_compile_definitions(
    sgemm 
    PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:GPU_ARCH=${GPU_ARCH}>  # usage: `#if GPU_ARCH >= 800`
)

# ---

# Additional packages.
include(FetchContent)
# Force `FetchContent` to try to find the package before downloading and building
# the dependency.
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)

# Add the `fmt` library, which is ~20% faster than `printf`.
# Once added, you can use `fmt::print` instead of `std::cout` to print to the console.
FetchContent_Declare(
    FMT
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
)
# Force set `FMT_SYSTEM_HEADERS=ON` to enable treating `fmt` headers as system headers.
set(FMT_SYSTEM_HEADERS ON CACHE BOOL "Setting FMT_SYSTEM_HEADERS=ON" FORCE)
FetchContent_MakeAvailable(FMT)
target_link_libraries(sgemm PRIVATE fmt::fmt)

# ---
