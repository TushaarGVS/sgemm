# Adapted from: https://github.com/NVIDIA/DALI/blob/main/cmake/CUDA_utils.cmake.

# Use `CMAKE_CUDA_COMPILER` to obtain the path to CUDA toolkit.
# Needed when compiling with Clang only.
function(CUDA_get_toolkit_from_compiler TOOLKIT_PATH)
    get_filename_component(TOOLKIT_PATH_TMP_VAR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
    set(${TOOLKIT_PATH} ${TOOLKIT_PATH_TMP_VAR} PARENT_SCOPE)
endfunction()