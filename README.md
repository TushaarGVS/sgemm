# SGEMM

Accompanying repo for the blog post: https://tushaargvs.github.io/posts/matmul.html.

> This repo was adapted from:
>
> - https://github.com/siboehm/SGEMM_CUDA and
> - https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE.

## Installation

Install dependencies in `env.yml`:

```shell
# Create and activate a new env called `sgemm-env`.
conda env create -f env.yml
# If already created, update the env (and remove unused packages).
conda env update -f env.yml --prune
conda activate sgemm-env

# Ensure that the installation was successful by checking the CUDA version.
nvcc --version
# Also, check to make sure all cuda-related packages are installed.
```

If you wish to delete the `sgemm-env` environment, run:

```shell
conda env remove -n sgemm-env --all
```

## Running

Build the project and run it:

```shell
make [-B] build
[DEVICE=<device-idx>] ./sgemm <kernel-num>
```

To profile a kernel, run:

```shell
make profile KERNEL=<kernel-num>
```

NOTE: You might need sudo access to run `ncu` since it accesses the GPU Performance
Counters. For more info, see:
https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters.

To benchmark all kernels, run:

```shell
make bench
```
