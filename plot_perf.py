# Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for details.

import json
import os
from typing import Dict, List, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["savefig.bbox"] = "tight"

KERNEL_IDX_TO_NAME = {
    0: "cuBLAS",
    1: "Naive",
    2: "GMEM coalesced",
    3: "SMEM tiling",
}


def read_jsonl(filepath: str) -> Dict[str, List[float]]:
    data = {"matsize": [], "ms": [], "tflops/s": [], "gb/s": [], "flops/b": []}
    with open(filepath, "r") as f:
        for line in f:
            line = json.loads(line)
            data["matsize"].append(line["matsize"])
            data["ms"].append(line["runtime"])
            data["tflops/s"].append(line["flopsThroughput"])
            data["gb/s"].append(line["memThroughput"])
            data["flops/b"].append(line["arithmeticIntensity"])
    return data


def get_data(benchmark_dir: str) -> pd.DataFrame:
    records = []
    for kernel_idx, _ in KERNEL_IDX_TO_NAME.items():
        filepath = os.path.join(benchmark_dir, f"{kernel_idx}.jsonl")
        if not os.path.exists(filepath):
            print(f"WARNING: kernel-{kernel_idx} performance data not found (omitted)")
            continue
        _data = read_jsonl(filepath)
        n = len(_data["matsize"])
        for i in range(n):
            records.append(
                {
                    "matsize": _data["matsize"][i],
                    "ms": _data["ms"][i],
                    "tflops/s": _data["tflops/s"][i],
                    "gb/s": _data["gb/s"][i],
                    "flops/b": _data["flops/b"][i],
                    "kernel_idx": kernel_idx,
                }
            )
    data = pd.DataFrame.from_records(records)
    data["kernel_idx"] = data["kernel_idx"].astype("int32")
    return data


def print_md_table(data: pd.DataFrame, matsize: int) -> None:
    print(f"\nM = K = N = {matsize}")
    print(f"| Kernel | FLOPS throughput (TFLOPs/s) | Relative to cuBLAS |")
    print("|:------:|:---------------------------:|:------------------:|")
    # Only retain rows in df corresponding to the given matsize.
    data = data.loc[data["matsize"] == matsize]
    for _, row in data.iterrows():
        kernel_idx = row["kernel_idx"]
        kernel_name = KERNEL_IDX_TO_NAME[kernel_idx]
        tflops_s = row["tflops/s"]
        cublas_tflops_s = data.loc[data["kernel_idx"] == 0, "tflops/s"].iloc[-1]
        print(
            f"| {kernel_idx}: {kernel_name} "
            f"| {tflops_s:.3e} "
            f"| {tflops_s * 100 / cublas_tflops_s:.3f}% |"
        )
    print("\n")


def plot_perf(
    data: pd.DataFrame,
    output_dir: str,
    perf_colname: str,
    *,
    xscale: Literal["linear", "log", "log2"] = "linear",
    yscale: Literal["linear", "log", "log2"] = "linear",
    also_save_svg: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    _title = ""
    _y_colname = perf_colname
    if perf_colname == "tflops/s":
        _title = " FLOPs throughput"
        _y_colname = "FLOPs throughput\nTFLOPs/s"
    elif perf_colname == "gb/s":
        _title = " memory throughput"
        _y_colname = "Memory throughput\nGB/s"
    elif perf_colname == "ms":
        _title = " runtime"
        _y_colname = "Runtime\nms"
    plt.title(f"SGEMM{_title} [A100 (CUDA 12.9), unlocked clock, 400W]", fontsize=10)
    # cuBLAS is red, the rest are viridis.
    num_kernels = data["kernel_idx"].nunique()
    colors = ["red"] + sns.color_palette("viridis", n_colors=num_kernels - 1)
    plt.xlabel("M = K = N")
    plt.ylabel(_y_colname)
    if xscale == "log2":
        plt.xscale("log", base=2)
    else:
        plt.xscale(xscale)
    if yscale == "log2":
        plt.yscale("log", base=2)
    else:
        plt.yscale(yscale)

    sns.scatterplot(
        x="matsize",
        y=perf_colname,
        data=data,
        hue="kernel_idx",
        palette=colors,
    )
    sns.lineplot(
        x="matsize", y=perf_colname, data=data, hue="kernel_idx", palette=colors
    )
    # Don't use the legend; instead, add the text to the plot.
    max_matsize = data["matsize"].max()
    for _, row in data.iterrows():
        if row["matsize"] != max_matsize:
            continue
        kernel_idx = int(row["kernel_idx"])
        kernel_name = KERNEL_IDX_TO_NAME[kernel_idx]
        plt.text(
            row["matsize"] + 80,
            row[perf_colname],
            f"{kernel_idx}: {kernel_name}",
            fontsize=12,
            ha="left",
            color=colors[kernel_idx],
        )
    plt.xticks(data["matsize"])
    plt.gca().get_legend().remove()  # remove the legend

    perf_colname_no_slash = perf_colname.replace("/", "p")  # "gb/s" -> "gbps"
    plt.savefig(os.path.join(output_dir, f"perf_{perf_colname_no_slash}.png"))
    if also_save_svg:
        plt.savefig(os.path.join(output_dir, f"perf_{perf_colname_no_slash}.svg"))


def plot_flops_roofline(
    data: pd.DataFrame, output_dir: str, also_save_svg: bool = False
):
    print("Plotting flops roofline ...")
    os.makedirs(output_dir, exist_ok=True)

    num_kernels = data["kernel_idx"].nunique()
    colors = ["red"] + sns.color_palette("viridis", n_colors=num_kernels - 1)
    plt.figure(figsize=(12, 6))
    plt.title(f"SGEMM FLOPs roofline [A100 (CUDA 12.9), unlocked clock, 400W]")
    plt.xlabel("Arithmetic intensity\nFLOPs/B")
    plt.ylabel("FLOPS throughput\nTFLOPs/s")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)

    sns.scatterplot(
        x="flops/b",
        y="tflops/s",
        data=data,
        hue="kernel_idx",
        palette=colors,
    )
    sns.lineplot(x="flops/b", y="tflops/s", data=data, hue="kernel_idx", palette=colors)
    max_flops_b = data["flops/b"].max()
    for _, row in data.iterrows():
        if row["flops/b"] != max_flops_b:
            continue
        kernel_idx = int(row["kernel_idx"])
        kernel_name = KERNEL_IDX_TO_NAME[kernel_idx]
        plt.text(
            row["flops/b"] + 30,
            row["tflops/s"],
            f"{kernel_idx}: {kernel_name}",
            fontsize=12,
            ha="left",
            color=colors[kernel_idx],
        )
    plt.gca().get_legend().remove()
    plt.savefig(os.path.join(output_dir, f"flops_roofline.png"))
    if also_save_svg:
        plt.savefig(os.path.join(output_dir, f"flops_roofline.svg"))


if __name__ == "__main__":
    _cwd = os.path.dirname(__file__)
    benchmark_dir = os.path.join(_cwd, "benchmark")
    output_dir = os.path.join(_cwd, "plots")

    data = get_data(benchmark_dir)
    print_md_table(data, matsize=8192)
    for perf_colname in ["tflops/s", "gb/s", "ms"]:
        print(f"Plotting `{perf_colname}` ...")
        plot_perf(
            data=data,
            output_dir=output_dir,
            perf_colname=perf_colname,
            xscale="linear",
            yscale="log",
        )
    plot_flops_roofline(data=data, output_dir=output_dir)
