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
    3: "SMEM blocking",
}


def read_jsonl(filepath: str) -> Dict[str, List[float]]:
    data = {"size": [], "runtime": [], "tflops/s": [], "gb/s": []}
    with open(filepath, "r") as f:
        for line in f:
            line = json.loads(line)
            data["size"].append(line["size"])
            data["runtime"].append(line["runtime"])
            data["tflops/s"].append(line["flopsThroughput"])
            data["gb/s"].append(line["memThroughput"])
    return data


def get_data(benchmark_dir: str) -> pd.DataFrame:
    records = []
    for kernel_idx, _ in KERNEL_IDX_TO_NAME.items():
        filepath = os.path.join(benchmark_dir, f"{kernel_idx}.jsonl")
        if not os.path.exists(filepath):
            print(f"WARNING: kernel-{kernel_idx} performance data not found (omitted)")
            continue
        _data = read_jsonl(filepath)
        n = len(_data["size"])
        for i in range(n):
            records.append(
                {
                    "size": _data["size"][i],
                    "runtime": _data["runtime"][i],
                    "tflops/s": _data["tflops/s"][i],
                    "gb/s": _data["gb/s"][i],
                    "kernel_idx": kernel_idx,
                }
            )
    data = pd.DataFrame.from_records(records)
    return data


def print_md_table(data: pd.DataFrame) -> None:
    print("| Kernel | FLOPS throughput (TFLOPs/s) | Relative to cuBLAS |")
    print("|:------:|:---------------------------:|:------------------:|")
    for kernel_idx, kernel_name in KERNEL_IDX_TO_NAME.items():
        tflops_s = data.loc[data["kernel_idx"] == kernel_idx, "tflops/s"].iloc[-1]
        cublas_tflops_s = data.loc[data["kernel_idx"] == 0, "tflops/s"].iloc[-1]
        print(
            f"| {kernel_idx}: {kernel_name} "
            f"| {tflops_s:.3e} "
            f"| {tflops_s * 100/ cublas_tflops_s:.3f}% |"
        )
    print()


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

    plt.figure(figsize=(12, 5))
    # cuBLAS is red, the rest are viridis.
    colors = ["red"] + sns.color_palette(
        "viridis", n_colors=len(KERNEL_IDX_TO_NAME) - 1
    )
    plt.xlabel("M = K = N")
    plt.ylabel(perf_colname)
    if xscale == "log2":
        plt.xscale("log", base=2)
    else:
        plt.xscale(xscale)
    if yscale == "log2":
        plt.yscale("log2")
    else:
        plt.yscale(yscale)

    sns.scatterplot(
        x="size",
        y=perf_colname,
        data=data,
        hue="kernel_idx",
        palette=colors,
    )
    sns.lineplot(x="size", y=perf_colname, data=data, hue="kernel_idx", palette=colors)
    # Don't use the legend; instead, add the text to the plot.
    for kernel_idx, kernel_name in KERNEL_IDX_TO_NAME.items():
        plt.text(
            data.loc[data["kernel_idx"] == kernel_idx, "size"].iloc[-1] + 300,
            data.loc[data["kernel_idx"] == kernel_idx, perf_colname].iloc[-1],
            f"{kernel_idx}: {kernel_name}",
            fontsize=12,
            ha="left",
            color=colors[kernel_idx],
        )
    plt.xticks(data["size"])
    plt.gca().get_legend().remove()  # remove the legend.

    perf_colname_no_slash = perf_colname.replace("/", "p")  # "gb/s" -> "gbps"
    plt.savefig(os.path.join(output_dir, f"perf_{perf_colname_no_slash}.png"))
    if also_save_svg:
        plt.savefig(os.path.join(output_dir, f"perf_{perf_colname_no_slash}.svg"))


if __name__ == "__main__":
    _cwd = os.path.dirname(__file__)
    benchmark_dir = os.path.join(_cwd, "benchmark")
    output_dir = os.path.join(_cwd, "plots")

    data = get_data(benchmark_dir)
    print_md_table(data)
    for perf_colname in ["tflops/s", "gb/s"]:
        print(f"Plotting `{perf_colname}` ...")
        plot_perf(
            data=data,
            output_dir=output_dir,
            perf_colname=perf_colname,
            xscale="log2",
            yscale="log",
            also_save_svg=True if perf_colname == "tflops/s" else False,
        )
