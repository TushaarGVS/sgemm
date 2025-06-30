# Copyright (C) 2025, Tushaar Gangavarapu <TG352@cornell.edu>.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for details.

import json
import os
from functools import lru_cache
from typing import Dict, List, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["savefig.bbox"] = "tight"

KERNEL_IDX_TO_NAME = {
    0: "cuBLAS",
    1: "Naive",
    2: "GMEM coalesced",
}


@lru_cache(maxsize=None)
def read_jsonl(filepath: str) -> Dict[str, List[float]]:
    data = {"size": [], "runtime": [], "tflops/s": [], "gb/s": []}
    with open(filepath, "r") as f:
        for line in f:
            line = json.loads(line)
            data["size"].append(line["size"])
            data["runtime"].append(line["runtime"])
            data["tflops/s"].append(line["performance"])
            data["gb/s"].append(line["bandwidth"])
    return data


def plot_perf(
    benchmark_dir: str,
    output_dir: str,
    perf_colname: str,
    *,
    xscale: Literal["linear", "log", "log2"] = "linear",
    yscale: Literal["linear", "log", "log2"] = "linear",
    also_save_svg: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    colors = sns.color_palette("viridis")
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

    for kernel_idx, kernel_name in KERNEL_IDX_TO_NAME.items():
        data = read_jsonl(os.path.join(benchmark_dir, f"{kernel_idx}.jsonl"))
        color = colors[kernel_idx] if kernel_idx != 0 else "red"
        sns.lineplot(
            x="size", y=perf_colname, data=data, label=kernel_name, color=color
        )
        plt.scatter(data["size"], data[perf_colname], label=kernel_name, color=color)
        # Don't use the legend; instead, add the text to the plot.
        plt.text(
            data["size"][-1] + 50,
            data[perf_colname][-1],
            f"{kernel_idx}: {kernel_name}",
            fontsize=12,
            ha="left",
            color=color,
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
    for perf_colname in ["tflops/s", "gb/s"]:
        print(f"Plotting `{perf_colname}` ...")
        plot_perf(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            perf_colname=perf_colname,
            xscale="log2",
            yscale="log",
            also_save_svg=True if perf_colname == "tflops/s" else False,
        )
