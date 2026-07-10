#!/usr/bin/env python
"""Generate all perf-section figures for the blog/example README.

Inputs (produced by bench_throughput.py / gpu_monitor.py / parse_train_log.py):
  ~/work/logs/bench_results.json   local loader matrix
  ~/work/logs/bench_s3.json        s3 streaming rows
  ~/work/logs/gpu_{base,lance}.csv GPU util samples during the two runs
  ~/work/logs/train_{base,lance}.csv parsed training logs

Outputs PNGs into --out-dir.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"image": "#8c8c94", "video": "#5b8def", "lance": "#e8590c"}
LABELS = {
    "image": "LeRobot parquet (image dtype, 33 GB)",
    "video": "LeRobot parquet+mp4 (1.9 GB)",
    "lance": "Lance video format (1.9 GB)",
}


def fig_throughput(logs: Path, out: Path):
    rows = json.loads((logs / "bench_results.json").read_text())
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for backend, g in df.groupby("backend"):
        g = g.sort_values("num_workers")
        ax.plot(g.num_workers, g.samples_per_s, "o-", color=COLORS[backend], label=LABELS[backend])
        for _, r in g.iterrows():
            ax.annotate(f"{r.samples_per_s:.0f}", (r.num_workers, r.samples_per_s),
                        textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_yscale("log")
    ax.set_xticks([4, 8, 16])
    ax.set_xlabel("DataLoader workers")
    ax.set_ylabel("samples / s (log scale)")
    ax.set_title("LIBERO dataloading throughput — SmolVLA read pattern (2 cams + 50-step action chunk)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "throughput_local.png")


def fig_gpu_util(logs: Path, out: Path):
    # nvidia-smi "utilization" counts NCCL busy-wait as busy, so starved DDP
    # ranks still read ~70%. Power draw separates real work from spinning.
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), dpi=150, sharey=True, sharex=True)
    for ax, run, color, name in [
        (axes[0], "base", COLORS["image"], "Base LeRobot (official parquet dataset)"),
        (axes[1], "lance", COLORS["lance"], "lerobot-lancedb (Lance video format)"),
    ]:
        df = pd.read_csv(logs / f"gpu_{run}.csv")
        df["t"] = df.wall_time - df.wall_time.min()
        mean = df.groupby("t").power_w.mean().rolling(15, min_periods=1).mean()
        ax.plot(mean.index / 60, mean.values, color=color, lw=1)
        ax.fill_between(mean.index / 60, 0, mean.values, color=color, alpha=0.25)
        avg = df.power_w.mean()
        dur = df.t.max() / 60
        kwh = avg * 4 * dur * 60 / 3.6e6
        ax.axhline(avg, ls="--", color="black", lw=0.8)
        ax.set_ylabel("GPU power W (4-GPU mean)")
        ax.set_title(f"{name} — {dur:.0f} min, avg {avg:.0f} W, {kwh:.2f} kWh", fontsize=10)
        ax.set_ylim(0, 700)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("wall-clock minutes")
    fig.suptitle("Same 20k-step SmolVLA finetune, 4×H100 — GPUs working vs GPUs waiting")
    fig.tight_layout()
    fig.savefig(out / "gpu_util.png")


def fig_loss(logs: Path, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for run, color, label in [("base", COLORS["image"], "base LeRobot"), ("lance", COLORS["lance"], "lerobot-lancedb")]:
        df = pd.read_csv(logs / f"train_{run}.csv")
        ax.plot(df.step, df.loss.astype(float).rolling(5, min_periods=1).mean(), color=color, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("Same data, same loss curve — storage format changes nothing about learning")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_parity.png")


def fig_data_time(logs: Path, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for run, color, label in [("base", COLORS["image"], "base LeRobot"), ("lance", COLORS["lance"], "lerobot-lancedb")]:
        df = pd.read_csv(logs / f"train_{run}.csv")
        total = df.updt.astype(float) + df.data.astype(float)
        frac = df.data.astype(float) / total * 100
        ax.plot(df.step, frac.rolling(5, min_periods=1).mean(), color=color, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("% of step time waiting on data")
    ax.set_title("Fraction of each training step spent waiting for the dataloader")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "data_wait.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=str(Path.home() / "work/logs"))
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    logs, out = Path(args.logs), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_throughput(logs, out)
    try:
        fig_gpu_util(logs, out)
        fig_loss(logs, out)
        fig_data_time(logs, out)
    except FileNotFoundError as e:
        print("skipping (missing input):", e)
    print("figures ->", out)
