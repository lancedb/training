"""Run the full rollout-store benchmark matrix and print a results table."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

COLS = [
    ("payload_mb", "payload MB"),
    ("disk_mb", "on disk MB"),
    ("write_mb_s", "write MB/s"),
    ("rss_after_write_mb", "RAM held MB"),
    ("p1_cold_p50_ms", "P1 fetch cold p50 ms"),
    ("p1_warm_p50_ms", "P1 fetch warm p50 ms"),
    ("p2_cold_s", "P2 sweep cold s"),
    ("p2_warm_s", "P2 sweep warm s"),
    ("p3_cold_ms", "P3 batch cold ms"),
    ("p3_warm_ms", "P3 batch warm ms"),
    ("peak_rss_mb", "peak RSS MB"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="memory,pickle,json,parquet,lance")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--per-step", type=int, default=16)
    ap.add_argument("--root", default="./bench_data")
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    rows = []
    for name in args.backends.split(","):
        out = os.path.join(args.results, f"{name}.json")
        cmd = [
            sys.executable, "bench_one.py",
            "--backend", name,
            "--steps", str(args.steps),
            "--per-step", str(args.per_step),
            "--root", args.root,
            "--out", out,
        ]
        print(f"\n=== {name} ===", flush=True)
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        rows.append(json.load(open(out)))

    # markdown table
    header = "| backend | " + " | ".join(label for _, label in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    lines = [header, sep]
    for r in rows:
        cells = [str(r.get(key, "—")) for key, _ in COLS]
        lines.append(f"| {r['backend']} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(args.results, "results.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
