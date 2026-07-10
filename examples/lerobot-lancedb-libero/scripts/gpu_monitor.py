#!/usr/bin/env python
"""Sample GPU utilization / power / memory to CSV via nvidia-smi.

Usage: python gpu_monitor.py out.csv [interval_s]
Runs until killed. One row per GPU per sample.
"""

import subprocess
import sys
import time

QUERY = "timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,sm_clock"


def main() -> None:
    out_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    with open(out_path, "w", buffering=1) as f:
        f.write("wall_time,gpu,util_pct,mem_util_pct,mem_used_mib,power_w,sm_mhz\n")
        while True:
            t = time.time()
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                f.write(f"{t:.1f},{','.join(parts)}\n")
            time.sleep(max(0.0, interval - (time.time() - t)))


if __name__ == "__main__":
    main()
