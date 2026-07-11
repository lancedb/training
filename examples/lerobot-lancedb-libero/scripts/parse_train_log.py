#!/usr/bin/env python
"""Parse lerobot-train console logs into CSV (step, loss, grad_norm, lr, update_s, data_s).

Usage: python parse_train_log.py train.log out.csv
"""

import csv
import re
import sys

# example: "step:200 smpl:13K ep:8 epch:0.05 loss:0.123 grdn:1.234 lr:1.0e-04 updt_s:0.451 data_s:0.012"
PAT = re.compile(
    r"step:(?P<step>\d+(?:\.\d+)?[KM]?) .*?loss:(?P<loss>[\d.eE+-]+).*?grdn:(?P<grdn>[\d.eE+-]+)"
    r".*?lr:(?P<lr>[\d.eE+-]+).*?updt_s:(?P<updt>[\d.eE+-]+).*?data_s:(?P<data>[\d.eE+-]+)"
)

MULT = {"K": 1_000, "M": 1_000_000}


def expand(step: str) -> int:
    if step and step[-1] in MULT:
        return int(float(step[:-1]) * MULT[step[-1]])
    return int(step)


def main() -> None:
    log_path, out_path = sys.argv[1], sys.argv[2]
    rows = []
    with open(log_path, errors="replace") as f:
        text = f.read().replace("\r", "\n")
    for line in text.splitlines():
        m = PAT.search(line)
        if m:
            d = m.groupdict()
            d["step"] = expand(d["step"])
            rows.append(d)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "grdn", "lr", "updt", "data"])
        w.writeheader()
        w.writerows(rows)
    print(f"parsed {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
