"""End-to-end pipeline wall-clock bench.

Reads existing artifacts created by ``scripts/run_pipeline.sh`` and
emits a single JSON record with the time spent in each stage:

    ingest -> tier1_backfill -> tier2_backfill -> tier3_backfill ->
    layouts_export -> train -> eval

Each stage's runner writes a one-line ``stage.json`` file with at
least ``{"wall_s": ..., "rows": ...}`` into its output dir, and this
script collects them.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOG = logging.getLogger("bench.pipeline")


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="dir where stage timing files live (run_pipeline.sh writes them)")
    p.add_argument("--out",     default="bench_outputs/pipeline.json")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    stages = [
        "ingest", "tier1_backfill", "tier2_backfill", "tier3_backfill",
        "layouts_export", "train", "eval",
    ]
    out: dict = {"stages": {}, "total_wall_s": 0.0}
    for stage in stages:
        f = run_dir / f"{stage}.stage.json"
        if not f.exists():
            f = run_dir / f"{stage}.json"  # legacy path
        if f.exists():
            data = json.loads(f.read_text())
            out["stages"][stage] = data
            out["total_wall_s"] += data.get("wall_s", 0.0)
        else:
            LOG.warning("missing %s — skipping", f)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    LOG.info("wrote %s", out_path)

    print()
    print(f"{'stage':<20}  {'rows':>8}  {'wall_s':>8}  {'rows/s':>8}")
    print("-" * 50)
    for stage, data in out["stages"].items():
        rows = data.get("rows", "-")
        ws = data.get("wall_s", 0.0)
        rps = (rows / ws) if isinstance(rows, (int, float)) and ws > 0 else "-"
        rps_str = f"{rps:>8.2f}" if isinstance(rps, float) else f"{rps:>8}"
        print(f"{stage:<20}  {str(rows):>8}  {ws:>8.1f}  {rps_str}")
    print("-" * 50)
    print(f"{'TOTAL':<20}  {'':>8}  {out['total_wall_s']:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
