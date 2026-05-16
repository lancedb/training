"""
B10 — End-to-end wall-clock for the full videogen pipeline.

Times every stage and prints a single summary table.  Operates on an
existing Lance DB that's already had ChronoMagic-Pro / -ProH ingested
(via ``videogen.ingest_chronomagic --manifest … --video-dir …``).

Stages timed:

  tier1           caption keyword UDFs (CPU)
  tier2           CLIP text/video, motion, MTScore (GPU)
  tier3 t5        UMT5-XXL last hidden state
  tier3 vae       Wan2.2-VAE latent
  tier4 dhash     GPU first+last frame dHash
  tier4 idx       L2 index build on dhash_first_last
  tier4 dup       is_duplicate per-row NN lookup
  curate t1       per-transition Geneva MVs
  train           N LoRA training steps with the cached dataloader

Each stage is its own subprocess so a crash in one (say, GPU OOM on
Tier-3) doesn't kill the rest of the run.

Usage
-----
python -m bench.bench_e2e --db data/videos/lancedb --train-steps 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _run(label: str, cmd: list[str], *, env_extra: dict | None = None,
         silent: bool = True) -> float:
    print(f"\n[{label}] running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    if silent:
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-1500:])
            print("STDERR:")
            print(out.stderr[-1500:])
            raise SystemExit(f"[{label}] failed with rc={out.returncode}")
    else:
        out = subprocess.run(cmd)
        if out.returncode != 0:
            raise SystemExit(f"[{label}] failed with rc={out.returncode}")
    dt = time.perf_counter() - t0
    print(f"  done in {dt:.2f}s")
    return dt


def _py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db",          default="data/videos/lancedb")
    p.add_argument("--train-steps", type=int, default=10)
    p.add_argument("--train-view",  default="videos_raw")
    args = p.parse_args(argv)

    timings: dict[str, float] = {}

    # ---- 2) Tier 1 ---------------------------------------------------------
    timings["tier1"] = _run(
        "tier1",
        _py("videogen.backfill_geneva", "--tier", "1", "--db", args.db),
    )

    # ---- 3) Tier 2 (CLIP + motion + MTScore) -------------------------------
    timings["tier2"] = _run(
        "tier2",
        _py("videogen.backfill_geneva", "--tier", "2", "--db", args.db),
    )

    # ---- 4) Tier 3 — split into two for clarity ---------------------------
    timings["tier3 t5"] = _run(
        "tier3 t5",
        _py("videogen.backfill_geneva", "--columns", "t5_hidden_states",
            "--db", args.db),
    )
    timings["tier3 vae"] = _run(
        "tier3 vae",
        _py("videogen.backfill_geneva", "--columns", "vae_latent",
            "--db", args.db),
    )

    # ---- 5) Tier 4 — dedup -------------------------------------------------
    timings["tier4 dhash"] = _run(
        "tier4 dhash",
        _py("videogen.backfill_geneva", "--columns", "dhash_first_last",
            "--db", args.db),
    )
    # Build the L2 index inline so the next step can vector-search.
    t0 = time.perf_counter()
    print("\n[tier4 idx] building L2 index on dhash_first_last …")
    import lancedb
    tbl = lancedb.connect(args.db).open_table("videos_raw")
    try:
        tbl.create_index(metric="l2",
                         vector_column_name="dhash_first_last",
                         index_type="IVF_FLAT",
                         num_partitions=max(2, len(tbl) // 32))
    except Exception as e:
        print(f"  index build warning: {e}")
    timings["tier4 idx"] = time.perf_counter() - t0
    print(f"  done in {timings['tier4 idx']:.2f}s")
    timings["tier4 dup"] = _run(
        "tier4 dup",
        _py("videogen.backfill_geneva", "--columns", "is_duplicate",
            "--db", args.db),
    )

    # ---- 6) Curate ---------------------------------------------------------
    timings["curate t1"] = _run(
        "curate t1",
        _py("videogen.manage_views", "--action", "curate", "--db", args.db),
    )

    # ---- 7) Train ----------------------------------------------------------
    timings["train"] = _run(
        "train",
        _py("videogen.train_wan22_lora",
            "--db", args.db,
            "--train-view", args.train_view,
            "--steps", str(args.train_steps),
            "--batch-size", "1",
            "--num-workers", "0",
            "--log-every", str(max(args.train_steps, 1)),
            "--output-dir", f"{args.db}/_e2e_ckpt"),
    )

    # ---- Summary -----------------------------------------------------------
    total = sum(timings.values())
    print("\n" + "─" * 64)
    print(f"  {'stage':<14}  {'wall-clock':>10}  {'% of total':>10}")
    print("  " + "─" * 38)
    for k, v in timings.items():
        print(f"  {k:<14}  {v:>9.2f}s  {v / total * 100:>9.1f}%")
    print("  " + "─" * 38)
    print(f"  {'total':<14}  {total:>9.2f}s")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
