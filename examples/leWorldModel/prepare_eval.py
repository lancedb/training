"""
Prepare a LeWM checkpoint for evaluation with eval.py.

stable_worldmodel's AutoCostModel expects:
  - checkpoint at $STABLEWM_HOME/<run_name>_object.ckpt
  - policy argument passed as <run_name> (without _object.ckpt suffix)

This script copies/symlinks the checkpoint to the right location and prints
the exact eval.py command to run.

Usage:
  python prepare_eval.py --checkpoint checkpoints/lewm_pusht_lewm_epoch_10_object.ckpt
  python prepare_eval.py --checkpoint checkpoints/lewm_pusht_lewm_epoch_10_object.ckpt --run-name lewm_pusht
"""
import argparse
import os
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to *_object.ckpt file")
    parser.add_argument("--run-name", default=None,
                        help="Name to use under STABLEWM_HOME (default: derived from checkpoint filename)")
    parser.add_argument("--dataset", default="pusht", choices=["pusht", "cube", "reacher", "tworoom"],
                        help="Dataset to evaluate on")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of symlinking (use when src and dst are on different filesystems)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    stablewm_home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    stablewm_home.mkdir(parents=True, exist_ok=True)

    # Derive run_name: strip _object.ckpt suffix if present
    stem = ckpt.stem  # e.g. lewm_pusht_lewm_epoch_10_object
    if stem.endswith("_object"):
        stem = stem[: -len("_object")]  # lewm_pusht_lewm_epoch_10
    run_name = args.run_name or stem

    dst = stablewm_home / f"{run_name}_object.ckpt"

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if args.copy:
        shutil.copy2(ckpt, dst)
        print(f"Copied  {ckpt}")
    else:
        dst.symlink_to(ckpt)
        print(f"Symlinked  {ckpt}")

    print(f"      → {dst}")
    print()
    print("Run evaluation with:")
    print(f"  python eval.py --config-name={args.dataset}.yaml policy={run_name}")


if __name__ == "__main__":
    main()
