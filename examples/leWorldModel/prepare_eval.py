"""
Prepare a LeWM checkpoint for evaluation with eval.py.

stable_worldmodel's AutoCostModel expects:
  - checkpoint at $STABLEWM_HOME/<run_name>_object.ckpt
  - policy argument passed as <run_name> (without _object.ckpt suffix)

eval.py also needs the source HDF5 file at $STABLEWM_HOME/<dataset_hdf5_name>.h5
to sample episode starting states and goals. This script downloads it from
HuggingFace automatically if it is not already present.

This script handles all of the above and prints the exact eval.py command to run.

Usage:
  python prepare_eval.py --checkpoint checkpoints/lewm_pusht_lewm_epoch_10_object.ckpt
  python prepare_eval.py --checkpoint checkpoints/lewm_pusht_lewm_epoch_10_object.ckpt --run-name lewm_pusht
"""
import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path


# HuggingFace repo and expected HDF5 filename for each dataset
_DATASET_META = {
    "pusht":    ("quentinll/lewm-pusht",    "pusht_expert_train.h5"),
    "cube":     ("quentinll/lewm-cube",     "cube_single_expert.h5"),
    "reacher":  ("quentinll/lewm-reacher",  "reacher.h5"),
    "tworoom":  ("quentinll/lewm-tworooms", "tworoom.h5"),
}


def _ensure_hdf5(dataset: str, stablewm_home: Path) -> Path:
    """
    Ensure the source HDF5 file is present at $STABLEWM_HOME/<name>.h5.
    Downloads from HuggingFace if missing.
    """
    hf_repo, hdf5_name = _DATASET_META[dataset]
    dst = stablewm_home / hdf5_name
    if dst.exists():
        return dst

    # Check if already cached from a previous create_data.py run
    cache_dir = stablewm_home / "datasets" / hf_repo.replace("/", "--")
    existing = glob.glob(str(cache_dir / "*.h5")) + glob.glob(str(cache_dir / "*.hdf5"))
    if existing:
        dst.symlink_to(existing[0])
        print(f"Linked HDF5 from cache → {dst}")
        return dst

    # Download from HuggingFace
    print(f"HDF5 not found. Downloading {hf_repo} from HuggingFace...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    repo_files = list(list_repo_files(hf_repo, repo_type="dataset"))
    data_file = next(
        (f for f in repo_files
         if f.endswith(".tar.zst") or f.endswith(".h5.zst")
         or f.endswith(".h5") or f.endswith(".hdf5")),
        None,
    )
    if not data_file:
        raise FileNotFoundError(f"No HDF5 file found in HuggingFace repo {hf_repo}")

    local = hf_hub_download(
        repo_id=hf_repo, filename=data_file,
        repo_type="dataset", local_dir=str(cache_dir),
    )

    if local.endswith(".tar.zst"):
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", local, "-C", str(cache_dir)],
            check=True,
        )
        os.remove(local)
    elif local.endswith(".h5.zst"):
        out = local[:-4]
        subprocess.run(["zstd", "-d", local, "-o", out], check=True)
        os.remove(local)

    h5_files = glob.glob(str(cache_dir / "*.h5")) + glob.glob(str(cache_dir / "*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in {cache_dir} after download")

    dst.symlink_to(h5_files[0])
    print(f"Downloaded and linked HDF5 → {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to *_object.ckpt file")
    parser.add_argument("--run-name", default=None,
                        help="Name to use under STABLEWM_HOME (default: derived from checkpoint filename)")
    parser.add_argument("--dataset", default="pusht", choices=list(_DATASET_META),
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

    # 1. Link checkpoint
    dst = stablewm_home / f"{run_name}_object.ckpt"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if args.copy:
        shutil.copy2(ckpt, dst)
        print(f"Copied    {ckpt}")
    else:
        dst.symlink_to(ckpt)
        print(f"Symlinked {ckpt}")
    print(f"       → {dst}")

    # 2. Ensure HDF5 is present
    _ensure_hdf5(args.dataset, stablewm_home)

    print()
    print("Run evaluation with:")
    print(f"  python eval.py --config-name={args.dataset}.yaml policy={run_name}")


if __name__ == "__main__":
    main()
