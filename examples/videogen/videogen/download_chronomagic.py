"""
Download the full **ChronoMagic** dataset (the base 2,265-clip subset).

Unlike ChronoMagic-Pro / -ProH (which only ship YouTube IDs and need
yt-dlp), the base ``BestWishYsh/ChronoMagic`` repo ships every clip as
an actual mp4 inside a single ``video/video.zip`` (~2.5 GB).

This script:
  1. Downloads ``video/video.zip`` and ``caption/ChronoMagic_train.csv``
     from the HF Hub.
  2. Unzips the clips into ``--out`` (one mp4 per ``videoid``).
  3. Writes a parquet manifest in the same shape as
     ``download_manifest.py`` outputs — drop-in to ``ingest_chronomagic``.

Usage
-----
python -m videogen.download_chronomagic \\
    --out      data/clips \\
    --manifest data/chronomagic.parquet
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq


REPO = "BestWishYsh/ChronoMagic"


def _download(zip_dir: Path) -> tuple[Path, Path]:
    print(f"Downloading {REPO}/video/video.zip and ChronoMagic_train.csv …")
    t0 = time.time()
    zp = hf_hub_download(
        REPO, filename="video/video.zip", repo_type="dataset",
        local_dir=str(zip_dir), local_dir_use_symlinks=False,
    )
    cp = hf_hub_download(
        REPO, filename="caption/ChronoMagic_train.csv", repo_type="dataset",
        local_dir=str(zip_dir), local_dir_use_symlinks=False,
    )
    print(f"  done in {time.time() - t0:.1f}s")
    return Path(zp), Path(cp)


def _unzip(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} → {out_dir} …")
    t0 = time.time()
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".mp4")]
        for m in members:
            # Some zips nest inside a `video/` directory; strip it
            target = out_dir / Path(m).name
            if target.exists() and target.stat().st_size > 1024:
                continue
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            n += 1
    print(f"  extracted {n} new mp4s in {time.time() - t0:.1f}s "
          f"(directory now has {len(list(out_dir.glob('*.mp4')))} clips)")
    return n


def _build_manifest(csv_path: Path, manifest_path: Path, out_dir: Path) -> int:
    """Read the captions CSV and emit a manifest parquet matching
    ``download_manifest.py``'s format (videoid + caption columns).

    Drops rows whose mp4 isn't actually in ``out_dir``.
    """
    print(f"Reading captions from {csv_path.name} …")
    available = {p.stem for p in out_dir.glob("*.mp4")}

    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            vid = r.get("videoid") or r.get("video_id") or r.get("id")
            cap = r.get("name") or r.get("caption") or ""
            if not vid:
                continue
            # Some entries use vid as the filename; some include extensions
            stem = Path(vid).stem
            if stem not in available:
                continue
            rows.append({"videoid": vid, "caption": cap})

    if not rows:
        raise SystemExit(
            f"No captioned rows match the on-disk clips in {out_dir}.  "
            f"Expected stems like {list(available)[:3]}…"
        )

    t = pa.Table.from_pylist(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, str(manifest_path), compression="zstd")
    print(f"  wrote {len(rows):,} manifest rows → {manifest_path}")
    return len(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download ChronoMagic mp4 clips + captions.")
    p.add_argument("--out",      type=Path, default=Path("data/clips"),
                   help="Directory to extract mp4 clips into.")
    p.add_argument("--manifest", type=Path, default=Path("data/chronomagic.parquet"),
                   help="Manifest parquet to emit (drop-in for ingest_chronomagic).")
    p.add_argument("--zip-dir",  type=Path, default=Path("data/chronomagic_raw"),
                   help="Where to download the raw zip + csv.")
    p.add_argument("--skip-unzip", action="store_true",
                   help="If the mp4s are already extracted, skip the unzip step "
                        "and just (re)build the manifest.")
    args = p.parse_args(argv)

    if not args.skip_unzip:
        zip_path, csv_path = _download(args.zip_dir)
        _unzip(zip_path, args.out)
    else:
        csv_path = args.zip_dir / "caption" / "ChronoMagic_train.csv"
        if not csv_path.exists():
            _, csv_path = _download(args.zip_dir)

    n = _build_manifest(csv_path, args.manifest, args.out)
    print(f"\n  → {n:,} clips ready. Next:")
    print(f"     python -m videogen.ingest_chronomagic \\\n"
          f"         --manifest {args.manifest} --video-dir {args.out} \\\n"
          f"         --require-clips --overwrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
