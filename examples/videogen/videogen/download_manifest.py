"""
Download just the **caption manifest** for ChronoMagic-Pro or ChronoMagic-ProH.

The full ChronoMagic-Pro corpus is multi-TB of video.  The HF dataset is
ultimately `(videoid, name)` pairs where `videoid` is a YouTube ID — the
video bytes are *not* in the HF repo.  This script pulls just the small
auto-converted Parquet so you can:

  * run the keyword + FTS curation immediately with no clip download
  * pick the subset you want to actually fetch via yt-dlp later

Usage
-----
python -m videogen.download_manifest --variant pro   --out data/chronomagic_pro.parquet
python -m videogen.download_manifest --variant proh  --out data/chronomagic_proh.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIANTS = {
    # HF auto-converts the CSV to Parquet under refs/convert/parquet.
    "pro":  "BestWishYsh/ChronoMagic-Pro",
    "proh": "BestWishYsh/ChronoMagic-ProH",
}


def download(variant: str, out: Path) -> None:
    repo = VARIANTS[variant]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading manifest parquet for {repo} → {out}")
    print("  (this is just the caption + videoid table, no clips)")

    try:
        # Streaming-friendly path: the HF datasets library will pull the
        # auto-converted parquet and we materialise it locally.
        # ChronoMagic-Pro and -ProH publish their data under the "test"
        # split despite the contents being a full training corpus —
        # detect the split at runtime rather than hardcoding.
        from datasets import get_dataset_split_names, load_dataset

        splits = get_dataset_split_names(repo)
        split = "train" if "train" in splits else splits[0]
        print(f"  using split '{split}' (available: {splits})")
        ds = load_dataset(repo, split=split, streaming=True)
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for i, row in enumerate(ds):
            rows.append({
                "videoid": row.get("videoid") or row.get("id"),
                "caption": row.get("name") or row.get("caption"),
            })
            if (i + 1) % 50_000 == 0:
                print(f"  {i + 1:,} rows…")
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, str(out), compression="zstd")
        print(f"Wrote {len(tbl):,} rows → {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"\nERROR — falling back to manual URL is needed: {e}", file=sys.stderr)
        print(
            "  HF dataset page: https://huggingface.co/datasets/" + repo + "\n"
            "  You can also `huggingface-cli download` directly.", file=sys.stderr,
        )
        raise


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="proh",
                   help="`pro` = 466K full ChronoMagic-Pro;  "
                        "`proh` = 144K higher-quality subset (default).")
    p.add_argument("--out", type=Path, default=Path("data/chronomagic_proh.parquet"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    download(args.variant, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
