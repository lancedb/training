"""
Download ChronoMagic-Pro clips via yt-dlp into a directory ingest_chronomagic
can read with ``--video-dir``.

The dataset's ``videoid`` column is a YouTube ID, not a URL.  We turn each
id into a YouTube watch URL and download a tight clip into
``<out>/<videoid>.mp4``.  Errors per id (private, deleted, geo-blocked,
age-gated) are logged and the script moves on; the rest of the pipeline
treats missing files as "no video bytes for this row".

This is intentionally a thin wrapper — yt-dlp has hundreds of knobs
(quality, codec, retries, cookies); we expose the ones we care about
for a curation-driven training run and leave the rest as yt-dlp
defaults.

Usage
-----
# Pull 200 clips listed in the manifest into data/videos/raw/
python -m videogen.download_clips \\
    --manifest data/chronomagic_proh.parquet \\
    --out      data/videos/raw \\
    --limit 200 --quality 480

# Restart-safe: already-downloaded ids are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def _yt_dlp_command(yt_dlp: str, url: str, dest: Path,
                    quality: int, max_duration: int) -> list[str]:
    """yt-dlp invocation tuned for short curated clips."""
    fmt = f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]/mp4"
    return [
        yt_dlp,
        "--quiet", "--no-warnings",
        "--match-filter", f"duration <= {max_duration}",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", str(dest),
        url,
    ]


def _is_videoid(s: str) -> bool:
    return len(s) >= 8 and "/" not in s and " " not in s


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="Parquet manifest with a 'videoid' column.")
    p.add_argument("--out",      type=Path, required=True,
                   help="Directory to download clips into.")
    p.add_argument("--limit",    type=int, default=None,
                   help="Cap how many clips to attempt to download.")
    p.add_argument("--quality",  type=int, default=480,
                   help="Max video height to keep (default: 480p).")
    p.add_argument("--max-duration", type=int, default=120,
                   help="Skip clips longer than this many seconds.")
    p.add_argument("--filter",   default=None,
                   help="Optional SQL-ish filter on the manifest, e.g. "
                        "\"caption like '%melting%'\".  Naïve string match, "
                        "case-insensitive.")
    args = p.parse_args(argv)

    # Resolve yt-dlp executable — try PATH first, fall back to sibling of
    # the running interpreter (handles uv-managed venvs that don't activate).
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        candidate = Path(sys.executable).parent / "yt-dlp"
        if candidate.exists():
            yt_dlp = str(candidate)
    if yt_dlp is None:
        raise SystemExit("yt-dlp not found — `uv pip install yt-dlp` first")

    args.out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.manifest)
    cols = {c.lower(): c for c in table.column_names}
    id_col   = cols.get("videoid") or cols.get("id") or cols.get("clip_id")
    cap_col  = cols.get("caption") or cols.get("name") or cols.get("text")
    if id_col is None:
        raise SystemExit(f"Manifest must contain (videoid|id|clip_id) — "
                         f"got {table.column_names}")

    ids = table.column(id_col).to_pylist()
    caps = table.column(cap_col).to_pylist() if cap_col else [None] * len(ids)

    if args.filter:
        needle = args.filter.lower()
        pairs = [(i, c) for i, c in zip(ids, caps) if c and needle in c.lower()]
        print(f"Filter '{args.filter}' kept {len(pairs)} / {len(ids)} rows")
    else:
        pairs = list(zip(ids, caps))

    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Will attempt {len(pairs):,} downloads → {args.out}")

    n_ok = n_skipped = n_err = 0
    t0 = time.perf_counter()
    for vid, cap in tqdm(pairs):
        if not _is_videoid(str(vid)):
            n_err += 1
            continue
        target = args.out / f"{vid}.mp4"
        if target.exists() and target.stat().st_size > 1024:
            n_skipped += 1
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            subprocess.run(
                _yt_dlp_command(yt_dlp, url, target, args.quality, args.max_duration),
                check=True, capture_output=True, timeout=120,
            )
            if target.exists() and target.stat().st_size > 1024:
                n_ok += 1
            else:
                n_err += 1
        except subprocess.CalledProcessError:
            n_err += 1
        except subprocess.TimeoutExpired:
            n_err += 1
            if target.exists():
                target.unlink(missing_ok=True)

    dt = time.perf_counter() - t0
    print(f"\n  ok={n_ok}  skipped={n_skipped}  err={n_err}  "
          f"wall={dt:.1f}s  ({n_ok / max(dt, 1e-6):.2f} clips/s)")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
