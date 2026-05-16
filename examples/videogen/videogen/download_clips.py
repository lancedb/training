"""
Download ChronoMagic-Pro clips via yt-dlp into a directory ingest_chronomagic
can read with ``--video-dir``.

The dataset's ``videoid`` column is a YouTube ID, not a URL.  We turn each
id into a YouTube watch URL and download a tight clip into
``<out>/<videoid>.mp4``.  Errors per id (private, deleted, geo-blocked,
age-gated) are logged and the script moves on; the rest of the pipeline
treats missing files as "no video bytes for this row".

Parallelism: ``--parallel N`` spawns N concurrent yt-dlp subprocesses via
a thread pool.  Each subprocess is independent so the speedup scales
near-linearly with N until you hit network or YouTube rate-limits
(empirically 8-16 is a sweet spot on a single home IP).

Usage
-----
# 8-way parallel, ~80 attempts:
python -m videogen.download_clips \\
    --manifest data/chronomagic_proh.parquet \\
    --out      data/clips \\
    --filter melting --limit 80 --parallel 8

# Restart-safe: already-downloaded ids are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _attempt_one(yt_dlp: str, vid: str, target: Path,
                 quality: int, max_duration: int) -> str:
    """Download a single clip.  Returns 'ok' / 'skipped' / 'err'."""
    if target.exists() and target.stat().st_size > 1024:
        return "skipped"
    url = f"https://www.youtube.com/watch?v={vid}"
    try:
        subprocess.run(
            _yt_dlp_command(yt_dlp, url, target, quality, max_duration),
            check=True, capture_output=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if target.exists():
            target.unlink(missing_ok=True)
        return "err"
    if target.exists() and target.stat().st_size > 1024:
        return "ok"
    return "err"


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
    p.add_argument("--max-duration", type=int, default=60,
                   help="Skip clips longer than this many seconds.")
    p.add_argument("--filter",   default=None,
                   help="Optional caption-substring filter, case-insensitive. "
                        "Pass --filter '' to skip filtering.")
    p.add_argument("--filter-any", nargs="+", default=None,
                   help="Multiple substring filters; rows match if ANY matches.")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle manifest order before limiting.")
    p.add_argument("--seed",    type=int, default=0,
                   help="Shuffle seed.")
    p.add_argument("--parallel", type=int, default=1,
                   help="Concurrent yt-dlp workers (default: 1).")
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

    pairs = list(zip(ids, caps))
    if args.filter:
        needle = args.filter.lower()
        pairs = [(i, c) for i, c in pairs if c and needle in c.lower()]
        print(f"Filter '{args.filter}' kept {len(pairs)} / {len(ids)} rows")
    elif args.filter_any:
        needles = [s.lower() for s in args.filter_any]
        pairs = [(i, c) for i, c in pairs
                 if c and any(n in c.lower() for n in needles)]
        print(f"--filter-any {args.filter_any} kept {len(pairs)} / {len(ids)} rows")

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(pairs)

    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Will attempt {len(pairs):,} downloads → {args.out}  "
          f"(parallel={args.parallel})")

    counts = {"ok": 0, "skipped": 0, "err": 0}
    counts_lock = threading.Lock()
    t0 = time.perf_counter()

    def _job(vid):
        if not _is_videoid(str(vid)):
            return "err"
        return _attempt_one(yt_dlp, str(vid), args.out / f"{vid}.mp4",
                            args.quality, args.max_duration)

    progress = tqdm(total=len(pairs), unit="clip")
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futures = [ex.submit(_job, vid) for vid, _ in pairs]
        for f in as_completed(futures):
            status = f.result()
            with counts_lock:
                counts[status] += 1
            progress.update(1)
            progress.set_postfix(counts, refresh=False)
    progress.close()

    dt = time.perf_counter() - t0
    print(f"\n  ok={counts['ok']}  skipped={counts['skipped']}  "
          f"err={counts['err']}  wall={dt:.1f}s  "
          f"({counts['ok'] / max(dt, 1e-6):.2f} clips/s)")
    return 0 if counts["ok"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
