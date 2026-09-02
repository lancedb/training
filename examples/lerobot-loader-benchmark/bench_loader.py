#!/usr/bin/env python
"""LeRobot dataloader throughput benchmark. One process, one backend, one number.

Emulates a VLA training step's read pattern (every camera + state at t=0, plus an action
chunk) and reports steady-state samples/s, time-to-first-batch, bytes/sample over the wire,
and peak memory across the whole DataLoader worker tree.

    python bench_loader.py --backend s3 --repo-id lerobot/droid_1.0.1 \
        --root s3://my-bucket/droid_1.0.1-lance

Backends: local | s3 | bucket | hub | stream. The first three are map-style at --root; `hub`
is map-style through the Hub cache (downloads); `stream` is StreamingLeRobotDataset.

READ PITFALLS.md BEFORE TRUSTING A NUMBER. Memory, not speed, is the interesting axis for
the streaming path, and shuffle scope is not comparable across styles by default.
"""
import argparse, glob, json, os, resource, sys, threading, time, traceback
from pathlib import Path

CHUNK = 50    # policy action-chunk length (smolvla default)


def rx_bytes():
    tot = 0
    for pth in glob.glob("/sys/class/net/*/statistics/rx_bytes"):
        if "/lo/" in pth:
            continue
        try: tot += int(Path(pth).read_text())
        except Exception: pass
    return tot


def delta_ts(meta, chunk=CHUNK):
    """Read pattern of a real VLA training step: every camera at t=0, state at t=0, and an
    action chunk. Derived from the dataset's own metadata so this works on any LeRobot set."""
    dt = {k: [0.0] for k in meta.camera_keys}
    for name, feat in meta.features.items():
        if name.startswith("observation.state") and tuple(feat.get("shape") or ()) not in ((), (1,)):
            dt[name] = [0.0]
    if "action" in meta.features:
        dt["action"] = [i / meta.fps for i in range(chunk)]
    return dt


class RSSTracker(threading.Thread):
    """Poll total RSS of this process + all descendants (the DataLoader workers)."""
    daemon = True
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval, self.peak_gb, self.stop_flag = interval, 0.0, False
        self.base_avail_gb = self.min_avail_gb = 0.0
    def _tree_pss_kb(self):
        """Sum PSS (proportional set size) over self + children.

        PSS divides each shared page by the number of processes mapping it, so a
        worker tree that shares libraries and torch shm segments is not counted
        many times over -- which plain RSS-summing does.
        """
        # Read /proc directly. This used to shell out to `ps --ppid`; under memory pressure
        # that subprocess fails, the except-branch falls back to self-only, and the tracker
        # silently reports a SMALLER number for a BIGGER workload. That is exactly how the
        # reservoir sweep produced 0.61 GB at buffer=64,000 vs 39.85 GB at buffer=1,000.
        # Also walk descendants transitively -- DataLoader workers may not be direct children.
        pid = os.getpid()
        try:
            kids = {}
            for e in os.listdir("/proc"):
                if not e.isdigit():
                    continue
                try:
                    with open(f"/proc/{e}/stat") as fh:
                        parts = fh.read().rsplit(") ", 1)[1].split()
                    kids.setdefault(int(parts[1]), []).append(int(e))
                except Exception:
                    pass
            pids, stack = [pid], [pid]
            while stack:
                for c in kids.get(stack.pop(), []):
                    if c not in pids:
                        pids.append(c); stack.append(c)
        except Exception:
            pids = [pid]
        total = 0
        for p_ in pids:
            try:
                for line in open(f"/proc/{p_}/smaps_rollup"):
                    if line.startswith("Pss:"):
                        total += int(line.split()[1]); break
            except Exception:
                pass
        return total
    @staticmethod
    def _mem_available_gb():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
        return 0.0

    def run(self):
        self.base_avail_gb = self._mem_available_gb()
        self.min_avail_gb = self.base_avail_gb
        while not self.stop_flag:
            gb = self._tree_pss_kb() / 1024 / 1024
            self.peak_gb = max(self.peak_gb, gb)
            self.min_avail_gb = min(self.min_avail_gb, self._mem_available_gb())
            time.sleep(self.interval)


def build(args):
    """Return (dataset, style). `style` decides whether shuffle=True is even legal."""
    from lerobot.datasets.storage import load_dataset_metadata
    tol = dict(tolerance_s=args.tolerance_s) if args.tolerance_s else {}

    if args.backend == "stream":
        from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
        meta = load_dataset_metadata(args.repo_id)
        dt = delta_ts(meta, args.chunk)
        return StreamingLeRobotDataset(args.repo_id, delta_timestamps=dt, return_uint8=True,
                                       max_num_shards=args.num_workers,
                                       buffer_size=args.stream_buffer_size, **tol), "iterable"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if args.backend == "hub":            # map-style from the Hub cache (downloads)
        meta = load_dataset_metadata(args.repo_id)
        dt = delta_ts(meta, args.chunk)
        return LeRobotDataset(args.repo_id, delta_timestamps=dt, return_uint8=True, **tol), "map"

    if not args.root:
        raise SystemExit(f"--root is required for backend={args.backend}")
    kw = dict(repo_type="bucket") if args.backend == "bucket" else {}
    meta = load_dataset_metadata(args.repo_id, root=args.root, **kw)
    dt = delta_ts(meta, args.chunk)
    return LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=dt,
                          return_uint8=True, **kw, **tol), "map"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=["local", "s3", "bucket", "hub", "stream"],
                   help="local/s3 = map-style at --root; hub = map-style via Hub cache "
                        "(downloads); stream = StreamingLeRobotDataset (iterable)")
    p.add_argument("--repo-id", required=True, help="e.g. lerobot/droid_1.0.1")
    p.add_argument("--root", default="", help="dataset root: a path, s3:// URI, or bucket id")
    p.add_argument("--chunk", type=int, default=CHUNK, help="action-chunk length")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--mp-context", default=None,
                   choices=[None, "fork", "forkserver", "spawn"],
                   help="DataLoader worker start method; lancedb warns that fork "
                        "can deadlock its async runtime.")
    p.add_argument("--tolerance-s", type=float, default=None,
                   help="frame-timestamp tolerance; droid needs >1e-4 at long timestamps")
    p.add_argument("--decoder-cache", type=int, default=None,
                   help="override LanceDatasetReader video_decoder_cache_size")
    p.add_argument("--no-persistent", action="store_true")
    p.add_argument("--stream-buffer-size", type=int, default=1000,
                   help="StreamingLeRobotDataset reservoir size. Default 1000 = a shuffle "
                        "window covering 0.004%% of DROID's 27.6M frames. Sweeping this "
                        "trades throughput and RAM for shuffle quality, which is the whole "
                        "point: 802 samples/s is not a speed, it is this dial set low.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Force shuffle=False on the map-style side. The harness default is "
                        "shuffle=(style=='map'), because PyTorch REFUSES shuffle=True on an "
                        "IterableDataset -- so lance gets a global shuffle over 27.6M frames "
                        "and streaming gets its internal 1000-frame reservoir window. That "
                        "makes samples/s incomparable. This flag matches the access patterns "
                        "so the raw IO paths can be compared directly.")
    p.add_argument("--label", default="")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    import torch
    from torch.utils.data import DataLoader
    import lancedb

    # The Lance reader caches only 16 video decoders per worker by default. On a dataset with
    # many video rows under a global shuffle, nearly every sample re-pays ffmpeg's ~256 KB
    # open-probe read. Raising this showed +41.8% on DROID -- worth sweeping.
    if args.decoder_cache:
        from lerobot.datasets import lance_backend as _lb
        _orig = _lb.DATASET_READER.__init__
        def _patched(self, *a, **kw):
            kw.setdefault("video_decoder_cache_size", args.decoder_cache)
            return _orig(self, *a, **kw)
        _lb.DATASET_READER.__init__ = _patched
        print(f"[decoder cache set to {args.decoder_cache}]", flush=True)

    rss = RSSTracker(); rss.start()
    res = dict(backend=args.backend, repo_id=args.repo_id, root=args.root, label=args.label, lancedb=lancedb.__version__,
               mp_context=args.mp_context, tolerance_s=args.tolerance_s,
               decoder_cache=args.decoder_cache,
               batch_size=args.batch_size, num_workers=args.num_workers,
               num_batches=args.num_batches)
    try:
        t = time.perf_counter()
        ds, style = build(args)
        res["open_s"] = round(time.perf_counter() - t, 1)
        res["style"] = style
        res["shuffle"] = bool(style == "map" and not args.no_shuffle)
        res["stream_buffer_size"] = args.stream_buffer_size if style == "iterable" else None
        res["frames"] = None if style == "iterable" else len(ds)
        print(f"[{args.backend}] style={style} open={res['open_s']}s frames={res['frames']}", flush=True)

        mp_kw = {}
        if args.mp_context and args.num_workers > 0:
            mp_kw["multiprocessing_context"] = args.mp_context
        dl = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=(style == "map" and not args.no_shuffle),
                        num_workers=args.num_workers, **mp_kw,
                        persistent_workers=(args.num_workers > 0 and not args.no_persistent),
                        prefetch_factor=2 if args.num_workers else None,
                        drop_last=True, pin_memory=True)
        t = time.perf_counter(); it = iter(dl); next(it)
        res["ttfb_s"] = round(time.perf_counter() - t, 1)
        print(f"  ttfb={res['ttfb_s']}s", flush=True)
        for _ in range(args.warmup - 1): next(it)
        t = time.perf_counter(); b0 = rx_bytes()
        for _ in range(args.num_batches): next(it)
        el = time.perf_counter() - t
        mb = (rx_bytes() - b0) / 1e6
        res["net_MB"] = round(mb, 1)
        res["MB_per_sample"] = round(mb / (args.num_batches * args.batch_size), 3)
        res["seconds"] = round(el, 2)
        res["samples_per_s"] = round(args.num_batches * args.batch_size / el, 1)
        res["status"] = "ok"
        print(f"  {res['samples_per_s']} samples/s   "
              f"{res.get('MB_per_sample','?')} MB/sample over the wire", flush=True)
    except Exception as e:
        res["status"] = "error"
        res["error_type"] = type(e).__name__
        res["error"] = str(e)[:2000]
        print(f"  FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
        traceback.print_exc()
    finally:
        rss.stop_flag = True
        res["peak_pss_gb_tree"] = round(rss.peak_gb, 2)
        # System-wide floor: how much MemAvailable the run consumed at its worst. Cannot be
        # broken by per-process accounting failures, so it is the number to trust for memory.
        res["sys_mem_consumed_gb"] = round(max(0.0, rss.base_avail_gb - rss.min_avail_gb), 2)
        res["peak_rss_gb_self_maxrss"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)

    print(json.dumps(res, indent=2), flush=True)
    o = Path(args.out); rows = json.loads(o.read_text()) if o.exists() else []
    rows.append(res); o.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
