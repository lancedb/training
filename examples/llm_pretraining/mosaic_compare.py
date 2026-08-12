"""LanceDB StreamingDataset vs MosaicML Streaming, on identical samples.

Both loaders serve the byte-identical pre-packed 1024-token blocks written by
build_packed_datasets.py (MDS shards + Lance table). Sample identity is a
crc32 fingerprint of the token block.

Subcommands
-----------
rank-worker   (internal) one simulated rank: emit fingerprints per step
elastic       global batches at ws=1/2/4 must match, per loader
resume        mid-epoch checkpoint at ws=A, resume at ws=B, per loader
throughput    single-process sample/s + tok/s for N seconds, per loader

Usage
-----
python mosaic_compare.py elastic --mds ~/blogrun/mds_blocks --lance ~/blogrun/blocks_db
python mosaic_compare.py resume --mds ... --lance ... --from-ws 4 --to-ws 2
python mosaic_compare.py throughput --loader mosaic --mds s3://... --seconds 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib

GLOBAL_BATCH = 32
SEED = 9
NCN = 8  # mosaic num_canonical_nodes, pinned for elastic determinism
LANCE_SPLITS = 32  # lance num_splits, pinned likewise


def fp(arr) -> int:
    import numpy as np

    return zlib.crc32(np.asarray(arr, dtype=np.int32).tobytes())


def make_lance(lance_path: str, rank: int, world: int):
    import lancedb
    from lancedb.streaming import StreamingDataset

    tbl = lancedb.connect(lance_path).open_table("blocks")
    return StreamingDataset(
        tbl,
        columns=["input_ids"],
        num_splits=LANCE_SPLITS,
        shuffle_seed=SEED,
        rank=rank,
        world_size=world,
        read_batch_size=8,
    )


def make_mosaic(mds_path: str, batch_size: int, cache: str):
    from streaming import StreamingDataset as MosaicSD

    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        shuffle_seed=SEED,
        num_canonical_nodes=NCN,
    )
    if mds_path.startswith("s3://"):
        return MosaicSD(remote=mds_path, local=cache, **kwargs)
    return MosaicSD(local=mds_path, **kwargs)


def cmd_rank_worker(args) -> None:
    """Emit `steps` global-step fingerprint groups for this rank as JSON."""
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    per_rank = GLOBAL_BATCH // world
    out = []
    if args.loader == "lance":
        ds = make_lance(args.lance, rank, world)
        it = iter(ds)
        skip_state = None
        if args.state:
            import torch

            ds.load_state_dict(torch.load(args.state, weights_only=False))
            it = iter(ds)
        for _ in range(args.steps):
            out.append([fp(next(it)["input_ids"]) for _ in range(per_rank)])
        if args.save_state:
            import torch

            torch.save(ds.state_dict(), args.save_state)
    else:
        from streaming import StreamingDataLoader

        ds = make_mosaic(args.mds, per_rank, args.cache)
        dl = StreamingDataLoader(ds, batch_size=per_rank, num_workers=0)
        if args.state:
            import torch

            dl.load_state_dict(torch.load(args.state, weights_only=False))
        it = iter(dl)
        for _ in range(args.steps):
            batch = next(it)["input_ids"]
            out.append([fp(row) for row in batch])
        if args.save_state:
            import torch

            torch.save(dl.state_dict(), args.save_state)
    json.dump(out, open(args.out, "w"))


def _run_ranks(args, loader: str, world: int, steps: int, tmp: str,
               state_in: str = "", state_out: str = ""):
    """Launch one subprocess per simulated rank; return per-step global sets."""
    procs, outs = [], []
    for r in range(world):
        out = f"{tmp}/{loader}_ws{world}_r{r}.json"
        outs.append(out)
        env = dict(
            os.environ,
            RANK=str(r),
            WORLD_SIZE=str(world),
            LOCAL_RANK=str(r),
            LOCAL_WORLD_SIZE=str(world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29799",
        )
        cmd = [
            sys.executable, __file__, "rank-worker",
            "--loader", loader, "--steps", str(steps), "--out", out,
            "--mds", args.mds, "--lance", args.lance,
            "--cache", f"{tmp}/cache_{loader}_ws{world}_r{r}",
        ]
        if state_in:
            cmd += ["--state", state_in]
        if state_out and r == 0:
            cmd += ["--save-state", state_out]
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        assert p.wait() == 0, f"{loader} rank worker failed"
    steps_sets = []
    per_rank = [json.load(open(o)) for o in outs]
    for s in range(steps):
        group = []
        for r in range(world):
            group.extend(per_rank[r][s])
        steps_sets.append(sorted(group))
    return steps_sets


def cmd_elastic(args) -> None:
    tmp = tempfile.mkdtemp(prefix="elastic_")
    for loader in ["lance", "mosaic"]:
        ref = _run_ranks(args, loader, 1, args.steps, tmp)
        results = {}
        for world in [2, 4]:
            got = _run_ranks(args, loader, world, args.steps, tmp)
            results[world] = got == ref
        status = " ".join(f"ws1==ws{w}: {ok}" for w, ok in results.items())
        print(f"{loader:>7}: elastic determinism over {args.steps} steps -> {status}")


def cmd_resume(args) -> None:
    tmp = tempfile.mkdtemp(prefix="resume_")
    total = args.consume + args.steps
    for loader in ["lance", "mosaic"]:
        ref = _run_ranks(args, loader, args.from_ws, total, tmp)
        state = f"{tmp}/{loader}_state.pt"
        _run_ranks(args, loader, args.from_ws, args.consume, tmp, state_out=state)
        rest = _run_ranks(args, loader, args.to_ws, args.steps, tmp, state_in=state)
        ok = rest == ref[args.consume :]
        print(
            f"{loader:>7}: resume ws={args.from_ws} -> ws={args.to_ws} after "
            f"{args.consume} steps, next {args.steps} global batches match: {ok}"
        )


def cmd_throughput(args) -> None:
    tmp = tempfile.mkdtemp(prefix="tput_")
    if args.loader == "lance":
        src = args.lance
        ds = make_lance(src, 0, 1)
        if args.workers:
            from torch.utils.data import DataLoader

            it = iter(DataLoader(ds, batch_size=GLOBAL_BATCH, num_workers=args.workers))
        else:
            it = iter(ds)
    else:
        src = args.mds
        ds = make_mosaic(src, GLOBAL_BATCH, f"{tmp}/cache")
        from streaming import StreamingDataLoader

        it = iter(StreamingDataLoader(ds, batch_size=GLOBAL_BATCH, num_workers=args.workers))
    next(it)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        item = next(it)
        n += len(item["input_ids"]) if (args.loader == "mosaic" or args.workers) else 1
    dt = time.perf_counter() - t0
    print(
        f"{args.loader:>7} ({src.split('://')[0] if '://' in src else 'local'}, "
        f"workers={args.workers}): {n / dt:,.0f} samples/s = "
        f"{n * 1024 / dt:,.0f} tok/s over {dt:.0f}s"
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["rank-worker", "elastic", "resume", "throughput"])
    p.add_argument("--mds", default="")
    p.add_argument("--lance", default="")
    p.add_argument("--loader", default="lance", choices=["lance", "mosaic"])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--consume", type=int, default=25)
    p.add_argument("--from-ws", type=int, default=4)
    p.add_argument("--to-ws", type=int, default=2)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--cache", default="")
    p.add_argument("--state", default="")
    p.add_argument("--save-state", default="")
    args = p.parse_args(argv)
    {
        "rank-worker": cmd_rank_worker,
        "elastic": cmd_elastic,
        "resume": cmd_resume,
        "throughput": cmd_throughput,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
