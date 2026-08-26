"""Pretrain a GPT on a LanceDB table with the elastic StreamingDataset.

The table is the dataset: no webdataset shards, no tokenized parquet copies,
no manifest files.  Training reads only the `input_ids` column, prefiltered
by SQL, streamed in a deterministic order — optionally sequence-packed by
the loader itself (``pack_sequences``), so every trained position is a real
token.

Usage
-----
# Single process (debug / CPU smoke):
python train.py --model tiny --steps 40

# The blog run — GPT-2 124M, Chinchilla-ish 2.5B tokens, 4x H100:
torchrun --nproc-per-node 4 train.py --model small --tokenizer hf:gpt2 \
    --pack --compile --batch-size 32 --grad-accum 4 --seq-len 1024 \
    --epochs 1 --ckpt-every 1000

# Kill it mid-run, then resume (same topology when --pack):
torchrun --nproc-per-node 4 train.py ... --resume auto
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import math
import os
import sys
import time

import pyarrow as pa
import torch
import torch.distributed as dist
from lancedb.streaming import StreamingDataLoader, StreamingDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from common import (
    DEFAULT_DB,
    DEFAULT_TABLE,
    TRAIN_FILTER,
    VAL_FILTER,
    connect_table,
    load_tokenizer,
)
from model import make_model

H100_BF16_FLOPS = 989e12  # dense peak, SXM


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument(
        "--tokenizer", default="byte", help="'byte' or 'hf:<model>' (vocab/pad/eos ids)"
    )
    p.add_argument(
        "--model", default="tiny", choices=["tiny", "small", "medium", "large", "xl"]
    )
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="docs (or packed blocks, with --pack) per rank per micro-step",
    )
    p.add_argument(
        "--pack",
        action="store_true",
        help="sequence packing in the loader (pack_sequences): EOS-joined "
        "fixed-length blocks, no padding waste. Resume packed runs at the "
        "same world_size",
    )
    p.add_argument(
        "--transform-queue-depth",
        type=int,
        default=0,
        help="cap on post-transform rows buffered per split, in read batches "
        "(loader default: unbounded). Long runs otherwise accumulate hundreds "
        "of thousands of cooked rows per worker — GBs of Python ints that "
        "trigger periodic multi-second GC pauses. 16 is plenty of headroom",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes per rank via lancedb's "
        "StreamingDataLoader: moves the (GIL-bound) packer off the training "
        "thread; checkpoints are committed only for batches the trainer has "
        "received, so state_dict() stays exact. num_splits must be divisible "
        "by world_size * num_workers",
    )
    p.add_argument(
        "--mp-context",
        default="forkserver",
        choices=["forkserver", "spawn"],
        help="multiprocessing start method for --num-workers (fork is unsafe "
        "with CUDA + lancedb's async runtime; forkserver is faster than spawn)",
    )
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--compile-mode", default="default", help="torch.compile mode, e.g. max-autotune")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after N optimizer steps (0 = full epochs)",
    )
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument(
        "--lr-total-steps",
        type=int,
        default=0,
        help="cosine horizon; 0 = the computed total step count. Pin this "
        "when a run will be interrupted so schedules line up on resume",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="quality-score cutoff (SQL prefilter)",
    )
    p.add_argument(
        "--num-splits",
        type=int,
        default=0,
        help="elastic splits; 0 = one per global-batch slot",
    )
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--read-batch-size", type=int, default=8)
    p.add_argument(
        "--io-queue-depth",
        type=int,
        default=1,
        help="I/O batches in flight per split (loader default 4). Threads = "
        "splits x depth per rank; oversubscription costs more than latency "
        "hiding gains on local NVMe",
    )
    p.add_argument(
        "--transform-parallelism",
        type=int,
        default=2,
        help="transform threads per rank (loader default = os.cpu_count()). "
        "Packing is GIL-bound Python; 8 ranks x 112 threads starves the "
        "pipeline",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--eval-every", type=int, default=0, help="eval every N steps (0 = only at end)"
    )
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--resume", default="", help="checkpoint path, or 'auto' for latest")
    p.add_argument(
        "--blocks-mode",
        default="off",
        choices=["off", "lance", "mosaic", "parquet-random", "parquet-seq"],
        help="A/B harness: train on identical pre-packed 1024-token blocks via "
        "the lance loader, MosaicML streaming, Parquet random-take, or "
        "pre-shuffled Parquet shards read sequentially (build_packed_datasets.py)",
    )
    p.add_argument("--blocks-path", default="", help="blocks db dir / MDS dir or s3://")
    p.add_argument(
        "--blocks-per-epoch",
        default="",
        help="packed block budget; default: exact value computed from the "
        "n_tokens column (production path). 'auto' uses the loader's sampler",
    )
    return p.parse_args(argv)


def setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def make_transform(seq_len: int, pad_id: int):
    """Row mode: RecordBatch -> per-row dicts, pad/truncate to seq_len."""

    def transform(batch: pa.RecordBatch) -> list[dict]:
        rows = []
        for ids in batch.column("input_ids").to_pylist():
            ids = ids[:seq_len]
            n = len(ids)
            input_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
            input_ids[:n] = torch.tensor(ids, dtype=torch.long)
            loss_mask = torch.zeros(seq_len, dtype=torch.bool)
            loss_mask[:n] = True
            rows.append({"input_ids": input_ids, "loss_mask": loss_mask})
        return rows

    return transform


def make_dataset(
    args,
    rank: int,
    world_size: int,
    *,
    split_filter: str,
    epoch: int,
    shuffle: bool,
    tok,
    num_splits: int,
    packed: bool = False,
    blocks_per_epoch=None,
) -> StreamingDataset:
    tbl = connect_table(args.db, args.table)
    filt = f"NOT is_dup AND score >= {args.min_score} AND ({split_filter})"
    kwargs = dict(
        columns=["input_ids"],
        filter=filt,
        num_splits=num_splits,
        shuffle=shuffle,
        shuffle_seed=args.shuffle_seed,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        read_batch_size=args.read_batch_size,
        io_queue_depth=args.io_queue_depth,
        transform_parallelism=args.transform_parallelism,
        transform_queue_depth=args.transform_queue_depth or None,
    )
    if packed:
        # Packing lives in the loader now: EOS-joined blocks, pad only when a
        # split runs dry.  Pads are masked from the loss in the train step.
        kwargs.update(
            pack_sequences=args.seq_len,
            eos_id=tok.eos_token_id,
            pad_id=tok.pad_token_id,
            blocks_per_epoch=blocks_per_epoch,
        )
    else:
        kwargs.update(transform=make_transform(args.seq_len, tok.pad_token_id))
    return StreamingDataset(tbl, **kwargs)


def make_loader(ds: StreamingDataset, args):
    """Training loader: in-process (num_workers=0) or lancedb's consumer-
    committed StreamingDataLoader with worker processes."""
    if args.num_workers > 0:
        # Not fork: ranks have CUDA + lancedb's async runtime initialised, and
        # forked workers can deadlock (lancedb warns about exactly this).
        return StreamingDataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            multiprocessing_context=args.mp_context,
        )
    return DataLoader(ds, batch_size=args.batch_size, num_workers=0)


def batch_loss(trainable, batch, device, pad_id):
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    if "loss_mask" in batch:
        mask = batch["loss_mask"].to(device, non_blocking=True)
    else:  # packed blocks carry doc_ids; pads are identified by id
        mask = input_ids != pad_id
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return trainable(input_ids, mask), mask.sum()


def evaluate(trainable, args, rank, world_size, device, tok, num_splits) -> float:
    ds = make_dataset(
        args,
        rank,
        world_size,
        split_filter=VAL_FILTER,
        epoch=0,
        shuffle=False,
        tok=tok,
        num_splits=num_splits,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    trainable.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.eval_batches:
                break
            loss, _ = batch_loss(trainable, batch, device, tok.pad_token_id)
            losses.append(loss)
    trainable.train()
    val = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    if world_size > 1:
        dist.all_reduce(val, op=dist.ReduceOp.AVG)
    return val.item()


def lr_at(step: int, total: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    t = (step - args.warmup_steps) / max(total - args.warmup_steps, 1)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def flops_per_token(model) -> float:
    """nanoGPT-style estimate: 6N + 12 * n_layer * d_model * seq_len."""
    cfg = model.cfg
    return 6 * model.num_params() + 12 * cfg.n_layer * cfg.d_model * cfg.seq_len


def blocks_to_rows(batch: pa.RecordBatch, seq_len: int) -> list[dict]:
    """Row transform for the pre-packed Lance blocks table (module-level so it
    pickles into DataLoader worker processes)."""
    import numpy as np

    mat = batch.column("input_ids").values.to_numpy().reshape(-1, seq_len)
    return [{"input_ids": torch.from_numpy(r.astype(np.int64))} for r in mat]


def run_blocks_ab(args, rank, world_size, device, trainable, model, opt, tok) -> None:
    """Identical-samples A/B: same trainer, loader swapped (see mosaic_compare)."""
    import contextlib as _ctx
    import functools

    global_batch = args.batch_size * world_size
    fpt = flops_per_token(model.module if hasattr(model, "module") else model)
    if args.blocks_mode == "lance":
        import lancedb

        btbl = lancedb.connect(args.blocks_path).open_table("blocks")
        to_rows = functools.partial(blocks_to_rows, seq_len=args.seq_len)
        ds = StreamingDataset(
            btbl,
            columns=["input_ids"],
            num_splits=args.num_splits or global_batch,
            shuffle_seed=args.shuffle_seed,
            rank=rank,
            world_size=world_size,
            read_batch_size=args.read_batch_size,
            io_queue_depth=args.io_queue_depth,
            transform_parallelism=args.transform_parallelism,
            transform=to_rows,
        )
        loader = make_loader(ds, args)
    elif args.blocks_mode.startswith("parquet"):
        from blocks_loaders import ParquetRandomBlocks, ParquetSeqBlocks

        if args.blocks_mode == "parquet-random":
            pds = ParquetRandomBlocks(args.blocks_path, rank, world_size, args.shuffle_seed)
            workers = args.num_workers or 8  # concurrent readers hide per-row-group latency
        else:
            pds = ParquetSeqBlocks(args.blocks_path, rank, world_size)
            workers = args.num_workers or 2
        loader = DataLoader(pds, batch_size=args.batch_size, num_workers=workers, prefetch_factor=4)
        ds = None
    else:
        from streaming import StreamingDataset as MosaicSD
        from streaming import StreamingDataLoader

        kwargs = dict(
            batch_size=args.batch_size,
            shuffle=True,
            shuffle_seed=args.shuffle_seed,
            num_canonical_nodes=8,
        )
        if args.blocks_path.startswith("s3://"):
            # One shared local cache per node (Mosaic's contract: rank 0 downloads
            # index.json / shards, the other local ranks wait for the same files).
            import hashlib

            cache = os.path.expanduser(
                f"~/mosaic_cache/{hashlib.md5(args.blocks_path.encode()).hexdigest()[:8]}"
            )
            mds = MosaicSD(remote=args.blocks_path, local=cache, **kwargs)
        else:
            mds = MosaicSD(local=args.blocks_path, **kwargs)
        loader = StreamingDataLoader(mds, batch_size=args.batch_size, num_workers=8)
        ds = None

    trainable.train()
    opt_step, tokens = 0, 0
    t_last, step_last = time.perf_counter(), 0
    micro = 0
    for batch in loader:
        is_sync = (micro + 1) % args.grad_accum == 0
        ctx = (
            trainable.no_sync()
            if (world_size > 1 and not is_sync)
            else _ctx.nullcontext()
        )
        with ctx:
            loss, real = batch_loss(trainable, batch, device, tok.pad_token_id)
            (loss / args.grad_accum).backward()
        micro += 1
        if not is_sync:
            continue
        for g in opt.param_groups:
            g["lr"] = args.lr
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        opt_step += 1
        if rank == 0 and opt_step % args.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_last
            tps = (opt_step - step_last) * global_batch * args.grad_accum * args.seq_len / dt
            mfu = fpt * tps / (world_size * H100_BF16_FLOPS)
            q = ""
            if ds is not None:
                q = f" | q {ds.unscanned_rows}/{ds.raw_queue_depth}/{ds.prefetch_queue_depth}/{ds.consumed_rows}"
            print(
                f"[{args.blocks_mode}] step {opt_step}/{args.steps} | "
                f"loss {loss.item():.4f} | {tps:,.0f} tok/s | mfu {mfu:.1%}{q}"
            )
            t_last, step_last = time.perf_counter(), opt_step
        if args.steps and opt_step >= args.steps:
            break
    if rank == 0:
        print(f"[{args.blocks_mode}] final: opt_step={opt_step}")
    if world_size > 1:
        dist.destroy_process_group()
    sys.stdout.flush()
    os._exit(0)  # see main(): worker teardown can hang at exit


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.num_workers:
        # Before any StreamingDataset exists: its shared-memory stats are
        # created in the *default* mp context, so make that the same
        # (non-fork) context the DataLoader workers will use.
        import multiprocessing as mp

        mp.set_start_method(args.mp_context, force=True)
    rank, world_size, device = setup_distributed()
    is_main = rank == 0
    torch.set_float32_matmul_precision("high")

    tok = load_tokenizer(args.tokenizer)
    global_batch = args.batch_size * world_size
    num_splits = args.num_splits or global_batch
    if args.num_workers and num_splits % (world_size * args.num_workers):
        raise SystemExit(
            f"num_splits ({num_splits}) must be divisible by world_size x "
            f"num_workers ({world_size} x {args.num_workers})"
        )
    if global_batch % num_splits != 0:
        raise SystemExit(
            f"global batch ({global_batch} = {args.batch_size} x {world_size} ranks) "
            f"must be a multiple of num_splits ({num_splits}) so optimizer steps "
            f"align with loader cycles."
        )

    torch.manual_seed(1234)
    model = make_model(args.model, tok.vocab_size, args.seq_len).to(device)
    fpt = flops_per_token(model)
    if is_main:
        print(
            f"model={args.model} params={model.num_params():,} pack={args.pack} "
            f"world_size={world_size} global_batch={global_batch} "
            f"num_splits={num_splits} compile={args.compile} "
            f"num_workers={args.num_workers}"
        )
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    trainable = model
    if world_size > 1:
        trainable = DDP(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    try:
        opt = torch.optim.AdamW(
            trainable.parameters(), lr=args.lr, weight_decay=0.1,
            betas=(0.9, 0.95), fused=device.type == "cuda",
        )
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(
            trainable.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
        )

    # ── Resume (per-rank loader state: packed state covers owned splits) ──
    start_epoch, opt_step, loader_state = 0, 0, None
    if args.resume:
        path = args.resume
        if path == "auto":
            os.makedirs(args.ckpt_dir, exist_ok=True)
            mains = sorted(
                p
                for p in glob.glob(os.path.join(args.ckpt_dir, "step_*.pt"))
                if "_dsrank" not in p
            )
            if not mains:
                raise SystemExit(f"--resume auto: no checkpoints in {args.ckpt_dir}")
            path = mains[-1]
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        opt_step = ckpt["opt_step"]
        state_files = [path] + sorted(glob.glob(path.replace(".pt", "_dsrank*.pt")))
        states = [
            torch.load(f, map_location="cpu", weights_only=False)["dataset"]
            for f in state_files
        ]
        if args.pack and len(states) > 1:
            # Packed state is per logical split; merging every rank's state
            # yields a topology-independent checkpoint (resume on any
            # compatible world size).
            loader_state = StreamingDataset.merge_state_dicts(states)
        else:
            rank_path = path.replace(".pt", f"_dsrank{rank}.pt") if rank else path
            loader_state = torch.load(
                rank_path, map_location="cpu", weights_only=False
            )["dataset"]
        if is_main:
            print(f"resumed from {path} (epoch {start_epoch}, opt step {opt_step})")

    if args.blocks_mode != "off":
        run_blocks_ab(args, rank, world_size, device, trainable, model, opt, tok)
        return

    tbl = connect_table(args.db, args.table)
    base_filter = f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})"
    train_rows = tbl.count_rows(base_filter)
    if args.pack:
        toks = (
            tbl.search()
            .select(["n_tokens"])
            .where(base_filter)
            .limit(train_rows)
            .to_arrow()
        )
        import pyarrow.compute as pc

        total_tokens = pc.sum(toks.column("n_tokens")).as_py() + train_rows
        if args.blocks_per_epoch == "auto":
            blocks_per_epoch = "auto"
        elif args.blocks_per_epoch:
            blocks_per_epoch = int(args.blocks_per_epoch)
        else:  # exact budget from the materialized token counts
            b = total_tokens // args.seq_len
            blocks_per_epoch = b - b % num_splits
        if is_main:
            print(f"blocks_per_epoch: {blocks_per_epoch}")
        micro_steps_per_epoch = total_tokens // args.seq_len // global_batch
    else:
        blocks_per_epoch = None
        micro_steps_per_epoch = train_rows // num_splits
    steps_per_epoch = max(micro_steps_per_epoch // args.grad_accum, 1)
    total_steps = args.lr_total_steps or args.steps or steps_per_epoch * args.epochs
    if is_main:
        print(
            f"train rows (post-filter): {train_rows:,}  ~steps/epoch: "
            f"{steps_per_epoch:,}  target steps: {total_steps:,}"
        )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    tokens_seen = torch.zeros((), dtype=torch.int64, device=device)
    t_last, step_last = time.perf_counter(), opt_step

    def save_ckpt(epoch: int, ds: StreamingDataset) -> None:
        # Every rank persists its own loader state: packed checkpoints cover
        # only the splits an iterator owns.  Rank 0 also saves model/optim.
        rank_payload = {"dataset": ds.state_dict(), "epoch": epoch, "opt_step": opt_step}
        base = os.path.join(args.ckpt_dir, f"step_{opt_step:08d}")
        if is_main:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    **rank_payload,
                },
                base + ".pt",
            )
            print(f"  saved {base}.pt")
        else:
            torch.save(rank_payload, f"{base}_dsrank{rank}.pt")

    done = False
    for epoch in range(start_epoch, args.epochs):
        ds = make_dataset(
            args,
            rank,
            world_size,
            split_filter=TRAIN_FILTER,
            epoch=epoch,
            shuffle=True,
            tok=tok,
            num_splits=num_splits,
            packed=args.pack,
            blocks_per_epoch=blocks_per_epoch,
        )
        if loader_state is not None:
            ds.load_state_dict(loader_state)
            loader_state = None
        loader = make_loader(ds, args)

        micro = 0
        for batch in loader:
            is_sync = (micro + 1) % args.grad_accum == 0
            ctx = (
                trainable.no_sync()
                if (world_size > 1 and not is_sync)
                else contextlib.nullcontext()
            )
            with ctx:
                loss, real_tokens = batch_loss(
                    trainable, batch, device, tok.pad_token_id
                )
                (loss / args.grad_accum).backward()
            micro += 1
            tokens_seen += real_tokens  # GPU accumulator: no per-step sync
            if not is_sync:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(opt_step, total_steps, args)
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1

            if is_main and opt_step % args.log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t_last
                tps = (
                    (opt_step - step_last)
                    * global_batch
                    * args.grad_accum
                    * args.seq_len
                    / dt
                )
                mfu = fpt * tps / (world_size * H100_BF16_FLOPS)
                q = f"{ds.unscanned_rows}/{ds.raw_queue_depth}/{ds.prefetch_queue_depth}/{ds.consumed_rows}"
                print(
                    f"epoch {epoch} step {opt_step}/{total_steps} | "
                    f"loss {loss.item():.4f} | {tps:,.0f} tok/s | mfu {mfu:.1%} | "
                    f"q {q} | fetch {ds.fetch_time:.1f}s tx {ds.transform_time:.1f}s"
                )
                t_last, step_last = time.perf_counter(), opt_step

            if opt_step % args.ckpt_every == 0:
                save_ckpt(epoch, ds)
            if args.eval_every and opt_step % args.eval_every == 0:
                val = evaluate(
                    trainable, args, rank, world_size, device, tok, num_splits
                )
                if is_main:
                    print(f"  val loss @ step {opt_step}: {val:.4f}")
            if args.steps and opt_step >= args.steps:
                done = True
                break
        if done:
            if opt_step % args.ckpt_every != 0:  # not already saved this step
                save_ckpt(epoch, ds)
            break

    val = evaluate(trainable, args, rank, world_size, device, tok, num_splits)
    if is_main:
        print(
            f"final: opt_step={opt_step} tokens_seen(rank0)={int(tokens_seen):,} "
            f"val_loss={val:.4f}"
        )
    if world_size > 1:
        dist.destroy_process_group()
    if args.num_workers:
        # Worker-process teardown can hang at interpreter exit (mixed Rust
        # runtime + torch + forkserver finalizers); everything is flushed and
        # checkpointed by now, so leave without running finalizers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
