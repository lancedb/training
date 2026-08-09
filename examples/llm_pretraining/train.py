"""Pretrain a GPT on a LanceDB table with the elastic StreamingDataset.

The table is the dataset: no webdataset shards, no tokenized parquet copies,
no manifest files.  Training reads only the `input_ids` column, prefiltered
by SQL, streamed in a deterministic elastic order.

Key properties demonstrated (see README for the full story):

- **Elastic determinism** — the samples that form each global step depend
  only on (num_splits, shuffle_seed, epoch), not on world size or worker
  count.  Scale from 8 to 64 GPUs and step N still trains on the same data.
- **Mid-epoch resume** — `state_dict`/`load_state_dict` checkpoint the
  loader; kill the job at any step and resume exactly, even on a different
  number of GPUs.
- **Prefiltered streaming** — `filter=` pushes curation predicates
  (dedup flag, quality score, holdout split) into the scan; rejected rows
  are never read from storage.

Usage
-----
# Single process (debug / CPU smoke):
python train.py --model tiny --steps 40

# One 8x GPU node:
torchrun --nproc-per-node 8 train.py --model medium --batch-size 8

# 4 or 8 H200 nodes (set MASTER_ADDR on all nodes):
torchrun --nnodes 4 --nproc-per-node 8 --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDR:29500 \
    train.py --model large --batch-size 4 --num-splits 256

# Kill it mid-run, then resume (works with a different node count too):
torchrun --nnodes 3 --nproc-per-node 8 ... train.py ... --resume auto
"""

from __future__ import annotations

import argparse
import math
import os
import time

import pyarrow as pa
import torch
import torch.distributed as dist
from lancedb.streaming import StreamingDataset
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument(
        "--tokenizer", default="byte", help="'byte' or 'hf:<model>' (vocab size only)"
    )
    p.add_argument(
        "--model", default="tiny", choices=["tiny", "small", "medium", "large"]
    )
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument(
        "--batch-size", type=int, default=8, help="docs per rank per micro-step"
    )
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--steps",
        type=int,
        default=0,
        help="stop after N optimizer steps (0 = full epochs)",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
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
    p.add_argument("--read-batch-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--eval-every", type=int, default=0, help="eval every N steps (0 = only at end)"
    )
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--resume", default="", help="checkpoint path, or 'auto' for latest")
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
    """RecordBatch -> per-row dicts of fixed-length tensors.

    Runs on StreamingDataset's transform thread pool, overlapped with I/O.
    Each document is truncated/padded to seq_len; the loss mask covers real
    tokens only.  (Fixed-length rows keep elastic determinism exact at the
    token level; see README for the packed-sequence variant and its
    trade-off.)
    """

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
    pad_id: int,
    num_splits: int,
) -> StreamingDataset:
    tbl = connect_table(args.db, args.table)
    filt = f"NOT is_dup AND score >= {args.min_score} AND ({split_filter})"
    return StreamingDataset(
        tbl,
        columns=["input_ids"],
        filter=filt,
        num_splits=num_splits,
        shuffle=shuffle,
        shuffle_seed=args.shuffle_seed,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        read_batch_size=args.read_batch_size,
        transform=make_transform(args.seq_len, pad_id),
    )


def evaluate(model, args, rank, world_size, device, pad_id, num_splits) -> float:
    ds = make_dataset(
        args,
        rank,
        world_size,
        split_filter=VAL_FILTER,
        epoch=0,
        shuffle=False,
        pad_id=pad_id,
        num_splits=num_splits,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.eval_batches:
                break
            loss = model(batch["input_ids"].to(device), batch["loss_mask"].to(device))
            losses.append(loss)
    model.train()
    val = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    if world_size > 1:
        dist.all_reduce(val, op=dist.ReduceOp.AVG)
    return val.item()


def lr_at(step: int, total: int, args) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    t = (step - args.warmup_steps) / max(total - args.warmup_steps, 1)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def main(argv=None) -> None:
    args = parse_args(argv)
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    tok = load_tokenizer(args.tokenizer)
    global_batch = args.batch_size * world_size
    # Elastic determinism holds when the global batch is a multiple of
    # num_splits.  Default: one split per global-batch slot.
    num_splits = args.num_splits or global_batch
    if global_batch % num_splits != 0:
        raise SystemExit(
            f"global batch ({global_batch} = {args.batch_size} x {world_size} ranks) "
            f"must be a multiple of num_splits ({num_splits}); otherwise global "
            f"steps are not reproducible across world sizes."
        )

    torch.manual_seed(1234)
    model = make_model(args.model, tok.vocab_size, args.seq_len).to(device)
    if is_main:
        print(
            f"model={args.model} params={model.num_params():,} "
            f"world_size={world_size} global_batch={global_batch} num_splits={num_splits}"
        )
    trainable = model
    if world_size > 1:
        trainable = DDP(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    opt = torch.optim.AdamW(trainable.parameters(), lr=args.lr, weight_decay=0.1)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch, opt_step, loader_state = 0, 0, None
    if args.resume:
        path = args.resume
        if path == "auto":
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpts = sorted(os.listdir(args.ckpt_dir))
            if not ckpts:
                raise SystemExit(f"--resume auto: no checkpoints in {args.ckpt_dir}")
            path = os.path.join(args.ckpt_dir, ckpts[-1])
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        # The loader state pins (num_splits, shuffle_seed) and the position
        # inside the epoch.  The epoch itself must be passed back to the
        # StreamingDataset constructor — it selects the permutation.
        start_epoch = ckpt["epoch"]
        opt_step = ckpt["opt_step"]
        loader_state = ckpt["dataset"]
        if is_main:
            print(f"resumed from {path} (epoch {start_epoch}, opt step {opt_step})")

    tbl = connect_table(args.db, args.table)
    train_rows = tbl.count_rows(
        f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})"
    )
    micro_steps_per_epoch = train_rows // num_splits
    steps_per_epoch = micro_steps_per_epoch // args.grad_accum
    total_steps = args.lr_total_steps or args.steps or steps_per_epoch * args.epochs
    if is_main:
        print(
            f"train rows (post-filter): {train_rows}  steps/epoch: {steps_per_epoch}  "
            f"target steps: {total_steps}"
        )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    tokens_seen = 0
    t_last, step_last = time.perf_counter(), opt_step

    def save_ckpt(epoch: int, ds: StreamingDataset) -> None:
        if not is_main:
            return
        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "dataset": ds.state_dict(),
            "epoch": epoch,
            "opt_step": opt_step,
        }
        path = os.path.join(args.ckpt_dir, f"step_{opt_step:08d}.pt")
        torch.save(payload, path)
        print(f"  saved {path}")

    done = False
    for epoch in range(start_epoch, args.epochs):
        # New epoch = new permutation: the dataset is reconstructed with the
        # epoch number (there is no set_epoch), which reshuffles rows into
        # the SAME splits so cross-epoch caching stays valid.
        ds = make_dataset(
            args,
            rank,
            world_size,
            split_filter=TRAIN_FILTER,
            epoch=epoch,
            shuffle=True,
            pad_id=tok.pad_token_id,
            num_splits=num_splits,
        )
        if loader_state is not None:
            ds.load_state_dict(loader_state)
            loader_state = None
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

        micro = 0
        for batch in loader:
            loss = trainable(
                batch["input_ids"].to(device), batch["loss_mask"].to(device)
            )
            (loss / args.grad_accum).backward()
            micro += 1
            tokens_seen += int(batch["loss_mask"].sum())
            if micro % args.grad_accum:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(opt_step, total_steps, args)
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1

            if is_main and opt_step % args.log_every == 0:
                dt = time.perf_counter() - t_last
                tps = (
                    (opt_step - step_last)
                    * global_batch
                    * args.grad_accum
                    * args.seq_len
                    / dt
                )
                # Queue depths tell you where the bottleneck is:
                # unscanned/raw/cooked/consumed (see the dataloading guide).
                q = f"{ds.unscanned_rows}/{ds.raw_queue_depth}/{ds.prefetch_queue_depth}/{ds.consumed_rows}"
                print(
                    f"epoch {epoch} step {opt_step}/{total_steps} | "
                    f"loss {loss.item():.4f} | {tps:,.0f} tok/s | q {q} | "
                    f"fetch {ds.fetch_time:.1f}s tx {ds.transform_time:.1f}s"
                )
                t_last, step_last = time.perf_counter(), opt_step

            if opt_step % args.ckpt_every == 0:
                save_ckpt(epoch, ds)
            if args.eval_every and opt_step % args.eval_every == 0:
                val = evaluate(
                    trainable,
                    args,
                    rank,
                    world_size,
                    device,
                    tok.pad_token_id,
                    num_splits,
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

    val = evaluate(
        trainable, args, rank, world_size, device, tok.pad_token_id, num_splits
    )
    if is_main:
        print(
            f"final: opt_step={opt_step} tokens_seen(rank0)={tokens_seen:,} val_loss={val:.4f}"
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
