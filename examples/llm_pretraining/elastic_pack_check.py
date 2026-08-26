"""Packed-stream elasticity + cross-topology resume, verified on the real corpus.

The packed loader emits one block per logical split per cycle, so a global
step (all ranks' blocks for one cycle) is the same set of blocks at any
world_size that divides ``num_splits``.  This harness fingerprints the blocks
and checks, on the actual training table:

1. elastic:  global steps at ws=A == global steps at ws=B
2. resume:   run ws=A for K steps, checkpoint every rank, merge with
             ``merge_state_dicts``, resume at ws=B — the next steps match the
             uninterrupted ws=A stream exactly.

Each simulated rank is its own process (like torchrun).

Usage
-----
python elastic_pack_check.py --db ~/runs/small/db --num-splits 256 --ws 8 4 --steps 30 --consume 12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zlib

from common import DEFAULT_TABLE, TRAIN_FILTER, connect_table, load_tokenizer


def make_ds(args, rank: int, world: int, blocks: int):
    from lancedb.streaming import StreamingDataset

    tok = load_tokenizer(args.tokenizer)
    return StreamingDataset(
        connect_table(args.db, args.table),
        columns=["input_ids"],
        filter=f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})",
        num_splits=args.num_splits,
        shuffle_seed=args.seed,
        rank=rank,
        world_size=world,
        read_batch_size=8,
        io_queue_depth=1,
        transform_parallelism=2,
        pack_sequences=args.seq_len,
        eos_id=tok.eos_token_id,
        pad_id=tok.pad_token_id,
        blocks_per_epoch=blocks,
    )


def cmd_worker(args) -> None:
    import torch

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    per_rank = args.num_splits // world
    ds = make_ds(args, rank, world, args.blocks)
    if args.state:
        ds.load_state_dict(torch.load(args.state, weights_only=False))
    it = iter(ds)
    out = []
    for _ in range(args.steps):
        out.append(
            [zlib.crc32(next(it)["input_ids"].numpy().tobytes()) for _ in range(per_rank)]
        )
    if args.save_state:
        torch.save(ds.state_dict(), args.save_state)
    json.dump(out, open(args.out, "w"))


def run_ranks(args, world, steps, tmp, tag, state_in="", save_states=False):
    procs, outs, states = [], [], []
    for r in range(world):
        out = f"{tmp}/{tag}_ws{world}_r{r}.json"
        outs.append(out)
        cmd = [
            sys.executable, __file__, "worker", "--db", args.db, "--table", args.table,
            "--tokenizer", args.tokenizer, "--num-splits", str(args.num_splits),
            "--seq-len", str(args.seq_len), "--seed", str(args.seed),
            "--blocks", str(args.blocks), "--steps", str(steps), "--out", out,
        ]
        if state_in:
            cmd += ["--state", state_in]
        if save_states:
            st = f"{tmp}/{tag}_ws{world}_r{r}_state.pt"
            states.append(st)
            cmd += ["--save-state", st]
        env = dict(os.environ, RANK=str(r), WORLD_SIZE=str(world))
        procs.append(subprocess.Popen(cmd, env=env))
    for p in procs:
        assert p.wait() == 0, "rank worker failed"
    per_rank = [json.load(open(o)) for o in outs]
    global_steps = [sorted(sum((per_rank[r][s] for r in range(world)), [])) for s in range(steps)]
    return global_steps, states


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", nargs="?", default="check", choices=["check", "worker"])
    p.add_argument("--db", required=True)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--tokenizer", default="hf:gpt2")
    p.add_argument("--num-splits", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-score", type=float, default=1.0)
    p.add_argument("--ws", type=int, nargs=2, default=[8, 4], help="from_ws to_ws")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--consume", type=int, default=12)
    p.add_argument("--blocks", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--state", default="")
    p.add_argument("--save-state", default="")
    args = p.parse_args(argv)
    if args.cmd == "worker":
        return cmd_worker(args)

    if not args.blocks:  # exact budget from n_tokens, as train.py does
        import pyarrow.compute as pc

        tbl = connect_table(args.db, args.table)
        filt = f"NOT is_dup AND score >= {args.min_score} AND ({TRAIN_FILTER})"
        n = tbl.count_rows(filt)
        tot = pc.sum(tbl.search().select(["n_tokens"]).where(filt).limit(n).to_arrow().column("n_tokens")).as_py() + n
        b = tot // args.seq_len
        args.blocks = b - b % args.num_splits
    a_ws, b_ws = args.ws
    tmp = tempfile.mkdtemp(prefix="elastic_pack_")

    from lancedb.streaming import StreamingDataset
    import torch

    ref, _ = run_ranks(args, a_ws, args.steps, tmp, "ref")
    alt, _ = run_ranks(args, b_ws, args.steps, tmp, "alt")
    print(f"elastic: {args.steps} global steps ws={a_ws} == ws={b_ws}: {ref == alt}")

    _, states = run_ranks(args, a_ws, args.consume, tmp, "part", save_states=True)
    merged = StreamingDataset.merge_state_dicts(
        [torch.load(s, weights_only=False) for s in states]
    )
    mpath = f"{tmp}/merged.pt"
    torch.save(merged, mpath)
    rest, _ = run_ranks(args, b_ws, args.steps - args.consume, tmp, "resume", state_in=mpath)
    ok = rest == ref[args.consume:]
    print(
        f"resume: ws={a_ws} for {args.consume} steps -> merge {a_ws} rank states -> "
        f"ws={b_ws}: next {args.steps - args.consume} global steps match: {ok}"
    )
    print(f"blocks_per_epoch={args.blocks:,} ({args.blocks // args.num_splits:,} per split)")


if __name__ == "__main__":
    main()
