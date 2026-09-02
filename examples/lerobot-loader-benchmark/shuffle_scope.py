#!/usr/bin/env python3
r"""Does shuffle SCOPE change what the model learns, holding IO constant?

The throughput story says Lance shuffles 100% of the dataset while upstream's streaming
reservoir covers 0.054%. That is a true statement about coverage, but it only matters if a
narrow shuffle actually damages training -- which we had asserted and never measured.

This isolates the variable. Same Lance reader, same dataset, same seed, same steps; the ONLY
difference is how wide a window the sampler draws from. A buffer of size N is a global shuffle;
a buffer of 1,000 is what the streaming reader does.

    python shuffle_scope.py --buffer 1000 --steps 3000 --out-dir runs/scope_1k

Emulating rather than using StreamingLeRobotDataset is deliberate: streaming cannot train a VLA
on DROID at all (three separate blockers), and using it would confound shuffle scope with a
different IO path. Here IO is identical and only the index order changes.
"""
import argparse, os, runpy, sys
import numpy as np
import torch


class ReservoirSampler(torch.utils.data.Sampler):
    """Faithful emulation of StreamingLeRobotDataset's ordering.

    That reader applies TWO levels of randomness (streaming_dataset.py:430):
      1. the dataset is split into `num_shards` CONTIGUOUS blocks, each with its own cursor,
         and every step picks one shard at random;
      2. the frame it yields goes through a reservoir of `buffer` frames.

    Emulating only (2) with a single sequential cursor is materially more pessimistic than the
    real reader: with 8 shards over 27.6M frames the cursors start ~3.45M frames apart, so a
    short run touches eight windows spread across the dataset rather than only its beginning.
    Getting this wrong overstates the penalty of a narrow buffer.

    buffer >= n is a global shuffle (sharding then makes no difference).
    """

    def __init__(self, n, buffer, seed=0, num_shards=8):
        self.n, self.buffer, self.seed = n, max(1, min(buffer, n)), seed
        self.num_shards = max(1, min(num_shards, n))

    def __len__(self):
        return self.n

    def _shard_bounds(self):
        # datasets.IterableDataset.shard() gives contiguous blocks (verified empirically).
        step = self.n // self.num_shards
        b = [(k * step, (k + 1) * step if k < self.num_shards - 1 else self.n)
             for k in range(self.num_shards)]
        return [list(x) for x in b]          # [start_cursor, end)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        cursors = self._shard_bounds()
        live = list(range(self.num_shards))
        buf = []

        def next_frame():
            """One frame from a randomly chosen live shard."""
            while live:
                k = live[rng.integers(len(live))]
                cur, end = cursors[k]
                if cur < end:
                    cursors[k][0] = cur + 1
                    return cur
                live.remove(k)
            return None

        while len(buf) < self.buffer:
            f = next_frame()
            if f is None:
                break
            buf.append(f)
        while buf:
            j = rng.integers(len(buf))
            yield int(buf[j])
            f = next_frame()
            if f is None:
                buf.pop(j)
            else:
                buf[j] = f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=int, required=True,
                    help="shuffle buffer size in frames; 0 = full global shuffle "
                         "(buffer = len(dataset)), so every arm uses the same sampler class "
                         "and ONLY the window width differs")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save-freq", type=int, default=0,
                    help="checkpoint every N steps; 0 = only at the end")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=os.environ.get("LANCE_ROOT",
                    "s3://lancedb-lerobot-blog-eu-north-1/droid_1.0.1-lance"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--exclude", default="",
                    help="JSON file of episode indices to hold out of TRAINING. Without this, "
                         "every arm can train on the eval episodes -- and because a windowed "
                         "sampler walks a contiguous region, arms differ in HOW MUCH of the eval "
                         "set they saw, which confounds the comparison with shuffle width.")
    ap.add_argument("--num-shards", type=int, default=8,
                    help="concurrent shard cursors, as StreamingLeRobotDataset uses; DROID has 8")
    a = ap.parse_args()

    orig = torch.utils.data.DataLoader.__init__
    state = {"patched": 0}

    def patched(self, dataset, *args, **kw):
        # lerobot does NOT pass shuffle=True for map-style datasets. It sets shuffle=False and
        # installs an EpisodeAwareSampler(shuffle=True), which is already a global permutation.
        # So the knob to turn is the sampler, not the shuffle flag.
        if kw.get("sampler") is not None and not state["patched"]:
            n = len(dataset)
            buf = a.buffer or n
            was = type(kw["sampler"]).__name__
            kw["sampler"] = ReservoirSampler(n, buf, seed=a.seed, num_shards=a.num_shards)
            state["patched"] += 1
            print(f"[shuffle_scope] replaced {was} -> reservoir "
                  f"buffer={buf:,} over {n:,} frames, {a.num_shards} shard cursors "
                  f"({100*min(buf,n)/n:.4f}% resident)",
                  flush=True)
        return orig(self, dataset, *args, **kw)

    torch.utils.data.DataLoader.__init__ = patched

    sys.argv = ["lerobot-train",
                "--dataset.repo_id=lerobot/droid_1.0.1", f"--dataset.root={a.root}",
                "--policy.path=lerobot/smolvla_base", "--policy.push_to_hub=false",
                f"--rename_map={open('/home/ubuntu/work/bench/rename_map.json').read().strip()}",
                "--dataloader_multiprocessing_context=fork",
                "--accelerator.mixed_precision=bf16",
                f"--batch_size={a.batch_size}", "--num_workers=4", f"--steps={a.steps}",
                "--log_freq=100", f"--save_freq={a.save_freq or a.steps}", "--eval_steps=0",
                "--tolerance_s=0.005", f"--seed={a.seed}", "--wandb.enable=false",
                f"--output_dir={a.out_dir}"]
    if a.exclude:
        sys.argv.append(f"--dataset.exclude_episodes={open(a.exclude).read().strip()}")
        print(f"[shuffle_scope] holding out {len(__import__('json').load(open(a.exclude)))} "
              f"episodes from training", flush=True)
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    print(f"[shuffle_scope] patched {state['patched']} dataloader(s)", flush=True)


if __name__ == "__main__":
    main()
