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
    """Reservoir/shuffle-buffer semantics, as tf.data.shuffle and LeRobot streaming implement it.

    Walk the index space in order, hold `buffer` indices, and repeatedly emit a random one and
    refill its slot from the stream. buffer=1 is sequential; buffer>=len is a global shuffle.
    """

    def __init__(self, n, buffer, seed=0):
        self.n, self.buffer, self.seed = n, max(1, min(buffer, n)), seed

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        buf = list(range(self.buffer))
        nxt = self.buffer
        while buf:
            j = rng.integers(len(buf))
            yield int(buf[j])
            if nxt < self.n:
                buf[j] = nxt; nxt += 1
            else:
                buf.pop(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=int, required=True,
                    help="shuffle buffer size in frames; 0 = full global shuffle "
                         "(buffer = len(dataset)), so every arm uses the same sampler class "
                         "and ONLY the window width differs")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=os.environ.get("LANCE_ROOT",
                    "s3://lancedb-lerobot-blog-eu-north-1/droid_1.0.1-lance"))
    ap.add_argument("--seed", type=int, default=100)
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
            kw["sampler"] = ReservoirSampler(n, buf, seed=a.seed)
            state["patched"] += 1
            print(f"[shuffle_scope] replaced {was} -> reservoir "
                  f"buffer={buf:,} over {n:,} frames ({100*min(buf,n)/n:.4f}% of the dataset)",
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
                "--log_freq=100", f"--save_freq={a.steps}", "--eval_steps=0",
                "--tolerance_s=0.005", f"--seed={a.seed}", "--wandb.enable=false",
                f"--output_dir={a.out_dir}"]
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    print(f"[shuffle_scope] patched {state['patched']} dataloader(s)", flush=True)


if __name__ == "__main__":
    main()
