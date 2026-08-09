"""Benchmark a single backend in an isolated process (clean peak-RSS + cache)."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time

import numpy as np
import psutil

from store_backends import BACKENDS, Ref
from tracegen import TraceGen


def drop_caches() -> bool:
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except (PermissionError, OSError):
        return False


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--per-step", type=int, default=16)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1-samples", type=int, default=32)
    ap.add_argument("--p3-batch", type=int, default=64)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    n_rows = args.steps * args.per_step
    rng = np.random.default_rng(7)
    res: dict = {"backend": args.backend, "steps": args.steps, "per_step": args.per_step}

    gen = TraceGen(seed=args.seed)
    backend = BACKENDS[args.backend](os.path.join(args.root, args.backend), args.per_step)

    # ---- write path --------------------------------------------------------
    rss_before_write = rss_mb()
    step_times, payload_bytes = [], 0
    t_gen = 0.0
    for step in range(args.steps):
        t0 = time.perf_counter()
        rollouts = gen.make_step(step, args.per_step)
        t_gen += time.perf_counter() - t0
        payload_bytes += sum(r.nbytes() for r in rollouts)
        t0 = time.perf_counter()
        backend.write_step(step, rollouts)
        step_times.append(time.perf_counter() - t0)
        if backend.persistent:
            del rollouts  # a persistent store does not keep rollouts in RAM
    backend.finalize()
    res["gen_s"] = round(t_gen, 3)
    res["payload_mb"] = round(payload_bytes / 1e6, 1)
    res["write_total_s"] = round(sum(step_times), 3)
    res["write_step_p50_ms"] = round(float(np.percentile(step_times, 50)) * 1e3, 1)
    res["write_mb_s"] = round(payload_bytes / 1e6 / max(sum(step_times), 1e-9), 1)
    res["disk_mb"] = round(backend.bytes_on_disk() / 1e6, 1)
    res["rss_after_write_mb"] = round(rss_mb() - rss_before_write, 1)

    refs_all = [Ref(s, i, s * args.per_step + i) for s in range(args.steps) for i in range(args.per_step)]

    def sample_refs(k: int) -> list[Ref]:
        k = min(k, n_rows)
        return [refs_all[j] for j in rng.choice(n_rows, size=k, replace=False)]

    # ---- P1: verifier fetch (one full trace at a time) ---------------------
    for mode in ("cold", "warm"):
        if mode == "cold":
            res["cold_supported"] = drop_caches()
            if not res["cold_supported"]:
                continue
        lat, touched = [], 0
        for ref in sample_refs(args.p1_samples):
            t0 = time.perf_counter()
            touched += backend.read_full([ref])
            lat.append(time.perf_counter() - t0)
        res[f"p1_{mode}_p50_ms"] = round(float(np.percentile(lat, 50)) * 1e3, 2)
        res[f"p1_{mode}_p95_ms"] = round(float(np.percentile(lat, 95)) * 1e3, 2)
        res[f"p1_{mode}_mb_s"] = round(touched / 1e6 / sum(lat), 1)

    # ---- P2: reward sweep over everything ----------------------------------
    for mode in ("cold", "warm"):
        if mode == "cold" and not drop_caches():
            continue
        t0 = time.perf_counter()
        mean_reward = backend.read_scalars_all(n_rows)
        res[f"p2_{mode}_s"] = round(time.perf_counter() - t0, 3)
    res["mean_reward"] = round(mean_reward, 4)

    # ---- P3: trainer batch (tokens + logprobs) ------------------------------
    for mode in ("cold", "warm"):
        if mode == "cold" and not drop_caches():
            continue
        batch = sample_refs(args.p3_batch)
        t0 = time.perf_counter()
        tokens = backend.read_training_batch(batch)
        res[f"p3_{mode}_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    res["p3_tokens"] = tokens

    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)

    if not args.keep:
        backend.cleanup()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
