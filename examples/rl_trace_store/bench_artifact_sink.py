"""Benchmark the verifiers artifact path: state-retention vs DirectorySink vs Lance.

Runs the REAL `verifiers.v1.utils.artifacts.collect()` (real tar, real host
runtime) over N synthetic rollout workspaces, in three modes:

  state  -- today's behavior: archives retained on trace state (host RAM)
  dir    -- DirectorySink from the PR: tars + manifest.json per trace on disk
  lance  -- LanceArtifactSink: one Lance blob table, queryable manifest

Each mode runs in its own subprocess for clean peak-RSS. Same workspaces (seeded)
feed every mode. Run from the verifiers repo root with its venv:
    .venv/bin/python bench_artifact_sink.py --workspaces ./bench_ws --rollouts 48
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import resource
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)

# Cycled over rollouts. Capped at 28 MB: verifiers rejects a collection over its
# 32 MiB budget (tar overhead included), and the sink preserves that invariant —
# this benchmark measures the storage path, not the rejection path.
SIZES_MB = [1, 1, 2, 2, 2, 4, 4, 8, 8, 16, 24, 28]


def drop_caches() -> bool:
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except (PermissionError, OSError):
        return False


def make_workspaces(root: str, n: int) -> list[str]:
    rng = random.Random(7)
    line = (
        "[2026-08-24T12:00:00.123] worker-3 pid=1442 step={} loss=0.031337 "
        "path=/workspace/run/deadbeef/out.txt status=OK took=112ms\n"
    )
    paths = []
    for i in range(n):
        ws = os.path.join(root, f"rollout_{i:03d}")
        os.makedirs(ws, exist_ok=True)
        target = SIZES_MB[i % len(SIZES_MB)] * 1024 * 1024
        with open(os.path.join(ws, "agent.log"), "wb") as f:
            written = 0
            while written < target:
                chunk = (line.format(rng.randrange(10**6)) * 200).encode()
                chunk = chunk[: min(len(chunk), target - written)]
                # ~25% incompressible bytes, like diffs/base64 in real logs
                if rng.random() < 0.25:
                    chunk = os.urandom(len(chunk))
                f.write(chunk)
                written += len(chunk)
        with open(os.path.join(ws, "result.json"), "w") as f:
            json.dump({"rollout": i, "passed": i % 3 != 0}, f)
        paths.append(ws)
    return paths


async def run_mode(mode: str, workspaces: list[str], out: str) -> None:
    from verifiers.v1.runtimes import make_runtime
    from verifiers.v1.runtimes.subprocess import SubprocessConfig
    from verifiers.v1.utils.artifact_sink import DirectorySink
    from verifiers.v1.utils.artifacts import Artifact, collect

    store_root = os.path.join(os.path.dirname(out), f"store_{mode}")
    sink = None
    if mode == "dir":
        sink = DirectorySink(store_root)
    elif mode == "lance":
        from lance_artifact_sink import LanceArtifactSink

        sink = LanceArtifactSink(os.path.join(store_root, "artifacts.lance"))

    runtime = make_runtime(SubprocessConfig())
    await runtime.start()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3

    retained = []  # simulates traces alive for the epoch, as in a real eval
    t0 = time.perf_counter()
    for i, ws in enumerate(workspaces):
        collected = await collect(
            runtime, [Artifact(source=ws)], sink=sink, trace_id=f"t{i:03d}"
        )
        retained.append(collected)
    collect_s = time.perf_counter() - t0

    retained_mb = sum(
        len(v) for c in retained for v in c.values() if isinstance(v, bytes)
    ) / 1e6

    # restore-side: fetch 16 random archives back (what grading does)
    rng = random.Random(11)
    picks = rng.sample(range(len(workspaces)), min(16, len(workspaces)))
    cold = drop_caches()
    lat = []
    fetched_mb = 0.0
    for i in picks:
        key = workspaces[i]
        value = retained[i][key]
        t0 = time.perf_counter()
        if isinstance(value, bytes):
            data = value
        else:
            data = await sink.get(value)
        lat.append(time.perf_counter() - t0)
        fetched_mb += len(data) / 1e6
    lat.sort()

    # inspection: which artifacts exceed 8 MB? (post-run debugging/curation)
    t0 = time.perf_counter()
    if mode == "state":
        big = sum(
            1
            for c in retained
            for v in c.values()
            if isinstance(v, bytes) and len(v) > 8 * 1024 * 1024
        )
    elif mode == "dir":
        big = 0
        for trace_dir in os.listdir(store_root):
            manifest = json.load(
                open(os.path.join(store_root, trace_dir, "manifest.json"))
            )
            big += sum(
                1
                for e in manifest.values()
                if e["status"] == "collected" and e["size"] > 8 * 1024 * 1024
            )
    else:
        big = sink.manifest(
            filter=f"status = 'collected' AND size > {8 * 1024 * 1024}"
        ).num_rows
    inspect_ms = (time.perf_counter() - t0) * 1e3

    await runtime.teardown()
    result = {
        "mode": mode,
        "rollouts": len(workspaces),
        "collect_s": round(collect_s, 2),
        "retained_in_ram_mb": round(retained_mb, 1),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3 - rss0, 1
        ),
        "restore_cold": cold,
        "restore_p50_ms": round(lat[len(lat) // 2] * 1e3, 2),
        "restore_p95_ms": round(lat[int(len(lat) * 0.95) - 1] * 1e3, 2),
        "restore_fetched_mb": round(fetched_mb, 1),
        "inspect_over_8mb": big,
        "inspect_ms": round(inspect_ms, 2),
        "survives_process_exit": mode != "state",
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["state", "dir", "lance"])
    ap.add_argument("--workspaces", default="./bench_ws")
    ap.add_argument("--rollouts", type=int, default=48)
    ap.add_argument("--results", default="./bench_sink_results")
    args = ap.parse_args()

    args.workspaces = os.path.abspath(args.workspaces)
    args.results = os.path.abspath(args.results)
    os.makedirs(args.results, exist_ok=True)
    if args.mode:
        ws = sorted(
            os.path.join(args.workspaces, d) for d in os.listdir(args.workspaces)
        )
        out = os.path.join(args.results, f"{args.mode}.json")
        asyncio.run(run_mode(args.mode, ws, out))
        return

    if not os.path.isdir(args.workspaces):
        os.makedirs(args.workspaces)
        make_workspaces(args.workspaces, args.rollouts)
        print(f"generated {args.rollouts} workspaces", flush=True)
    rows = []
    for mode in ["state", "dir", "lance"]:
        print(f"=== {mode} ===", flush=True)
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--mode", mode,
             "--workspaces", args.workspaces, "--results", args.results],
            check=True,
        )
        rows.append(json.load(open(os.path.join(args.results, f"{mode}.json"))))

    cols = ["mode", "collect_s", "retained_in_ram_mb", "peak_rss_mb",
            "restore_p50_ms", "restore_p95_ms", "inspect_ms", "survives_process_exit"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
