"""Producer -> Verifier -> Trainer over ONE Lance dataset.

This replaces the "sandbox -> host RAM -> sandbox" relay with a persisted
rollout store that every stage random-accesses independently:

  producers (N procs)   append rollout batches as sandboxes finish them
                        (concurrent writers, optimistic commits)
  verifier  (1 proc)    discovers new rows by polling the dataset version,
                        fetches ONLY the columns it needs (messages + lazy
                        blob) by row index, appends verdicts to a second
                        dataset -- never holds more than one chunk in RAM
  trainer   (1 proc)    reads ONLY (completion_ids, logprobs) for verified
                        rows -- the blob bytes never touch the trainer

Nothing is lost when a stage restarts; every append is a recoverable version.

Run:  python demo_pipeline.py --producers 3 --batches 4 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import shutil
import time

import numpy as np
import pyarrow as pa

import lance
from store_backends import LANCE_SCHEMA, rollouts_to_table
from tracegen import TraceGen

VERDICT_SCHEMA = pa.schema(
    [
        pa.field("rollout_id", pa.string()),
        pa.field("row", pa.int64()),
        pa.field("verified", pa.bool_()),
        pa.field("verify_score", pa.float32()),
    ]
)


def append_with_retry(table: pa.Table, uri: str, retries: int = 8) -> None:
    """Lance appends use optimistic concurrency; retry on rare commit races."""
    for attempt in range(retries):
        try:
            lance.write_dataset(table, uri, mode="append")
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.05 * (2**attempt) * (0.5 + np.random.random()))


def report(root: str, name: str, payload: dict) -> None:
    payload["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)
    with open(os.path.join(root, "reports", f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=2)


def producer(root: str, wid: int, batches: int, batch_size: int) -> None:
    uri = os.path.join(root, "rollouts.lance")
    gen = TraceGen(seed=1000 + wid)
    lat, produced_mb = [], 0.0
    for b in range(batches):
        rollouts = [gen.make(step=b, idx=wid * 1000 + i) for i in range(batch_size)]
        produced_mb += sum(r.nbytes() for r in rollouts) / 1e6
        table = rollouts_to_table(rollouts, LANCE_SCHEMA)
        t0 = time.perf_counter()
        append_with_retry(table, uri)
        lat.append(time.perf_counter() - t0)
    report(root, f"producer-{wid}", {
        "produced_mb": round(produced_mb, 1),
        "append_p50_ms": round(float(np.percentile(lat, 50)) * 1e3, 1),
        "append_p95_ms": round(float(np.percentile(lat, 95)) * 1e3, 1),
    })


def verifier(root: str, expected_total: int, chunk: int = 8) -> None:
    """Pulls each new rollout's trace by row index and judges it."""
    uri = os.path.join(root, "rollouts.lance")
    vuri = os.path.join(root, "verdicts.lance")
    seen, fetch_lat, verified_mb = 0, [], 0.0
    while True:
        ds = lance.dataset(uri)  # picks up the latest committed version
        total = ds.count_rows()
        while seen < total:
            rows = list(range(seen, min(seen + chunk, total)))
            t0 = time.perf_counter()
            meta = ds.take(rows, columns=["rollout_id", "reward"])
            blobs = ds.take_blobs("trace_blob", indices=rows)
            verdicts = []
            for i, blob in enumerate(blobs):
                with blob as f:
                    data = f.read()  # only THIS rollout's bytes, streamed from disk
                verified_mb += len(data) / 1e6
                retries = data.count(b"RETRY")
                score = 1.0 / (1.0 + retries / 1000.0)
                verdicts.append(
                    {
                        "rollout_id": meta.column("rollout_id")[i].as_py(),
                        "row": rows[i],
                        "verified": bool(score > 0.5 and meta.column("reward")[i].as_py() > 0.3),
                        "verify_score": score,
                    }
                )
            fetch_lat.append((time.perf_counter() - t0) / len(rows))
            append_with_retry(pa.Table.from_pylist(verdicts, VERDICT_SCHEMA), vuri)
            seen += len(rows)
        if seen >= expected_total:
            break
        time.sleep(0.1)
    report(root, "verifier", {
        "verified_rollouts": seen,
        "verified_mb": round(verified_mb, 1),
        "fetch_per_rollout_p50_ms": round(float(np.percentile(fetch_lat, 50)) * 1e3, 1),
        "fetch_per_rollout_p95_ms": round(float(np.percentile(fetch_lat, 95)) * 1e3, 1),
    })


def trainer(root: str, expected_total: int, batch: int = 32) -> None:
    """Consumes ONLY token/logprob columns of verified rollouts."""
    uri = os.path.join(root, "rollouts.lance")
    vuri = os.path.join(root, "verdicts.lance")
    done_rows, tokens, fetch_lat, steps = set(), 0, [], 0
    while True:
        try:
            vds = lance.dataset(vuri)
        except ValueError:
            time.sleep(0.1)
            continue
        vt = vds.to_table(filter="verified", columns=["row"])
        todo = [r for r in vt.column("row").to_pylist() if r not in done_rows]
        for i in range(0, len(todo), batch):
            rows = todo[i : i + batch]
            t0 = time.perf_counter()
            t = lance.dataset(uri).take(rows, columns=["completion_ids", "logprobs"])
            fetch_lat.append(time.perf_counter() - t0)
            tokens += sum(len(v) for v in t.column("completion_ids"))
            time.sleep(0.05)  # simulated optimizer step
            steps += 1
            done_rows.update(rows)
        total_seen = vds.count_rows()
        if total_seen >= expected_total and not todo:
            break
        time.sleep(0.1)
    report(root, "trainer", {
        "train_steps": steps,
        "trained_rollouts": len(done_rows),
        "tokens_consumed": tokens,
        "batch_fetch_p50_ms": round(float(np.percentile(fetch_lat, 50)) * 1e3, 1) if fetch_lat else None,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--producers", type=int, default=3)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--root", default="./demo_data")
    args = ap.parse_args()

    shutil.rmtree(args.root, ignore_errors=True)
    os.makedirs(os.path.join(args.root, "reports"), exist_ok=True)
    uri = os.path.join(args.root, "rollouts.lance")
    lance.write_dataset(
        LANCE_SCHEMA.empty_table(), uri, schema=LANCE_SCHEMA, data_storage_version="2.2"
    )
    lance.write_dataset(
        VERDICT_SCHEMA.empty_table(), os.path.join(args.root, "verdicts.lance"), schema=VERDICT_SCHEMA
    )

    expected = args.producers * args.batches * args.batch_size
    t0 = time.perf_counter()
    procs = [
        mp.Process(target=producer, args=(args.root, w, args.batches, args.batch_size))
        for w in range(args.producers)
    ]
    procs.append(mp.Process(target=verifier, args=(args.root, expected)))
    procs.append(mp.Process(target=trainer, args=(args.root, expected)))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    ds = lance.dataset(uri)
    ds.tags.create("run-complete", ds.version)
    curated = ds.to_table(filter="verdict AND reward > 0.55", columns=["rollout_id"])

    print(f"\n=== pipeline complete in {wall:.1f}s ===")
    print(f"rollouts committed : {ds.count_rows()} (expected {expected})")
    print(f"dataset versions   : {ds.version} (every append is a recoverable snapshot)")
    print(f"curated for SFT    : {curated.num_rows} rollouts via filter, zero copies")
    for fn in sorted(os.listdir(os.path.join(args.root, "reports"))):
        with open(os.path.join(args.root, "reports", fn)) as f:
            print(f"{fn[:-5]:>12}: {json.load(f)}")


if __name__ == "__main__":
    # fork would inherit the parent's initialized async runtime (deadlock);
    # every stage gets a fresh interpreter, as separate hosts would.
    mp.set_start_method("spawn", force=True)
    main()
