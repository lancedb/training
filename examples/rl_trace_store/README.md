# RL rollout & trace store — Lance vs. the ad-hoc data plane

Agentic RL has a data plane problem. Every post-training stack moves **rollouts**
(multi-turn traces with tool outputs, token ids, logprobs, rewards) between three
parties — *generators* (inference + sandboxed envs), *verifiers* (rubrics, judge
models, test runners), and the *trainer* — and today that plane is built from
host RAM, one-off pickle/JSON files, and object-store round trips. Traces from
sandboxed agents run 1–32 MB **each**; buffering them in memory stops scaling
exactly when RL runs start to work.

This repo reproduces that pipeline shape and measures Lance as the store in
the middle: **persisted like a lake, random-accessed like memory.**

**[`REPORT.md`](./REPORT.md) is the full research report** — the verified
32 MB sandbox-artifact flow in Prime Intellect's `verifiers` (with the open
issue asking for exactly this component), a 12-library survey of the rollout
data plane, capability mapping, and ranked integration proposals.

## What's here

| File | What it does |
|---|---|
| `tracegen.py` | Generates realistic agentic rollouts (multi-turn chat JSON, int32 token ids, float32 logprobs, 64 KB–32 MB execution-log blobs; ~5% are 24–32 MB "monsters") |
| `bench.py` / `bench_one.py` | Benchmarks 5 backends (host RAM, pickle-per-rollout, JSON-per-rollout, parquet-per-step, one Lance dataset) across the 3 access patterns every RL pipeline has |
| `demo_pipeline.py` | Live multi-process demo: N producers append concurrently, a verifier random-accesses each new trace, a trainer reads only token columns — all through one Lance dataset |
| `artifact_sink_poc.py` | The artifact sink [verifiers#2189](https://github.com/PrimeIntellect-ai/verifiers/issues/2189) asks for, in ~60 lines on a Lance blob table |
| `run_all.sh` | Reproduces everything |

## The three access patterns

- **P1 — verifier fetch**: get ONE rollout's full trace (messages + execution log) to judge it. This is the "sandbox → host → verifier sandbox" hop.
- **P2 — reward sweep**: read `(step, reward, verdict)` for EVERY rollout — orchestrator bookkeeping, curation queries, dashboards.
- **P3 — trainer batch**: read `(completion_ids, logprobs)` for a random batch — the trainer never needs the blobs.

## Results

256 rollouts / **1.16 GB** of traces (16 steps × 16 rollouts), local NVMe, 4 vCPU,
cold = page cache dropped. Full raw numbers in [`results/`](./results/).

| backend | on disk | write MB/s | host RAM held | P1 fetch one trace, cold p50 | P2 reward sweep (all rows), cold/warm | P3 trainer batch ×64, cold |
|---|---|---|---|---|---|---|
| host memory (status quo) | — | — | **1,161 MB** | 0 ms | ~0 ms | 0.1 ms |
| pickle / `.pt` per rollout | 1,157 MB | 140 | ~0 | 2.5 ms | 1,017 / 357 ms | 801 ms |
| JSON per rollout | 1,564 MB | 49 | ~0 | 18.7 ms | 7,414 / 4,373 ms | 3,742 ms |
| parquet per step | 631 MB | 98 | ~0 | 156.0 ms | **19 / 7 ms** | 74 ms |
| **Lance, one dataset (Blob v2)** | 1,139 MB | 83 | ~0 | **9.7 ms** | 262 / **6 ms** | **34 ms** |

What the table says:

- **Host memory is the only backend that "wins" everywhere — by holding the
  entire payload in RAM.** 1.16 GB for 256 rollouts; at 2,048 concurrent 32 MB
  rollouts that's **64 GB** of buffer RAM. That is the scaling wall.
- **Pickle-per-rollout** (what `.pt`-file relays are) point-reads fine but has no
  columns: sweeping rewards or feeding the trainer deserializes every blob byte
  (1.0 s / 0.8 s for 4-byte and 100 KB payloads respectively).
- **Parquet** is the columnar mirror image: great sweeps, but fetching ONE trace
  decodes whole row groups — 156 ms and ~450 MB of reads to return 4 MB.
- **Lance is the only backend fast on all three patterns at once** — 9.7 ms point
  reads (16× parquet) *and* 6 ms warm columnar sweeps (60× pickle) *and* 34 ms
  training batches (24× pickle) — while holding **zero** rollouts in host RAM.
  Scans never touch blob payloads (Blob v2 stores them out-of-line), and each
  trace is additionally readable as a lazy file handle (`take_blobs`).
- Encoding trade-off, measured: with legacy inline blob encoding (format 2.1)
  writes hit 183 MB/s and cold sweeps 54 ms (vs 83 MB/s / 262 ms on Blob v2,
  which buys lazy file-like reads, blob-skipping scans, and no write
  amplification on updates). Both beat every baseline on P1+P3; pick per workload.
- Lance stores blob payloads uncompressed by design (zero-copy reads); parquet's
  disk edge comes from compressing log text. App-level zstd on the blob field
  closes that gap if disk matters more than CPU.

## Live pipeline demo

`demo_pipeline.py` runs the real topology — concurrent producers, a verifier,
and a trainer as separate processes sharing ONE Lance dataset (144 rollouts,
581 MB of traces):

```
=== pipeline complete in 8.0s ===
rollouts committed : 144   dataset versions : 19   curated for SFT : 66 (zero copies)
producer-0: append_p50_ms 299.1   (3 concurrent writers, optimistic commits, 0 failures)
verifier  : fetched all 144 traces (565 MB) at 5.4 ms p50 / 60.1 ms p95 per rollout, peak RSS 468 MB
trainer   : 410k tokens via (completion_ids, logprobs) columns only, 3.1 ms per 32-rollout batch, peak RSS 256 MB
```

The verifier never holds more than one chunk in memory; the trainer never reads
a blob byte; every append is a recoverable snapshot (`dataset.checkout_version` /
tags), so a crashed stage resumes instead of losing rollouts; and the same
dataset answers the curation query that exports high-reward traces for SFT.

## Run it

```bash
pip install -r requirements.txt
./run_all.sh
```
