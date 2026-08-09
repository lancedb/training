# RL rollout & trace store — Lance vs. the ad-hoc data plane

Agentic RL has a data plane problem. Every post-training stack moves **rollouts**
(multi-turn traces with tool outputs, token ids, logprobs, rewards) between three
parties — *generators* (inference + sandboxed envs), *verifiers* (rubrics, judge
models, test runners), and the *trainer* — and today that plane is built from
host RAM, one-off pickle/JSON files, and object-store round trips. Traces from
sandboxed agents run 1–32 MB **each**; buffering them in memory stops scaling
exactly when RL runs start to work.

This example reproduces that pipeline shape and measures Lance as the store in
the middle: **persisted like a lake, random-accessed like memory.**

## What's here

| File | What it does |
|---|---|
| `tracegen.py` | Generates realistic agentic rollouts (multi-turn chat JSON, int32 token ids, float32 logprobs, 64 KB–32 MB execution-log blobs; ~5% are 24–32 MB "monsters") |
| `bench.py` / `bench_one.py` | Benchmarks 5 backends (host RAM, pickle-per-rollout, JSON-per-rollout, parquet-per-step, one Lance dataset) across the 3 access patterns every RL pipeline has |
| `demo_pipeline.py` | Live multi-process demo: N producers append concurrently, a verifier random-accesses each new trace, a trainer reads only token columns — all through one Lance dataset |
| `run_all.sh` | Reproduces everything |

## The three access patterns

- **P1 — verifier fetch**: get ONE rollout's full trace (messages + execution log) to judge it. This is the "sandbox → host → verifier sandbox" hop.
- **P2 — reward sweep**: read `(step, reward, verdict)` for EVERY rollout — orchestrator bookkeeping, curation queries, dashboards.
- **P3 — trainer batch**: read `(completion_ids, logprobs)` for a random batch — the trainer never needs the blobs.

## Results

256 rollouts / **1.16 GB** of traces (16 steps × 16 rollouts), local NVMe, 4 vCPU,
cold = page cache dropped. Full raw numbers in [`results/`](./results/).

| backend | on disk | write MB/s | host RAM held | P1 fetch one trace, cold p50 | P2 reward sweep (all rows), cold | P3 trainer batch ×64, cold |
|---|---|---|---|---|---|---|
| host memory (status quo) | — | — | **1,161 MB** | 0 ms | ~0 ms | 0.1 ms |
| pickle / `.pt` per rollout | 1,157 MB | 140 | ~0 | 2.5 ms | 1,017 ms | 801 ms |
| JSON per rollout | 1,564 MB | 49 | ~0 | 18.7 ms | 7,414 ms | 3,742 ms |
| parquet per step | 631 MB | 98 | ~0 | 156.0 ms | **19 ms** | 74 ms |
| **Lance, one dataset** | 1,140 MB | **183** | ~0 | **7.2 ms** | 54 ms | **41 ms** |

What the table says:

- **Host memory is the only backend that "wins" everywhere — by holding the
  entire payload in RAM.** 1.16 GB for 256 rollouts; at 2,048 concurrent 32 MB
  rollouts that's **64 GB** of buffer RAM. That is the scaling wall.
- **Pickle-per-rollout** (what `.pt`-file relays are) point-reads fine but has no
  columns: sweeping rewards or feeding the trainer deserializes every blob byte
  (1.0 s / 0.8 s for 4-byte and 100 KB payloads respectively).
- **Parquet** is the columnar mirror image: great sweeps, but fetching ONE trace
  decodes whole row groups — 156 ms and ~450 MB of reads to return 4 MB.
- **Lance is the only backend fast on all three patterns at once** — 7 ms point
  reads (21× parquet) *and* 54 ms columnar sweeps (19× pickle) *and* 41 ms
  training batches (20× pickle) — while holding **zero** rollouts in host RAM.
- Lance stores blob columns uncompressed by design (zero-copy reads); parquet's
  disk edge comes from compressing log text. App-level zstd on the blob field
  closes that gap if disk matters more than CPU.

## Live pipeline demo

`demo_pipeline.py` runs the real topology — concurrent producers, a verifier,
and a trainer as separate processes sharing ONE Lance dataset (144 rollouts,
581 MB of traces):

```
=== pipeline complete in 6.7s ===
rollouts committed : 144   dataset versions : 19   curated for SFT : 66 (zero copies)
producer-0: append_p50_ms 196.9   (3 concurrent writers, optimistic commits, 0 failures)
verifier  : fetched all 144 traces (565 MB) at 4.4 ms p50 / 34.2 ms p95 per rollout, peak RSS 432 MB
trainer   : 410k tokens via (completion_ids, logprobs) columns only, 8.1 ms per 32-rollout batch, peak RSS 257 MB
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
