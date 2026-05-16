# Video Generation Training with LanceDB

End-to-end fine-tune of a video-diffusion model (Wan2.2-TI2V-5B) on a curated
time-lapse phase-transition slice — using LanceDB + Geneva at every stage
from raw clips to high-MFU training.

See [PROPOSAL.md](PROPOSAL.md) for the full design + benchmark plan.

> Status: **CPU pipeline runnable end-to-end on synthetic data.**
> GPU UDFs (CLIP / RAFT / T5 / VAE / dHash) are scaffolded but not yet wired
> up — they land once an H100 frees up.  See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Setup

```bash
cd examples/videogen
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
```

> The dependency set in `pyproject.toml` is the **full training stack**
> (torch, diffusers, transformers, etc).  For CPU-only work you can drop
> them: `uv pip install lancedb pylance geneva pyarrow pandas imageio[ffmpeg]`.

## Quickstart — CPU only, ~30 seconds

Runs the full CPU pipeline on 500 synthetic clips.  No GPU.

```bash
# 1) Ingest
python -m videogen.ingest_chronomagic --synthetic 500 --overwrite

# 2) Tier 1 backfill (CPU keyword UDFs)
python -m videogen.backfill_geneva --tier 1

# 3) Curate per-transition materialised views
python -m videogen.manage_views --action curate

# 4) Status sentinel
python -m videogen.verify_pipeline
```

You should see something like:

```
  PASS    source table                                500 rows  version=22
  PASS    T1 column: caption_length                   500 / 500 filled (100%)
  PASS    T1 column: keyword_melting                  500 / 500 filled (100%)
  …
  PASS    view: phase_transitions_train               450 rows  version=4
  PASS    view: phase_transitions_val                 50 rows  version=4
```

## Real data — ChronoMagic-Pro (no clip download)

You don't need the 2 TB clip archive to start curating; the captions are
the small bit.

```bash
# Pull just the caption manifest (~few MB)
python -m videogen.download_manifest --variant proh --out data/chronomagic_proh.parquet

# Ingest captions only (video_bytes left empty until you download clips)
python -m videogen.ingest_chronomagic \
    --manifest data/chronomagic_proh.parquet \
    --limit 25000 --overwrite

# Tier 1 backfill + curate as before
python -m videogen.backfill_geneva --tier 1
python -m videogen.manage_views --action curate
```

Now jump into `notebooks/eda_phase_transitions.ipynb` to explore.

## Pipeline shape

```
chronomagic-pro.parquet ─┐
synthetic mp4s ──────────┤
on-disk mp4 dir ─────────┘
                  │
                  │  ingest_chronomagic.py        (CPU)
                  ▼
         videos_raw [Lance]
                  │  backfill_geneva --tier 1     (CPU)  ← runnable today
                  │   keyword_* · caption_length
                  │
                  │  backfill_geneva --tier 2     (GPU)  ← deferred
                  │   clip_emb_* · motion · MTScore
                  │
                  │  backfill_geneva --tier 3     (GPU)  ← deferred
                  │   t5_hidden_states · vae_latent     ← headline trick
                  │
                  │  backfill_geneva --tier 4     (GPU + CPU)
                  │   dhash · is_duplicate
                  │
                  │  manage_views --action curate(-2)
                  ▼
        phase_<transition>_{train,val}   [Geneva MVs]
                  │
                  │  dataloader.make_cached_loader(...)
                  ▼
            train_wan22_lora.py   (deferred — GPU)
```

## Layout

```
videogen/                  Pipeline package
├── schema.py              Lance schema (per-tier field declarations)
├── ingest_chronomagic.py  Streaming ingest: synthetic / manifest / manifest+mp4
├── download_manifest.py   Pull just the HF caption parquet (no clips)
├── geneva_udfs.py         Tier 1 (CPU keyword) + Tier 2/3/4 stubs
├── backfill_geneva.py     Geneva backfill orchestrator (--tier, --columns)
├── manage_views.py        Materialised views per phase transition
├── dataloader.py          Permutation-based loaders (cached + raw paths)
├── spec_queries.py        Curation helpers (count, preview, FTS)
└── verify_pipeline.py     End-to-end status sentinel

bench/                     Benchmark harness — see PROPOSAL.md §"Benchmarks"
├── bench_ingest.py        B1  ingest rows/sec
├── bench_curation.py      B2  SQL+FTS query timings
└── bench_storage.py       B8  on-disk footprint per table/view

notebooks/
└── eda_phase_transitions.ipynb   Curation EDA

PROPOSAL.md                Full design doc + benchmark plan
KNOWN_ISSUES.md            Upstream regressions we're working around
```

## What's not done yet

These are tracked in `PROPOSAL.md §"Milestones"` and will be filled in
once the GPU is free:

* Tier 2/3/4 Geneva UDF bodies (CLIP / RAFT / MTScore / T5 / Wan-VAE / dHash).
* `train_wan22_lora.py` — Wan2.2 LoRA training loop reading from cached MVs.
* `bench_dataloader.py` — B5/B6 clips/sec + GPU MFU.
* `bench_backfill.py` — B3/B4 incremental backfill timings.
* `bench_e2e.py` — B10 end-to-end wall-clock.
* `eval_chronomagic.py` / `eval_vbench.py` — quantitative eval.
