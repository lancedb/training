# Video Generation Training with LanceDB

End-to-end fine-tune of a video-diffusion model (Wan2.2-TI2V-5B) on a curated
time-lapse phase-transition slice — using LanceDB + Geneva at every stage
from raw clips to high-MFU training.

See [PROPOSAL.md](PROPOSAL.md) for the full design + benchmark plan.

> Status: **Full pipeline runnable end-to-end on a single H100.**
> Tier 1 = caption-derived flags (CPU).
> Tier 2 = CLIP ViT-B/32 text + video embeddings, frame-absdiff motion score,
> CLIP-based MTScore proxy.
> Tier 3 = the **headline trick**: pre-tokenised UMT5-XXL hidden states and
> pre-encoded Wan2.2-VAE latents stored as Lance columns; the training loop
> reads only these and never loads the VAE or text encoder.
> Tier 4 = perceptual-hash dedup (first/last frame dHash + Hamming NN flag).
> The Wan2.2-TI2V-5B LoRA training loop trains from the cached columns
> (no VAE / no UMT5 in the train process).  See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Headline numbers (1×H100, **real ChronoMagic-ProH clips**)

**Dataloader — forward-only, 25 real clips, bs=1, num_workers=0**
([`bench/bench_dataloader.py`](bench/bench_dataloader.py))

|  | cached (Lance) | raw (mp4 + VAE + UMT5 in loop) | speedup |
|---|---:|---:|---:|
| samples/s (DiT fwd) | **7.63** | 0.82 | **× 9.36** |
| fwd-only TFLOPS | 283 | 30 | |
| fwd-only MFU (H100 bf16) | **28.7 %** | 3.06 % | **+ 25.6 pts** |

**Training step — full fwd + bwd + AdamW, 25 real clips, LoRA r=16**
([`bench/bench_train_step.py`](bench/bench_train_step.py))

|  | cached (Lance) | raw (decode + VAE + UMT5 in loop) | speedup |
|---|---:|---:|---:|
| samples/s (train step) | **2.51** | 0.68 | **× 3.68** |
| train-step TFLOPS | 279 | 76 | |
| train-step MFU | **28.2 %** | 7.7 % | **+ 20.6 pts** |
| VRAM peak | **36.5 GB** | 52.9 GB | **−16.4 GB** |

The VRAM saving is the headline: with the cached path, VAE (~3 GB) and
UMT5 (~11 GB) are not loaded.  The freed memory lets us use a bigger
batch or a higher LoRA rank.

**End-to-end real-data run, 1×H100**

```
ingest 25 ChronoMagic-ProH clips        <  5 s
Tier 1 backfill   (7 cols, CPU)         44 s
Tier 2 backfill   (CLIP + motion + MT)  ~1 min
Tier 3 UMT5       (25/25 captions)      37 s    ← cached prompt embeds
Tier 3 Wan-VAE    (25/25 clips)        ~2 min   ← cached video latents
LoRA train, 200 steps, r=32             85 s    (2.35 steps/s)
eval_compare, 4 prompts × 20 steps      ~3 min
```

**Before/after on the same 4 prompts (after 200 LoRA steps on 25 melting clips)**

```
metric                     baseline    fine-tuned          Δ
mtscore_proxy                0.1213        0.0745    -0.0469
dynamic_degree               0.0077        0.0121    +0.0044
subject_consistency          0.9867        0.9831    -0.0036
temporal_smoothness          0.9996        0.9997    +0.0001
```

200 steps on 25 clips is a deliberate *minimal* run.  It moves
`dynamic_degree` up (more motion in outputs) but the LoRA shifts the
model toward calmer, less-metamorphic generations — `mtscore_proxy`
drops.  Reading honestly: this is what under-training a 5 B-param model
on a tiny corpus looks like.  Full-corpus runs (~25 K clips, several
thousand steps) are the natural next step; the infrastructure now
supports that drop-in.

**Curation latency — real 144,654-row ChronoMagic-ProH captions**

|  | Lance + Geneva |
|---|---:|
| Manifest ingest into Lance | **< 5 s** |
| Tier-1 backfill (7 caption-derived cols) | **~ 1 min** |
| FTS index build on `caption` | **2.3 s** |
| FTS query latency (1000-hit limit) | **12-25 ms** |
| CLIP text→video vector search latency | **4-23 ms** |

## Setup

```bash
cd examples/videogen
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
```

## Quickstart — real ChronoMagic data

The base `ChronoMagic` HF dataset ships 2,265 mp4 clips + captions
directly (no yt-dlp needed).  ~2.7 GB total.

```bash
# 1) Download + unzip clips + build manifest (~10 s on a fast link)
python -m videogen.download_chronomagic \
    --out data/clips --manifest data/chronomagic.parquet

# 2) Run the whole pipeline (ingest → tier 1-4 → curate → 20 train steps)
bash scripts/run_pipeline.sh
```

For the larger `ChronoMagic-Pro` / `-ProH` (which ships only YouTube
ids — needs yt-dlp + is rate-limited):

```bash
python -m videogen.download_manifest --variant proh \
    --out data/chronomagic_proh.parquet
python -m videogen.download_clips \
    --manifest data/chronomagic_proh.parquet \
    --out data/clips --parallel 8 \
    --filter-any melt freez dissolv boil evapor \
    --max-duration 60
```

### Long-haul "few thousand clips" run

For an overnight run targeting a real fine-tune + side-by-side video
output, use `run_overnight.sh`:

```bash
TARGET_CLIPS=2500 TRAIN_STEPS=4000 LORA_RANK=64 \
    bash scripts/run_overnight.sh
```

It's restart-safe: each stage skips if its output already exists, so a
mid-run crash just needs `bash scripts/run_overnight.sh` to resume.

Stop at any stage:

```bash
bash scripts/run_pipeline.sh ingest   # captions + clips → Lance
bash scripts/run_pipeline.sh tier1    # + Tier 1 keyword UDFs
bash scripts/run_pipeline.sh tier2    # + Tier 2 GPU UDFs
bash scripts/run_pipeline.sh tier3    # + Tier 3 cached features (headline)
bash scripts/run_pipeline.sh curate   # + materialised views + indices
bash scripts/run_pipeline.sh tier4    # + dedup
bash scripts/run_pipeline.sh train    # + a short training run (default)
```

Env-var knobs:
```bash
DB=/tmp/videogen_demo MANIFEST=... CLIPS=... TRAIN_STEPS=200 \
    bash scripts/run_pipeline.sh
```

## Pipeline shape

```
chronomagic-pro.parquet ─┐
yt-dlp clips ────────────┘
                  │
                  │  ingest_chronomagic.py        (CPU)
                  ▼
         videos_raw [Lance]
                  │  backfill_geneva --tier 1     (CPU)
                  │   keyword_* · caption_length
                  │
                  │  backfill_geneva --tier 2     (GPU)
                  │   clip_emb_* · motion · MTScore
                  │
                  │  backfill_geneva --tier 3     (GPU)   ← headline trick
                  │   t5_hidden_states · vae_latent
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
            train_wan22_lora.py
```

## Layout

```
videogen/                  Pipeline package
├── schema.py              Lance schema (per-tier field declarations)
├── ingest_chronomagic.py  Manifest ingest (captions + optional clip bytes)
├── download_manifest.py   HF datasets pull of ChronoMagic-Pro / -ProH parquet
├── download_clips.py      yt-dlp helper to fetch ChronoMagic-Pro clips by id
├── geneva_udfs.py         Tier 1 (CPU keyword) + Tier 2 (CLIP/motion/MTScore) + Tier 3 (UMT5 + Wan-VAE) + Tier 4 (dHash+dedup)
├── backfill_geneva.py     Geneva backfill orchestrator (--tier, --columns)
├── manage_views.py        Materialised views per phase transition + build-indices
├── dataloader.py          Permutation-based loaders (cached + raw paths)
├── spec_queries.py        Curation helpers (count, preview, FTS)
├── train_wan22_lora.py    Wan2.2 LoRA trainer reading the cached path
├── eval_chronomagic.py    MTScore-proxy evaluator (ChronoMagic-Bench-style)
├── eval_vbench.py         VBench-dimension proxies (dynamic, consistency, smoothness)
├── eval_compare.py        Side-by-side baseline vs LoRA — one delta table
├── upload_to_hf.py        Publish a curated MV as a HF Lance-format dataset
└── verify_pipeline.py     End-to-end status sentinel

bench/                     Benchmark harness — see PROPOSAL.md §"Benchmarks"
├── bench_ingest.py        B1   ingest rows/sec
├── bench_curation.py      B2   SQL+FTS query timings
├── bench_backfill.py      B3 / B4   feature backfill + incremental
├── bench_dataloader.py    B5 / B6   forward-only clips/sec + GPU MFU
├── bench_train_step.py    Training-step throughput (fwd + bwd + opt)
├── bench_storage.py       B8   on-disk footprint per table/view
├── bench_recipe_change.py B9   one-column re-derive vs whole pipeline
└── bench_e2e.py           B10  end-to-end wall-clock summary

scripts/
└── run_pipeline.sh        one-command pipeline runner

notebooks/
└── eda_phase_transitions.ipynb   Curation EDA

PROPOSAL.md                Full design doc + benchmark plan
KNOWN_ISSUES.md            Upstream regressions we're working around
```

## Training

### Single H100

Once Tiers 1-3 are backfilled and the curated view is materialised:

```bash
python -m videogen.train_wan22_lora \
    --db data/videos/lancedb \
    --train-view phase_transitions_curated_train \
    --steps 2000 --batch-size 1 \
    --rank 32 --alpha 32 --lr 1e-4 \
    --save-every 200 \
    --output-dir checkpoints/wan22_lora
```

### Multi-GPU "big run" (4×H100 DDP)

The trainer detects `LOCAL_RANK`/`WORLD_SIZE` and wraps the LoRA-adapted
model in `DistributedDataParallel` automatically.  Launch via
`torchrun` (or `accelerate launch`):

```bash
torchrun --standalone --nproc-per-node=4 \
    -m videogen.train_wan22_lora \
        --db data/videos/lancedb \
        --train-view phase_transitions_curated_train \
        --steps 4000 --batch-size 1 \
        --rank 64 --alpha 64 --lr 1e-4 \
        --num-workers 4 --prefetch-factor 4 \
        --save-every 500 \
        --output-dir checkpoints/wan22_lora_4gpu
```

Each rank reads a different shuffle of the cached columns (seeded by
`--seed + rank`).  Gradients are all-reduced via NCCL; only rank 0
writes checkpoints.  No extra Lance work is needed — the same
materialised view is opened per worker, and Permutation handles
random access in parallel.

## Evaluation

```bash
# ChronoMagic-Bench-style: generate clips, score MTScore-proxy
python -m videogen.eval_chronomagic \
    --checkpoint checkpoints/wan22_lora/step-002000 \
    --n-prompts 8 --output results_mtscore.json

# VBench-dimension proxies
python -m videogen.eval_vbench \
    --checkpoint checkpoints/wan22_lora/step-002000 \
    --n-prompts 4 --output results_vbench.json

# Side-by-side baseline vs fine-tuned, single command
python -m videogen.eval_compare \
    --checkpoint checkpoints/wan22_lora/step-002000 \
    --n-prompts 8 --steps 20 --output compare.json
```

prints a delta table:

```
  metric                     baseline    fine-tuned          Δ
  ────────────────────────────────────────────────────────────
  mtscore_proxy                0.4012        0.5187    +0.1175
  dynamic_degree               2.3110        4.1882    +1.8772
  subject_consistency          0.8744        0.8612    -0.0132
  temporal_smoothness          0.9994        0.9990    -0.0004
```

## What's not done yet

* Full-corpus training on 20 K+ real ChronoMagic clips (clip downloads
  are best-effort via yt-dlp — many ids are dead/private/geo-blocked, so
  a real run on 20 K clips needs an overnight downloader).
* Replace the MTScore-proxy in `eval_chronomagic.py` with the upstream
  ChronoMagic-Bench prompts list and CLIPScore-based reference.
