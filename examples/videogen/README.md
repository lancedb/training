# Video Generation Training with LanceDB

End-to-end fine-tune of a video-diffusion model (Wan2.2-TI2V-5B) on a curated
time-lapse phase-transition slice — using LanceDB + Geneva at every stage
from raw clips to high-MFU training.

> Status: **Full pipeline runnable end-to-end on a single H100.**
> Tier 1 = caption-derived flags (CPU).
> Tier 2 = CLIP ViT-B/32 text + video embeddings, frame-absdiff motion score,
> CLIP-based MTScore proxy.
> Tier 3 = the **headline trick**: pre-tokenised UMT5-XXL hidden states and
> pre-encoded Wan2.2-VAE latents stored as Lance columns; the training loop
> reads only these and never loads the VAE or text encoder.
> Tier 4 = perceptual-hash dedup (first/last frame dHash + Hamming NN flag).
> The Wan2.2-TI2V-5B LoRA training loop trains from the cached columns
> (no VAE / no UMT5 in the train process).

## Headline numbers (1×H100, **2,255 real ChronoMagic clips**)

**Dataloader throughput — forward-only**
([`bench/bench_dataloader.py`](bench/bench_dataloader.py))

| `num_workers` | cached (Lance) | raw (decode + VAE + UMT5 in loop) | speedup |
|---:|---:|---:|---:|
| 4 | **7.73 samples/s · 28.7 % MFU** | 0.55 samples/s · 2.05 % | **× 14.2** · +27.0 pts |
| 8 | **7.73 samples/s · 28.7 % MFU** | 0.55 samples/s · 2.05 % | **× 14.1** · +27.0 pts |

**Training step — full fwd + bwd + AdamW, LoRA r=16**
([`bench/bench_train_step.py`](bench/bench_train_step.py))

| `num_workers` | cached (Lance) | raw (decode + VAE + UMT5 in loop) | speedup |
|---:|---:|---:|---:|
| 4 | **2.51 samples/s · 28.3 % MFU** | 0.48 samples/s · 5.42 % | **× 5.22** · +22.9 pts |
| 8 | **2.51 samples/s · 28.3 % MFU** | 0.48 samples/s · 5.41 % | **× 5.23** · +22.9 pts |

`nw=4` and `nw=8` give identical throughput in both modes — the model
is the bottleneck once the cache hides the VAE/UMT5 work, which is
exactly the cached path's win.  More workers can't speed up a DiT-bound
step.

### End-to-end run, 1×H100

| Stage | Wall-clock |
|---|---:|
| HF download (2.66 GB zip) | 14 s |
| Ingest 2,255 mp4s | < 5 s |
| Tier 1 (7 caption cols, Geneva) | 44 s |
| Tier 2 (CLIP + motion + MTScore, Geneva, GPU) | ~12 min |
| Tier 3 t5_hidden_states (direct add_columns, UMT5-XXL) | 110 s (24 rows/s) |
| Tier 3 vae_latent (direct add_columns, Wan-VAE 705M) | 60 min (0.6 rows/s) |
| Tier 4 dhash + IVF index + is_duplicate | ~3 min |
| Curate Tier 1 + Tier 2 views | ~30 s |
| **Train Wan2.2 LoRA, 4000 steps, r=64** | **1 704.7 s = 28 min (2.35 steps/s)** |
| Eval 8 prompts × 2 models × 20 inference steps | ~5 min |

Loss curve: 0.54 → 0.32 (step 200) → 0.22 (1 600) → 0.20 (2 000) →
0.20 (4 000).  Plateaued ~step 2 000.

### Before / after on 8 phase-transition prompts

```
metric                     baseline   fine-tuned          Δ
mtscore_proxy                0.0515       0.0704   +0.0188   (+37 %)
dynamic_degree               0.0087       0.0023   -0.0065
subject_consistency          0.9916       0.9946   +0.0031
temporal_smoothness          0.9999       1.0000   +0.0001
```

MTScore-proxy went up cleanly (+37 % relative) — the LoRA is producing
videos with more visual change between first and last frame, which is
the metamorphic-amplitude signal ChronoMagic trains for.
`dynamic_degree` dropped: the model has shifted toward slower, more
deliberate transitions vs busy motion, consistent with time-lapse
training data.  Consistency + smoothness stayed near 1.0, so the LoRA
isn't hallucinating jumps.

Side-by-side mp4s per prompt + an `index.html` viewer are at
[`eval_outputs/overnight/`](eval_outputs/overnight/).

### Curation latency — real 144 654-row ChronoMagic-ProH captions

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

bench/                     Benchmark harness
├── bench_ingest.py        ingest rows/sec
├── bench_curation.py      SQL+FTS query timings
├── bench_backfill.py      feature backfill + incremental
├── bench_dataloader.py    forward-only clips/sec + GPU MFU
├── bench_train_step.py    training-step throughput (fwd + bwd + opt)
├── bench_storage.py       on-disk footprint per table/view
├── bench_recipe_change.py one-column re-derive vs whole pipeline
└── bench_e2e.py           end-to-end wall-clock summary

scripts/
├── run_pipeline.sh        one-command pipeline runner (short demo)
└── run_overnight.sh       longer training + side-by-side eval video grid

notebooks/
└── eda_phase_transitions.ipynb   Curation EDA
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
