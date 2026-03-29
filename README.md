# lancedb-training

A collection of training and benchmarking examples built on [LanceDB](https://lancedb.com) multimodal data lakehouse. The goal is to demonstrate how LanceDB performs as the data layer across different model types, training regimes, and storage backends.

## What this covers

| Model type | Example | Status |
|---|---|---|
| VLA (Vision-Language-Action) | Ray + LeRobot Diffusion Policy | ✅ |
| ViT (Vision Transformer) | MFU benchmark vs S3 / Parquet | ✅ |
| VLM | — | planned |
| LLM | — | planned |
| Video Generation | — | planned |
| World Model | — | planned |

## Repository layout

```
examples/
  ViT/
  lerobot_ray_lance/
  VLM/     
```

Each example is self-contained and targets one concrete question:


## Setup

```bash
uv sync
```

See each example's own README for run instructions.
