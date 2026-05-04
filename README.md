# lancedb-training

A collection of training and benchmarking examples built on [LanceDB](https://lancedb.com) multimodal data lakehouse. The goal is to demonstrate how LanceDB performs as the data layer across different model types, and training regimes along with benchmarks and best practices for training with LanceDB.

## Available examples

| Model type | Example |
|---|---|
| Object Detection (AV perception) | [object-detection/](./object-detection/) &nbsp; [![Blog](https://img.shields.io/badge/blog-read-blue)](https://www.lancedb.com/blog/unifying-the-av-ml-stack-lancedb) |
| ViT (MFU benchmark across backends) | [examples/ViT/](./examples/ViT/) |
| VLA (Vision-Language-Action) | [examples/lerobot_ray_lance/](./examples/lerobot_ray_lance/) |

## Coming soon

World Model / Video Generation · VLM · LLM

## Repository layout

```
object-detection/                 # AV perception — BDD100K + Geneva + Faster R-CNN
examples/
  ViT/                            # MFU benchmark: LanceDB vs S3 vs Parquet
  lerobot_ray_lance/              # VLA: Ray + LeRobot Diffusion Policy
  leWorldModel/                   # CogVideo / world model fine-tuning
```

Each example is self-contained and targets one concrete question.

## Setup

```bash
uv sync
```

See each example's own README for run instructions.
