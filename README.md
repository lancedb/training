A collection of training and benchmarking examples built on [LanceDB](https://lancedb.com) multimodal data lakehouse to demonstrate how LanceDB performs as the data layer across different model types, and training regimes along with benchmarks and best practices for training with LanceDB.

## Why use LanceDB for training?

One LanceDB table for the entire training loop

- **Curate & engineer features** — compute new signals (detections, CLIP embeddings, scene tags, dedup flags) as columns via distributed, checkpointed UDF backfills, then slice with SQL, full-text, and vector search directly on the table. Create a dataloder directly using filtered reads from the table or create a training split as *versioned materialized view*, not a CSV manifest.
- **Manage at scale** — bytes, metadata, annotations, and embeddings all live in one schema-enforced table, and **zero-copy schema evolution** lets you add a column without rewriting the data: pre-tokenize or pre-embed a multi-TB corpus once and append it as a new column for free.
- **Load & train** — With random-access, zero-copy dataloading reads straight from LanceDB tables, from local or object storage, keeping the GPU fed and shards cleanly across Ray workers.

---

### Projects using LanceDB

- **[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)** — a platform for reproducible world-model research ([paper](https://arxiv.org/abs/2605.21800)) built on a **LanceDB data layer**. **3–4× faster** data loading on Push-T vs HDF5 / MP4 at a fraction of the disk.
- **[le-wm](https://github.com/lucas-maes/le-wm)** — LeWorldModel, a stable joint-embedding predictive world model from pixels ([paper](https://arxiv.org/abs/2603.19312)), trained on the [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) platform and its LanceDB data layer.
- **[lerobot-lancedb](https://github.com/lancedb/lerobot-lancedb)** — a drop-in LanceDB backend for 🤗 LeRobot datasets, referenced in the [official LeRobot docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3#other-formats-and-implementations). **2–4× faster** data loading across PushT / ALOHA / Koch at identical training quality.

*Building on LanceDB and want to be listed here? Open a PR.*

## Examples

| Model type | Example |
|---|---|
| Object Detection (AV perception) | [object-detection/](./object-detection/) &nbsp; [![Blog](https://img.shields.io/badge/blog-read-blue)](https://www.lancedb.com/blog/unifying-the-av-ml-stack-lancedb) |
| ViT (MFU benchmark across backends) | [examples/ViT/](./examples/ViT/) |
| VLA (Vision-Language-Action) | [examples/lerobot_ray_lance/](./examples/lerobot_ray_lance/) |
| World Model / Video Generation |  🚧 |
| VLM | 🚧  |
| LLM | 🚧  |

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
