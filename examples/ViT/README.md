# ViT MFU Benchmark

Measures Model FLOP Utilization (MFU) for Vision Transformer training across three storage backends: LanceDB (OSS + Enterprise), raw S3 via Boto3, and Parquet on S3.

The benchmark uses synthetic JPEG images at a fixed resolution to isolate data pipeline throughput as the variable. All three scripts run the same ViT model with the same training loop, so the MFU difference is entirely a function of how fast each backend can feed batches.

## Scripts

| Script | Backend |
|---|---|
| `mfu_bench_fp16/lance.py` | LanceDB (S3 OSS + Enterprise) |
| `mfu_bench_fp16/boto.py` | Raw S3 (loose JPEGs via Boto3) |
| `mfu_bench_fp16/parquet.py` | Parquet file on S3 via PyArrow |

`create_data.py` generates the dataset and stages it in all three backends before benchmarking.

## Setup

Requires a CUDA GPU. Configure your AWS credentials and LanceDB connection details in each script (or via environment variables).

```bash
# Generate and stage data
uv run python examples/ViT/create_data.py

# Run benchmarks
uv run python examples/ViT/mfu_bench_fp16/lance.py
uv run python examples/ViT/mfu_bench_fp16/boto.py
uv run python examples/ViT/mfu_bench_fp16/parquet.py
```

## Configuration

Each benchmark script shares the same knobs at the top of the file:

- `MODEL_NAME`: `"vit_h_14"`, `"vit_l_16"`, or `"vit_b_16"`
- `EXPECT_IMAGE_SIZE`: input resolution, default `(224, 224)`
- `BATCH_SIZE`: images per training step
- `WARMUP_STEPS` / `BENCH_STEPS`: steps to discard / steps to time
- `NUM_WORKERS`: DataLoader workers
- `PEAK_FLOPS`: your GPU's peak bfloat16 FLOPS (default set for H100/H200)

## Output

Each script prints throughput, achieved TFLOPS, and GPU MFU after the timed window.

### H200 — `vit_h_14` @ 224×224

```
--- Synthetic Pure-GPU Baseline (vit_h_14) / (224, 224) ---
batch_size=350, warmup=5, steps=50
Time Taken:     43.349 sec
Throughput:     403.70 images/sec
Achieved FLOPS: 405.221 TFLOPS
GPU MFU:        40.97%

--- LanceDB OSS Training (vit_h_14) / (224, 224) ---
batch_size=350, workers=8, steps=50
Time Taken:     47.597 sec
Throughput:     367.67 images/sec
Achieved FLOPS: 369.060 TFLOPS
GPU MFU:        37.32%

--- LanceDB Enterprise Training (vit_h_14) / (224, 224) ---
batch_size=350, workers=8, steps=50
Time Taken:     45.773 sec
Throughput:     382.32 images/sec
Achieved FLOPS: 383.762 TFLOPS
GPU MFU:        38.80%

--- Boto3 S3 Training (vit_h_14) / (224, 224) ---
batch_size=350, workers=8, steps=50
Time Taken:     137.291 sec
Throughput:     127.47 images/sec
Achieved FLOPS: 127.948 TFLOPS
GPU MFU:        12.94%

--- S3 Parquet Training (vit_h_14) / (224, 224) ---
batch_size=350, workers=8, steps=50
Time Taken:     72.393 sec
Throughput:     207.20 images/sec
Achieved FLOPS: 207.984 TFLOPS
GPU MFU:        21.03%
```