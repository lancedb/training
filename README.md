# training

This project is a focused training example for running a LeRobot Diffusion Policy with Ray Train, using a Lance-backed dataset index for scalable sharding.

Primary references:

- https://huggingface.co/datasets/lance-format/lerobot_xvla-soft-fold
- https://lance.org/integrations/ray/#simple-example

Main entrypoint:

- `examples/ray_train_lerobot_lance.py`

## What this project does

The script keeps the standard LeRobot diffusion policy training shape, then adds Ray-oriented production behavior:

- persistent Ray result storage via `RunConfig(storage_path=...)`
- checkpoint retention policy via `CheckpointConfig(num_to_keep=...)`
- restart support with `--resume-from-checkpoint`
- optional `ray.data` sharding from Lance row indices (`--input-mode ray-data`)
- fallback mode that uses a regular PyTorch `DataLoader` (`--input-mode torch-dataloader`)

## High-level architecture

Training logic is split across two layers:

1. Driver process (`main`)
- Parses CLI args.
- Initializes Ray.
- Optionally builds a Ray Dataset from Lance (`read_lance(..., columns=[index])`).
- Configures and launches `TorchTrainer`.

2. Worker process (`train_loop_per_worker`)
- Loads dataset metadata from LeRobot.
- Converts dataset features to policy features.
- Normalizes visual feature shapes when metadata is channel-last (HWC) so the policy sees channel-first (CHW).
- Builds `DiffusionConfig` and `DiffusionPolicy`.
- Creates pre/post processors from dataset stats.
- Builds data input iterator (Ray shard or PyTorch DataLoader).
- Runs the optimization loop, reports metrics, and emits checkpoints.

## End-to-end data flow

1. Dataset metadata is loaded from `--dataset-repo-id`.
2. Features are split into model inputs and outputs.
3. `delta_timestamps` are inferred:
- observation features use `[-0.1, 0.0]`
- action features use `[-0.1, 0.0, ..., 1.4]`
4. `LeRobotDataset` materializes frame/action windows based on those deltas.
5. Batches are preprocessed and passed into `DiffusionPolicy`.
6. Loss is backpropagated with Adam (`--lr`).
7. Metrics are reported to Ray at `--log-freq`.
8. Checkpoints are reported at `--checkpoint-freq` and retained according to `--checkpoints-to-keep`.
9. Final artifacts are also written to `--output-directory`.

## Project layout

- `examples/ray_train_lerobot_lance.py`: training script and CLI.
- `pyproject.toml`: dependencies and project config.
- `uv.lock`: locked dependency versions for reproducible installs.

## Setup

```bash
uv sync
```

## Quickstart

```bash
uv run python examples/ray_train_lerobot_lance.py \
  --dataset-repo-id lerobot/xvla-soft-fold \
  --input-mode ray-data \
  --lance-uri hf://datasets/lance-format/lerobot_xvla-soft-fold/data/train.lance \
  --num-workers 1 \
  --storage-path file:///tmp/ray-results \
  --run-name lerobot-diffusion-ray
```

Add `--use-gpu` if GPUs are available.

## Input modes

### `ray-data` (default)

Uses Ray Data for distributed sharding. The Ray dataset contains only row indices from Lance, and each worker materializes samples from `LeRobotDataset` using those indices.

Use this for larger datasets and multi-worker runs.

### `torch-dataloader`

Bypasses Ray Data and uses a regular PyTorch `DataLoader` in each worker. This most closely matches a conventional single-process training shape.

Use this for local debugging or when you do not need Lance index sharding.

## Resume training

```bash
uv run python examples/ray_train_lerobot_lance.py \
  --resume-from-checkpoint /path/to/checkpoint_dir \
  --storage-path file:///tmp/ray-results
```

Expected checkpoint contents:

- `policy/`
- `preprocessor/`
- `postprocessor/`
- `trainer_state.pt` (model state, optimizer state, step)

When resumed, `step` continues from `trainer_state.pt`.

## Useful options

- `--training-steps`: total optimization steps (default `5000`)
- `--batch-size`: per-worker batch size (default `64`)
- `--num-workers`: Ray Train workers (default `1`)
- `--checkpoint-freq`: checkpoint/report interval in steps (default `200`)
- `--log-freq`: metric report interval in steps (default `10`)
- `--limit-rows`: cap Lance rows for quick tests (default `0`, meaning no cap)
- `--checkpoints-to-keep`: max Ray-managed checkpoints retained (default `2`)
- `--storage-path`: Ray result/checkpoint storage URI (default `file:///tmp/ray-results`)
- `--output-directory`: final local export path (default `outputs/train/example_diffusion`)

## Smoke test

```bash
uv run python examples/ray_train_lerobot_lance.py \
  --num-workers 1 \
  --limit-rows 5000 \
  --training-steps 50 \
  --log-freq 5 \
  --storage-path file:///tmp/ray-results
```

## Troubleshooting

### `PlacementGroupCleaner ... State API may be temporarily unavailable`

Usually a transient Ray warning. If training is still advancing (step/loss/checkpoint logs), it is typically safe to ignore.

### `ValueError: crop_shape should fit within the images shapes`

This can happen when image metadata is interpreted as HWC instead of CHW. The training script normalizes visual feature shapes before creating `DiffusionConfig` to avoid this mismatch.

### No progress in `ray-data` mode

Verify:

- `--lance-uri` points to a valid Lance dataset
- `--index-column` exists (default is `index`)
- network and dataset access are available
