"""
Ray Train Diffusion Policy example based on LeRobot PushT training style.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import ray
import torch
from lance_ray import read_lance
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from ray import train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader

ACTION_DELTAS = [
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
]
OBS_DELTAS = [-0.1, 0.0]


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _build_delta_timestamps(
    input_features: dict[str, Any],
    output_features: dict[str, Any],
) -> dict[str, list[float]]:
    delta_timestamps: dict[str, list[float]] = {}
    for key in input_features:
        if key.startswith("observation."):
            delta_timestamps[key] = OBS_DELTAS
    for key in output_features:
        delta_timestamps[key] = ACTION_DELTAS
    return delta_timestamps


def _normalize_visual_feature_shapes(features: dict[str, Any]) -> None:
    """Convert image/video feature shapes to CHW when metadata is HWC."""
    for feature in features.values():
        if feature.type is not FeatureType.VISUAL or len(feature.shape) != 3:
            continue

        channels, height, width = feature.shape
        if channels not in (1, 3, 4) and width in (1, 3, 4):
            feature.shape = (width, channels, height)


def _save_checkpoint_contents(
    checkpoint_dir: str,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    preprocessor: Any,
    postprocessor: Any,
    step: int,
) -> None:
    unwrapped = _unwrap_model(policy)
    policy_dir = os.path.join(checkpoint_dir, "policy")
    preprocessor_dir = os.path.join(checkpoint_dir, "preprocessor")
    postprocessor_dir = os.path.join(checkpoint_dir, "postprocessor")
    os.makedirs(policy_dir, exist_ok=True)
    os.makedirs(preprocessor_dir, exist_ok=True)
    os.makedirs(postprocessor_dir, exist_ok=True)

    unwrapped.save_pretrained(policy_dir)
    preprocessor.save_pretrained(preprocessor_dir)
    postprocessor.save_pretrained(postprocessor_dir)
    torch.save(
        {
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        os.path.join(checkpoint_dir, "trainer_state.pt"),
    )


def _iter_batches_from_shard(
    shard: ray.data.DataIterator,
    batch_size: int,
    local_shuffle_buffer_size: int,
) -> Iterable[dict[str, Any]]:
    for batch in shard.iter_torch_batches(
        batch_size=batch_size,
        local_shuffle_buffer_size=local_shuffle_buffer_size,
    ):
        yield batch


def _get_action_window_size(action: Any, expected_action_dim: int) -> int:
    if not isinstance(action, torch.Tensor):
        return len(ACTION_DELTAS)
    if action.ndim >= 3:
        return int(action.shape[1])
    if action.ndim == 2:
        return 1 if int(action.shape[1]) == expected_action_dim else int(action.shape[1])
    return 1


def _derive_action_is_pad(
    batch: dict[str, Any],
    episode_bounds: dict[int, tuple[int, int]],
    action_delta_indices: list[int],
) -> torch.Tensor:
    indices = batch["index"]
    episode_indices = batch["episode_index"]
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)
    if not isinstance(episode_indices, torch.Tensor):
        episode_indices = torch.as_tensor(episode_indices)

    action_is_pad = torch.zeros(
        (int(indices.shape[0]), len(action_delta_indices)),
        dtype=torch.bool,
        device=indices.device,
    )

    for row in range(int(indices.shape[0])):
        ep_idx = int(episode_indices[row].item())
        bounds = episode_bounds.get(ep_idx)
        if bounds is None:
            raise RuntimeError(
                f"Could not derive action padding mask: episode_index={ep_idx} "
                "not found in dataset metadata."
            )
        ep_start, ep_end = bounds
        abs_idx = int(indices[row].item())
        for col, delta in enumerate(action_delta_indices):
            target_idx = abs_idx + delta
            action_is_pad[row, col] = (target_idx < ep_start) or (target_idx >= ep_end)

    return action_is_pad


def train_loop_per_worker(config: dict[str, Any]) -> None:
    device = train.torch.get_device()
    world_rank = train.get_context().get_world_rank()

    training_steps = config["training_steps"]
    log_freq = config["log_freq"]
    checkpoint_freq = config["checkpoint_freq"]
    output_directory = Path(config["output_directory"])
    input_mode = config["input_mode"]
    video_backend = config["video_backend"]

    dataset_repo_id = config["dataset_repo_id"]
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    _normalize_visual_feature_shapes(features)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
    )
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    policy = train.torch.prepare_model(policy)

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    dataloader: DataLoader | None = None
    dataset: LeRobotDataset | None = None
    dataset_shard: ray.data.DataIterator | None = None
    episode_bounds: dict[int, tuple[int, int]] | None = None
    required_shard_keys: set[str] | None = None
    warned_action_pad_fallback = False
    shard_keys_validated = False
    if input_mode == "ray-data":
        dataset_shard = train.get_dataset_shard("train")
        if dataset_shard is None:
            raise RuntimeError(
                "input_mode='ray-data' requires a Ray dataset shard named 'train'."
            )
        episode_bounds = {
            int(ep_idx): (
                int(dataset_metadata.episodes[ep_idx]["dataset_from_index"]),
                int(dataset_metadata.episodes[ep_idx]["dataset_to_index"]),
            )
            for ep_idx in range(len(dataset_metadata.episodes))
        }
        required_shard_keys = set(input_features) | set(output_features)
    else:
        delta_timestamps = _build_delta_timestamps(input_features, output_features)
        if not delta_timestamps:
            raise RuntimeError(
                "No delta_timestamps could be inferred from dataset features. "
                "Provide a dataset with observation/action features compatible with DiffusionPolicy."
            )
        dataset = LeRobotDataset(
            dataset_repo_id,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )
        dataloader = DataLoader(
            dataset,
            num_workers=config["num_dataloader_workers"],
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=device.type != "cpu",
        )
        dataloader = train.torch.prepare_data_loader(dataloader)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"])

    step = 0
    checkpoint = train.get_checkpoint()
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                _unwrap_model(policy).load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                step = int(state.get("step", 0))

    done = step >= training_steps
    epoch = 0
    while not done:
        if dataset_shard is not None:
            batches = _iter_batches_from_shard(
                shard=dataset_shard,
                batch_size=config["batch_size"],
                local_shuffle_buffer_size=config["local_shuffle_buffer_size"],
            )
        else:
            if dataloader is None:
                raise RuntimeError(
                    "Internal error: both dataloader and dataset shard are missing."
                )
            batches = dataloader

        for batch in batches:
            if required_shard_keys is not None and not shard_keys_validated:
                missing_keys = required_shard_keys.difference(batch.keys())
                if missing_keys:
                    raise RuntimeError(
                        "Ray shard batch is missing required keys for training: "
                        f"{sorted(missing_keys)}. Ensure `--lance-uri` points to a dataset "
                        "with model-ready columns (not index-only rows)."
                    )
                shard_keys_validated = True

            if "action_is_pad" not in batch:
                action = batch.get("action")
                action_dim = cfg.action_feature.shape[0] if cfg.action_feature else 1
                action_window_size = _get_action_window_size(action, action_dim)
                if action_window_size == len(cfg.action_delta_indices):
                    action_delta_indices = cfg.action_delta_indices
                elif action_window_size == 1:
                    action_delta_indices = [0]
                else:
                    start = 1 - cfg.n_obs_steps
                    action_delta_indices = list(
                        range(start, start + action_window_size)
                    )

                if (
                    episode_bounds is not None
                    and "index" in batch
                    and "episode_index" in batch
                ):
                    batch["action_is_pad"] = _derive_action_is_pad(
                        batch=batch,
                        episode_bounds=episode_bounds,
                        action_delta_indices=action_delta_indices,
                    )
                else:
                    batch_size = (
                        int(action.shape[0])
                        if isinstance(action, torch.Tensor) and action.ndim > 0
                        else config["batch_size"]
                    )
                    mask_device = (
                        action.device if isinstance(action, torch.Tensor) else "cpu"
                    )
                    batch["action_is_pad"] = torch.zeros(
                        (batch_size, action_window_size),
                        dtype=torch.bool,
                        device=mask_device,
                    )
                    if not warned_action_pad_fallback:
                        print(
                            "Warning: deriving `action_is_pad` with an all-false fallback "
                            "because shard rows do not include both `index` and `episode_index`."
                        )
                        warned_action_pad_fallback = True
            print(f"Processing step {step}...")
            batch = preprocessor(batch)
            loss, _ = policy(batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            print(f"Finished processing step {step}...")

            step += 1
            should_log = (step % log_freq == 0) or (step == training_steps)
            should_checkpoint = (step % checkpoint_freq == 0) or (
                step == training_steps
            )

            if should_log or should_checkpoint:
                metrics = {
                    "loss": float(loss.item()),
                    "step": step,
                    "epoch": epoch,
                }

                if should_checkpoint and world_rank == 0:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        _save_checkpoint_contents(
                            checkpoint_dir=tmp_dir,
                            policy=policy,
                            optimizer=optimizer,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            step=step,
                        )
                        train.report(
                            metrics, checkpoint=Checkpoint.from_directory(tmp_dir)
                        )
                else:
                    train.report(metrics)

            if step >= training_steps:
                done = True
                break
        epoch += 1

    if world_rank == 0:
        output_directory.mkdir(parents=True, exist_ok=True)
        _save_checkpoint_contents(
            checkpoint_dir=str(output_directory),
            policy=policy,
            optimizer=optimizer,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            step=step,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--dataset-repo-id", type=str, default="lerobot/xvla-soft-fold")
    parser.add_argument(
        "--lance-uri",
        type=str,
        default="hf://datasets/lance-format/lerobot_xvla-soft-fold/data/train.lance",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["ray-data", "torch-dataloader"],
        default="ray-data",
        help="ray-data uses Lance+ray.data index sharding; torch-dataloader mirrors the original example.",
    )
    parser.add_argument("--index-column", type=str, default="index")
    parser.add_argument("--training-steps", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-dataloader-workers", type=int, default=4)
    parser.add_argument("--local-shuffle-buffer-size", type=int, default=10000)
    parser.add_argument(
        "--video-backend",
        type=str,
        choices=["pyav", "torchcodec", "video_reader"],
        default="pyav",
        help=(
            "Video decode backend for LeRobotDataset. "
            "Use pyav to avoid torchcodec shared-library issues on macOS."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument(
        "--output-directory", type=str, default="outputs/train/example_diffusion"
    )
    parser.add_argument("--run-name", type=str, default="ray-train-lerobot-diffusion")
    parser.add_argument("--storage-path", type=str, default="file:///tmp/ray-results")
    parser.add_argument("--checkpoints-to-keep", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = ray.init(address=args.ray_address)

    print(f"Dashboard URL: {context.dashboard_url}")

    datasets: dict[str, ray.data.Dataset] = {}
    if args.input_mode == "ray-data":
        train_index_ds = read_lance(args.lance_uri)
        if args.limit_rows > 0:
            train_index_ds = train_index_ds.limit(args.limit_rows)
        datasets["train"] = train_index_ds

    resume_checkpoint = (
        Checkpoint.from_directory(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )

    trainer_kwargs: dict[str, Any] = {
        "train_loop_per_worker": train_loop_per_worker,
        "train_loop_config": {
            "dataset_repo_id": args.dataset_repo_id,
            "input_mode": args.input_mode,
            "index_column": args.index_column,
            "video_backend": args.video_backend,
            "training_steps": args.training_steps,
            "log_freq": args.log_freq,
            "checkpoint_freq": args.checkpoint_freq,
            "batch_size": args.batch_size,
            "num_dataloader_workers": args.num_dataloader_workers,
            "local_shuffle_buffer_size": args.local_shuffle_buffer_size,
            "lr": args.lr,
            "output_directory": args.output_directory,
        },
        "scaling_config": ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
        ),
        "run_config": RunConfig(
            name=args.run_name,
            storage_path=args.storage_path,
            checkpoint_config=CheckpointConfig(num_to_keep=args.checkpoints_to_keep),
        ),
        "resume_from_checkpoint": resume_checkpoint,
    }
    if datasets:
        trainer_kwargs["datasets"] = datasets

    trainer = TorchTrainer(**trainer_kwargs)
    result = trainer.fit()

    print("Training finished.")
    print("Final metrics:", result.metrics)
    print("Checkpoint:", result.checkpoint)

    ray.shutdown()


if __name__ == "__main__":
    main()
