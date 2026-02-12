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
from torch.utils.data import DataLoader, default_collate

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


def _iter_batches_from_index_shard(
    shard: ray.data.DataIterator,
    dataset: LeRobotDataset,
    batch_size: int,
    index_column: str,
    local_shuffle_buffer_size: int,
) -> Iterable[dict[str, Any]]:
    for index_batch in shard.iter_batches(
        batch_size=batch_size,
        batch_format="numpy",
        local_shuffle_buffer_size=local_shuffle_buffer_size,
    ):
        indices = [int(i) for i in index_batch[index_column].tolist()]
        samples = [dataset[i] for i in indices]
        yield default_collate(samples)


def train_loop_per_worker(config: dict[str, Any]) -> None:
    device = train.torch.get_device()
    world_rank = train.get_context().get_world_rank()

    training_steps = config["training_steps"]
    log_freq = config["log_freq"]
    checkpoint_freq = config["checkpoint_freq"]
    output_directory = Path(config["output_directory"])
    input_mode = config["input_mode"]
    index_column = config["index_column"]
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

    dataloader: DataLoader | None = None
    dataset_shard: ray.data.DataIterator | None = None
    if input_mode == "ray-data":
        dataset_shard = train.get_dataset_shard("train")
        if dataset_shard is None:
            raise RuntimeError(
                "input_mode='ray-data' requires a Ray dataset shard named 'train'."
            )
    else:
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
            batches = _iter_batches_from_index_shard(
                shard=dataset_shard,
                dataset=dataset,
                batch_size=config["batch_size"],
                index_column=index_column,
                local_shuffle_buffer_size=config["local_shuffle_buffer_size"],
            )
        else:
            if dataloader is None:
                raise RuntimeError(
                    "Internal error: both dataloader and dataset shard are missing."
                )
            batches = dataloader

        for batch in batches:
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
    parser.add_argument("--training-steps", type=int, default=50)
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
    parser.add_argument("--checkpoints-to-keep", type=int, default=2)
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
        train_index_ds = read_lance(args.lance_uri, columns=[args.index_column])
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
