#!/usr/bin/env python
"""Train any LeRobot policy from a Lance-backed dataset.

This is the *entire* migration from stock `lerobot-train` to LanceDB:
we swap the dataset class that lerobot's factory instantiates for
`LeRobotLanceVideoDataset` (bit-exact Lance video format) and hand
control straight back to `lerobot.scripts.lerobot_train`. Every other
part of training — policy, processors, sampler, optimizer, accelerate
multi-GPU — is untouched upstream code.

Usage: identical to `lerobot-train`, e.g.

    accelerate launch --multi_gpu --num_processes=4 train_lance.py \
        --dataset.repo_id=HuggingFaceVLA/libero \
        --dataset.root=/path/to/libero_lance_video \
        --policy.path=lerobot/smolvla_base ...
"""

import lerobot.datasets.factory as factory
from lerobot_lancedb import LeRobotLanceVideoDataset


class LanceVideoDataset(LeRobotLanceVideoDataset):
    # The Lance reader serves every frame by absolute index, so the
    # absolute->relative row remap used by episode-filtered parquet
    # datasets never applies. Shadow the upstream property that would
    # otherwise try to build a parquet reader.
    absolute_to_relative_idx = None


def make_lance_dataset(
    repo_id,
    root=None,
    episodes=None,
    delta_timestamps=None,
    image_transforms=None,
    revision=None,
    **_ignored,  # parquet-only kwargs: video_backend, return_uint8, depth_output_unit, ...
):
    return LanceVideoDataset(
        root=root,
        repo_id=None if root is not None else repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=revision,
        tolerance_s=_ignored.get("tolerance_s", 1e-4),
        return_uint8=True,
    )


factory.LeRobotDataset = make_lance_dataset  # <-- the switch

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
