#!/usr/bin/env python
"""Train any LeRobot policy from a Lance-backed dataset.

The entire migration from stock `lerobot-train`: point lerobot's dataset
factory at the Lance dataset class. Everything else — policy, processors,
sampler, optimizer, accelerate multi-GPU — is untouched upstream code.

Usage: identical to `lerobot-train`, e.g.

    accelerate launch --multi_gpu --num_processes=4 train_lance.py \
        --dataset.repo_id=local/libero_video \
        --dataset.root=/path/to/libero_lance_video \
        --policy.path=lerobot/smolvla_base ...
"""

import lerobot.datasets.factory as factory
from lerobot_lancedb import LeRobotLanceVideoDataset

factory.LeRobotDataset = LeRobotLanceVideoDataset  # <-- the switch

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main

    main()
