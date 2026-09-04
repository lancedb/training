# What bad data costs, measured closed-loop on LIBERO

LIBERO is clean: 1,693 demonstrations, all successful, all labelled correctly. Real collections
are not. This experiment breaks LIBERO in three realistic ways, finds the damage with columns on
the training table, and measures the whole thing with closed-loop success in the simulator.

| step | script | what it does |
|---|---|---|
| 1 | `make_messy.py` | corrupts a disjoint random 10% of episodes per defect (30% total) and writes the truth to `messy_manifest.json`, outside the table |
| 2 | `detect.py` | adds `jerk_score` (per frame), `act_lag`, `goal_dist` and `quality_flag` (per episode) to `frames.lance`; flags episodes with robust z-scores; grades the flags against the manifest |
| 3 | `train_arm.sh` | SmolVLA full fine-tune, 40,000 steps, batch 32, one GPU, same recipe for every arm |
| 4 | `eval_arm.sh` | `lerobot-eval` closed-loop, 4 suites x 10 tasks x 10 rollouts, `n_action_steps=1` |
| all | `run_messy.sh` | the three main arms end to end (clean / messy / curated), skipping finished stages |

## The defects

| defect | what is changed | real-world analogue |
|---|---|---|
| `label_swap` | `task_index` replaced by another task of the same suite | mislabeled demonstration |
| `action_noise` | Gaussian noise (0.35 x per-dim std) on the arm dims, a spike on 4% of frames, gripper flipped on 2% | shaky or lagging teleop device |
| `misaligned` | action stream shifted 6 frames (0.6 s) ahead of the observations | logging / timestamp bug |

Only the tabular frames table is rewritten, in the same row order. Videos are hardlinked, not copied.

## The detectors

Each defect gets one column and one rule. Thresholds are robust z-scores (median / MAD) within
task, 3.0 by default. Nothing reads the manifest except the final grading.

| defect | column | rule | precision | recall |
|---|---|---|---|---|
| misaligned | `act_lag`: the lag (0..10 frames) at which commanded translation best correlates with the observed end-effector displacement | best lag >= 2 and gain > 0.05 | 1.00 | 1.00 |
| action_noise | `jerk_score`: per-frame sum of |d action| over the arm dims, averaged per episode | z > 3 within task | 0.91 | 1.00 |
| label_swap | `goal_dist`: cosine distance of the mean SigLIP2 embedding of the last 3 frames to the median of every other episode with the same label | z > 3 within task | 0.85 | 0.78 |
| any | `quality_flag != 'ok'` | | **0.96** | **0.93** |

494 of 1,693 episodes flagged: 474 truly bad, 20 clean. 33 bad episodes slipped through, 38 of them
label swaps, almost all in `libero_spatial`, where every task ends with the same bowl on the same
plate so the final frame cannot separate the labels. Writing the four columns took 0.2 s (table
version 1 -> 6). A first version of the misalignment check used a plain correlation threshold and
flagged 59 clean episodes; the lag test replaced it.

## Results (4xH100, SmolVLA, 40k steps each, 400 rollouts per model)

| trained on | episodes | spatial | object | goal | libero_10 | overall |
|---|---|---|---|---|---|---|
| clean | 1,693 | 91 | 87 | 88 | 78 | **86.0** |
| messy (30% bad) | 1,693 | 74 | 71 | 48 | 58 | **62.7** |
| curated (three queries) | 1,199 | 79 | 75 | 85 | 69 | **77.0** |
| oracle (exactly the bad episodes removed) | 1,186 | 84 | 86 | 80 | 71 | **80.2** |

The mess cost 23 points; three queries got 14 back. The goal suite, whose tasks share one scene and
differ only in the instruction, went 48 -> 85. Per-task numbers are in `results/eval_*.json`;
`results/curation.json` and `results/curation_episodes.csv` hold every episode's scores and flags.

Timings: training 2.6 h per arm at 0.239 s/step (134 samples/s, data wait under 1%); closed-loop
evaluation 70-77 min per model with two tasks in parallel on one GPU.

## Run

```bash
source ~/venv-libero/bin/activate      # lerobot main with [libero,smolvla,dataset,lancedb], pylance
# dataset: HuggingFaceVLA/libero -> video variant (../scripts/make_video_variant.py) -> lerobot-lance-convert
python messy/make_messy.py --src /data/libero_lance --dst /data/libero_messy_lance
CLEAN=/data/libero_lance MESSY=/data/libero_messy_lance messy/run_messy.sh
```

Headless rendering needs `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`, the NVIDIA EGL userspace library
(`libnvidia-gl-<driver>-server`) and `libegl1`. The first `import libero` asks an interactive
question; pipe `yes` into it once.

The oracle, which removes exactly the 507 bad episodes, reaches 80.2. So of the 9 points between the
query-curated model (77.0) and the clean one (86.0), roughly 3 come from the 33 bad episodes that
slipped through and roughly 6 from training on 30% fewer episodes. Per-suite differences between the
oracle and the curated model (+5, +11, -5, +2) are within binomial noise at 100 rollouts per suite.
