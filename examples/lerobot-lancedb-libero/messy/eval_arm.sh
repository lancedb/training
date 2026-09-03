#!/usr/bin/env bash
# Closed-loop LIBERO evaluation of one checkpoint: all four suites, ten tasks each, N rollouts
# per task, re-planning every step (n_action_steps=1, which the earlier LIBERO work showed is
# what makes SmolVLA numbers comparable).
#
#   CKPT=~/runs_libero/clean/checkpoints/040000/pretrained_model NAME=clean GPU=0 messy/eval_arm.sh
set -euo pipefail
: "${CKPT:?}"; : "${NAME:?}"
RUNS=${RUNS:-$HOME/runs_libero}; GPU=${GPU:-0}; N=${N:-10}; SUITES=${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT=$RUNS/eval_$NAME
rm -rf "$OUT"
CUDA_VISIBLE_DEVICES=$GPU lerobot-eval \
  --policy.path="$CKPT" --policy.device=cuda --policy.n_action_steps=1 \
  --env.type=libero --env.task="$SUITES" --env.max_parallel_tasks=${PARALLEL_TASKS:-2} \
  --eval.n_episodes="$N" --eval.batch_size="$N" --eval.use_async_envs=false \
  --output_dir="$OUT" > "$RUNS/eval_$NAME.log" 2>&1
python - "$OUT/eval_info.json" "$NAME" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = {k: v["pc_success"] for k, v in d.get("per_group", {}).items()}
overall = d.get("overall", {}).get("pc_success")
print(f"{sys.argv[2]}: " + "  ".join(f"{k}={v:.1f}" for k, v in rows.items()) + f"  overall={overall:.1f}  ({d.get('overall', {}).get('n_episodes')} episodes, {d.get('overall', {}).get('eval_s', 0) / 60:.0f} min)")
PY
