#!/bin/bash
# Before/after evals for the ACT speed leg: untrained "before" checkpoint,
# then before/base/lance evals in gym-aloha with videos.
set -e
source ~/work/env.sh
cd ~/work/exp

# 1-step untrained "before" checkpoint
if [ ! -d ~/work/runs/act_before/checkpoints/000001 ]; then
  rm -rf ~/work/runs/act_before
  python train_lance.py \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.root=$HOME/work/data/aloha_sim_lance \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=$HOME/work/runs/act_before \
    --steps=1 --batch_size=8 --num_workers=2 --log_freq=1 --save_freq=1 --save_checkpoint=true \
    --env_eval_freq=0 --seed=1000 --wandb.enable=false > ~/work/logs/act_before.log 2>&1
fi

for job in "0 before $HOME/work/runs/act_before/checkpoints/000001/pretrained_model" \
           "1 base $HOME/work/runs/act_final_base/checkpoints/020000/pretrained_model" \
           "2 lance $HOME/work/runs/act_final_lance/checkpoints/020000/pretrained_model"; do
  set -- $job
  CUDA_VISIBLE_DEVICES=$1 nohup lerobot-eval \
    --policy.path=$3 --policy.device=cuda \
    --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.batch_size=10 --eval.n_episodes=50 --eval.use_async_envs=false \
    --output_dir=$HOME/work/runs/eval_act_$2 > ~/work/logs/eval_act_$2.log 2>&1 &
done
wait
for r in before base lance; do
  echo "act $r: $(python -c "import json; print(json.load(open('$HOME/work/runs/eval_act_$r/eval_info.json'))['per_group'] if False else json.load(open('$HOME/work/runs/eval_act_$r/eval_info.json')).get('overall'))" 2>/dev/null)"
done
echo ACT_EVALS_DONE
