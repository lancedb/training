#!/bin/bash
# Constrained-CPU training smokes: taskset the entire launch to N threads,
# identical for both backends. Usage: smoke_cpubudget.sh <threads> <preset> <bs> [steps]
set -e
source ~/work/env.sh
cd ~/work/exp
T=$1; PRESET=$2; BS=$3; STEPS=${4:-400}; NW=${5:-8}
CPUS="0-$((T-1))"
case $PRESET in
  aloha_sim) REPO=lerobot/aloha_sim_transfer_cube_human; BROOT=$HOME/work/data/aloha_sim; LROOT=$HOME/work/data/aloha_sim_lance; POL="--policy.type=act";;
  aloha)     REPO=lerobot/aloha_static_cups_open; BROOT=$HOME/work/data/aloha_cups; LROOT=$HOME/work/data/aloha_cups_lance; POL="--policy.type=act";;
  robotwin)  REPO=lerobot/robotwin_unified; BROOT=$HOME/work/data/robotwin; LROOT=$HOME/work/data/robotwin_lance; POL="--policy.type=act";;
  abc)       REPO=lerobot/abc_130k_v3_smoke; BROOT=$HOME/work/data/abc_smoke; LROOT=$HOME/work/data/abc_smoke_lance; POL="--policy.type=act";;
  droid)     REPO=lerobot/droid_100; BROOT=$HOME/work/data/droid100; LROOT=$HOME/work/data/droid100_lance; POL="--policy.type=act";;
  libero)    REPO=local/libero_video; BROOT=$HOME/work/data/libero_video; LROOT=$HOME/work/data/libero_lance_video; POL="--policy.path=lerobot/smolvla_base --policy.input_features=null --policy.output_features=null";;
esac
for mode in base lance; do
  root=$([ $mode = lance ] && echo $LROOT || echo $BROOT)
  launcher=$([ $mode = lance ] && echo "$HOME/work/exp/train_lance.py" || echo "$(which lerobot-train)")
  out=$HOME/work/runs/smoke_${PRESET}_t${T}_bs${BS}_nw${NW}_$mode
  rm -rf $out
  taskset -c $CPUS accelerate launch --multi_gpu --num_processes=4 $launcher \
    --dataset.repo_id=$REPO --dataset.root=$root $POL \
    --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=$out --steps=$STEPS --batch_size=$BS --num_workers=$NW --log_freq=100 \
    --save_checkpoint=false --env_eval_freq=0 --seed=1000 --wandb.enable=false \
    > ~/work/logs/smoke_${PRESET}_t${T}_bs${BS}_nw${NW}_$mode.log 2>&1
  echo "$PRESET t$T bs$BS nw$NW $mode: $(tr '\r' '\n' < ~/work/logs/smoke_${PRESET}_t${T}_bs${BS}_nw${NW}_$mode.log | grep -oE "step:$STEPS .*smp/s:[0-9]+" | tail -1)"
done
