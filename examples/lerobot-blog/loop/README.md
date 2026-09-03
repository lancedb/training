# Closing the loop: model error -> column -> query -> training set -> better model

The rest of `examples/lerobot-blog/` shows that search, scores and embeddings can live on the
table the trainer reads. This directory uses that to actually improve a model, and measures it.

The loop, in order:

| step | script | what it produces |
|---|---|---|
| 0 | `select_subset.py` | `config/loop_subset.json`: 200 random **holdout** episodes (never trained on) and a 2,000-episode **pool** |
| 1 | `train_base.sh` | the base SmolVLA checkpoint, trained on everything except the holdout |
| 2 | `score_and_embed.py` (one process per GPU) | per-frame action error of the base model + a SigLIP2 embedding, every 3rd frame of the subset |
| 3 | `merge_columns.py` | those become columns on `frames.lance` (`err_chunk_mae_base`, `err_next_mae_base`, `emb_siglip2`). No video touched. |
| 4 | `build_sets.py` | `config/loop_sets.json`: four arms of K episodes from the pool, plus two holdout slices |
| 5 | `finetune_arms.sh` (one arm per GPU) | four checkpoints, identical except for `--dataset.episodes` |
| 6 | `eval_arms.py` (one process per GPU) then `report.py` | per-episode MAE of every checkpoint on the holdout, sliced |

The arms are:

- **mined**: seed frames are where the base model is worst (p98 error, one per episode, then a
  farthest-point sample so they cover different situations). Each seed is a vector query against
  the embedding column, filtered with SQL to the pool. The nearest distinct episodes form the set.
- **hard**: the pool episodes with the highest mean base error. Same signal, no index.
- **text**: keyword expansion of the seed instructions over `language_instruction`. What
  full-text search alone gets you.
- **random**: uniform draw from the pool. The control.

Evaluation slices: all 200 holdout episodes, the 50 with the highest base error
(`hard_holdout`), and the 50 nearest to the seed frames in embedding space
(`mined_similar_holdout`). The claim to test is that `mined` beats `random` on the slices it
targeted without losing on the whole holdout, and that `hard` and `text` fall between.

## Run

```bash
source ~/venv/bin/activate
export LANCE_ROOT=/data/droid_lance        # local copy or s3://bucket/droid_1.0.1-lance
export RENAME_MAP=config/rename_map.json RUNS=~/runs
cd examples/lerobot-blog

python loop/select_subset.py                                     # 0
GPUS=4 STEPS=10000 loop/train_base.sh                            # 1  (~2 h on 4xH100 from NVMe)

BASE=$RUNS/base/checkpoints/010000/pretrained_model
for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python loop/score_and_embed.py \
    --ckpt $BASE --rank $r --world 4 & done; wait                 # 2
python loop/merge_columns.py                                     # 3  columns land on the table (add --root s3://... to target S3)
python loop/build_sets.py --k 300                                # 4  mining queries against the same table

for s in 1 2; do BASE=$BASE SEED=$s STEPS=1500 loop/finetune_arms.sh; done   # 5

for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python loop/eval_arms.py --rank $r --world 4 \
    --checkpoints base=$BASE \
      $(for a in mined hard text random; do for s in 1 2; do \
        echo ${a}_s$s=$RUNS/ft_${a}_s$s/checkpoints/001500/pretrained_model; done; done) & done; wait
python loop/report.py                                            # 6
```

## Things that keep the comparison honest

- **One base checkpoint, one set of normalization statistics.** Every arm starts from the same
  weights and the same preprocessor. `eval_arms.py` aborts if any checkpoint's action statistics
  differ, because MAEs would then be in different units (see `../README.md`, "Why the baseline is
  two checkpoints of one run").
- **The holdout is a uniform random sample and is excluded everywhere.** The base run excludes it
  with `--dataset.exclude_episodes`, every arm trains on pool episodes only, and the arms are
  built from scores on the pool. `config/cur_holdout.json` (the first 0.76% of DROID) is not used
  here; it is wrong for comparing runs that saw different data.
- **Same frames, same noise.** Every checkpoint is scored on the same spread of 60 frames per
  holdout episode with flow-matching noise seeded per batch, so a difference is the policy.
- **Equal episode counts per arm.** Frame counts differ a little between arms and are reported
  in `loop_sets.json` (`arm_frames_scored`). Two seeds per arm.
- **Paired bootstrap.** `report.py` reports the per-episode paired difference against the base
  and against the random control with a 95% interval, and the fraction of episodes improved.

## Notes

- The Lance reader is `lerobot`'s own (`storage_format: "lance"`), installed from the main
  branch: `uv pip install "lerobot[smolvla,dataset,lancedb] @ git+https://github.com/huggingface/lerobot"`.
  The 0.6.1 release on PyPI does not have it.
- `merge_columns.py` uses `lance.dataset.merge` keyed on `index`. It adds columns without
  rewriting existing fragments, so it can target the S3 table directly (`--root s3://...`): the
  write is a few hundred MB of new tabular data, the previous version stays readable, and the
  videos table is never written. `build_sets.py --root s3://...` then queries the S3 table.
- This experiment ran from a local copy of the dataset (`aws s3 sync`, about 30 minutes for
  395 GB). The GPU box was a Lambda instance in Atlanta and the bucket is in us-east-2, so S3
  reads were cross-region and capped at roughly 250 MB/s for the whole box, about a third of
  what four H100s consume. The same-region object-storage numbers are the ones in the blog.

## What happened when we ran it (4xH100, September 2026)

Three rounds, all on the same 200-episode random holdout, all with two seeds. Metric: action-chunk
MAE of the policy against the teleoperator, lower is better; the base sits around 0.369. Full
tables are in `results/`.

| round | setup | outcome |
|---|---|---|
| 1 | preset schedule (peak LR 1e-4 restarted) on 300 pool episodes the base had already trained on | every arm **worse** than the base by 0.013 to 0.05. Mined beat random on the hard slice (-0.019, CI clear of zero) and was borderline elsewhere. Hard was worst by far. |
| 2 | same sets, peak LR 1e-5 | nothing moves: every arm within 0.002 of the base, including continuing on all 2,000 pool episodes |
| 3 | base retrained **without** the pool, arms add 300 *new* episodes at peak LR 3e-5 | mined = random (-0.0001, CI [-0.002, +0.002]) on every slice. Adding all 2,000 new episodes: -0.002 vs random. Hard: **+0.025 worse** than random everywhere. Nothing beat the base. |

So, for this model, budget and metric, *which* 300 episodes you add does not matter, with one
exception: picking the episodes the model is worst on actively hurts. Those episodes are not bad
demonstrations (86% success, same as random) but the fastest ones (mean |joint velocity| 1.54 vs
1.14), and the base's error tracks speed, not quality (AUROC of episode error vs the success
label: 0.58). That is the same lesson as the smoothness gate in the main post: a single
trajectory statistic makes a poor curation filter, and it is cheap to find that out.

What the table did make cheap is the loop itself. After the base exists, one full iteration is
about 45 minutes on 4 GPUs: scoring 218,596 frames with the policy, 8 min; merging the error
columns, 6 s (62 s when the 768-d embeddings go in too); 60 mining queries, 0.27 s each once the
vector and scalar indexes exist (7.7 s each without them, 10 min to build); four fine-tuning arms
in parallel, 6 min per seed; evaluating nine checkpoints on 12,000 held-out frames, 10 min. Every
number above came out of one table at a pinned version, and the trainer read the same table.

Reasons to treat this as a first pass, not a verdict: open-loop MAE on DROID is a blunt
instrument (the blog's own before/after gain was 0.51 -> 0.37 over 10,000 steps, and we are
looking for effects a hundred times smaller from 1,500 steps); the seeds cover 30 failure modes
at once rather than one behaviour; and a closed-loop success rate on LIBERO would be a far
better readout than MAE. The scripts are ready for that variant.
