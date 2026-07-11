# lerobot-lancedb blog — measured results (source of truth)

All: 4×H100, 104 threads (52 cores SMT2, Xeon 8480+), lerobot 0.6.0, plugin 0.2.1,
same seed/bytes both backends. "tN" = taskset to N threads total (N/4 per GPU).

## E2E training smokes (samples/s at step 300, bs64, nw8 unless noted)

### aloha_sim_transfer_cube_human (1 cam 480×640 @50fps, ACT)
| budget | base | lance | ratio |
|---|---|---|---|
| t12 (3 vCPU/GPU) | 585 | 1,442 | 2.47× |
| t16 (4) | 743 | 1,714 | 2.31× |
| t32 (8) | 1,591 | 2,052 | 1.29× |
| t48 (12) | 1,756 | 2,156 | 1.23× |
| t104 (26) | 2,068 | 2,186 | 1.06× |
| t104 nw4 (lerobot defaults) | 1,523 | 2,220 | 1.46× |
(bs32 curve: t16 1.98×, t32 1.28×, t104 1.05×)

### droid_100 (3 cams 180×320 @15fps — as published; ACT)
| budget | base | lance | ratio |
|---|---|---|---|
| t16 (4) | 655 | 1,684 | 2.57× |
| t32 (8) | 1,348 | 2,585 | 1.92× |
| t12 (3) | 453 | 1,293 | 2.85× |
| t104 (26) | 2,460 | 3,108 | 1.26× |

RoboTwin-2400eps: DNF for lance — correctness bug found+fixed (decoder-cache same-batch eviction, plugin PR#5), but many-large-files access pattern thrashes the blob/decoder cache (52 smp/s vs base 444); real plugin scaling limitation, roadmap: partial blob reads. TOURNAMENT WINNER: droid_100.

ABC-130k smoke: DISQUALIFIED — FrameTimestampError in base lerobot decode (dataset timestamp drift); no fair pair possible.

### Full 20k-step wall-clock pairs
- aloha_sim t16 bs32: lance 1,681s (28m01) vs base 3,083s (51m23) = 1.83×  [ckpts: runs/act_sim_t16_{base,lance}]
- aloha_sim t16 bs64: lance 3,193s (53m13) vs base 5,922s (98m42) = 1.85× wall
  [ckpts: runs/act_final_{base,lance}; steady smp/s over run: lance 1,603 vs base 865]
- droid t16 bs64 (4 vCPU/GPU): lance 3,004s (50m04) vs base 6,937s (115m37) = 2.31× wall; loss 0.225 both
- droid t32 bs64 (8 vCPU/GPU): lance 1,986s (33m06) vs base 3,637s (60m37) = 1.83× wall
  [ckpts: runs/droid_final_t{16,32}_{base,lance}]

## Loader benches (single proc, batch 32, steady-state smp/s)
- aloha_sim pattern: nw4 415/1,162 (2.80×), nw8 817/2,296 (2.81×), nw16 1,604/2,920 (1.82×)
- droid pattern nw8: 722/1,709 (2.37×)
- aloha static 4-cam nw8: 477/598 (1.25×)  ← heavy-decode collapse data point
- LIBERO smolvla pattern (older matrix): nw4 645/1,561, nw8 1,271/3,111, nw16 2,547/6,121 (2.4×)
- LIBERO image-parquet (banned from blog): 16/31/70 smp/s

## Quality parity (LIBERO, SmolVLA 40k identical runs, closed-loop nas=1, 100 eps/suite)
| suite | before | base | lance |
| spatial | 0 | 81 | 80 |
| object | 0 | 88 | 89 |
| goal | 0 | 83 | 77 |
| libero_10 | 0 | 71 | 82 |
| avg | 0 | 80.8 | 82.0 |
(nas sweep on lance ckpt object: 1→72, 2→23, 3→25, 5→50, 10→16, 25→1;
curated-subset model: 77% @10k ckpt; multi-suite ckpt 89-90% on object @nas1)
NOTE: LIBERO base ckpt trained on official image-parquet release — quality baseline only, NO perf claims.

## S3 (LIBERO lance table, us-east-2 same region) — capability, not benchmark
- stream training reads: 1,445 smp/s @8w, 2,458 @16w; TTFB 68s/108s
- 1.9 GB sync = 9s on this pipe

## S3 streaming (same-region, lance video, smp/s)
- droid: nw8 897 (ttfb 14.5s), nw16 1,690 — vs local mp4 722/1,604, local lance 1,709/2,920
- aloha_sim: nw8 1,013 (ttfb 9.9s), nw16 1,709 — vs local mp4 817/1,604, local lance 2,296/2,920
- CLAIM (all patterns): S3-streamed lance >= local-NVMe parquet+mp4

## Curation (LIBERO lance table, 273k rows)
- Geneva 0.14 stateful GPU UDF backfill (SigLIP2 768d): 273k frames ~10 min on 2 GPUs
- IVF-PQ index 47s; FTS 0.1s; btree <0.1s; semantic search 13-15ms; FTS 7ms; btree scan 7ms
- SQL task filter → 454 libero_object episodes → --dataset.episodes

## Sizes
- libero: image-parquet 33GB / video 1.9GB / lance 1.9GB (18MB tabular + blobs)
- aloha_cups: 486M → lance 488M; droid_100: 443M → 443M; aloha_sim: 67M

## ACT sim eval (gym-aloha transfer cube, 50 eps, sync envs)
- before: 0% | base ckpt (bs64 pair): 60% | lance ckpt: 58% — parity

## SHIPPED (final)
- blog artifact final-droid-231x; training PR#6 commit 00e812c; plugin v0.2.2 on PyPI
- headline: DROID 2.31x e2e @4vCPU/GPU (wall 50m04 vs 115m37), 1.83x @8; curve 2.85/2.57/1.92/1.26

## Pending
- aloha_sim bs64 20k pair wall-clocks; droid t12/t104 smokes; droid 20k pair
- ACT before/after evals (gym-aloha AlohaTransferCube-v0, eval_act_sim.sh; ckpts from act_sim_t16 pair@bs32 or act_final@bs64)
- blog/README rewrite (make_blog.py restructured, placeholders: WALL_RATIO, WALLCLOCK_SENTENCE, ACT_SUCCESS_SENTENCE, ACT_VIDEO_PAIRS)
- artifacts: blog fce8e873-af2c-46d2-9a00-b9b335bc8c80; videos 5e0ea9cd-c05d-4959-81bb-14466f843cee
- PRs: training#6 (needs restructure commit), plugin released 0.2.1
