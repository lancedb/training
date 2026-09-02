#!/usr/bin/env python3
"""Before/after on DROID, open-loop: does the policy predict what the teleoperator did?

DROID has no simulator, so closed-loop success is impossible. What is measurable on a held-out
episode is whether the predicted action chunk tracks the actions the human actually took.
Records ground-truth and both policies' predictions so they can be overlaid.
"""
import os
import argparse, json, time
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LANCE_ROOT", "./data/droid_lance"))
    ap.add_argument("--episode", type=int, default=90000)
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="out/before_after_traces.json")
    # NOTE: both checkpoints must come from the SAME run. A pretrained base checkpoint is not
    # a valid "before" -- see README, "Why the baseline is two checkpoints of one run".
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="label=path pairs, e.g. early=runs/ckpt/000080/pretrained_model "
                         "trained=runs/ckpt/010000/pretrained_model")
    a = ap.parse_args()

    import lerobot.policies.smolvla.configuration_smolvla  # register the choice type
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.storage import load_dataset_metadata
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    rmap = json.load(open(os.environ.get("RENAME_MAP", "config/rename_map.json")))
    meta = load_dataset_metadata("lerobot/droid_1.0.1", root=a.root)
    res = {"episode": a.episode, "traces": {}}

    for pair in a.checkpoints:
        label, path = pair.split("=", 1)
        cfg = PreTrainedConfig.from_pretrained(path); cfg.pretrained_path = path
        dt = resolve_delta_timestamps(cfg, meta, rmap)
        ds = LeRobotDataset("lerobot/droid_1.0.1", root=a.root, episodes=[a.episode],
                            delta_timestamps=dt, return_uint8=True, tolerance_s=5e-3)
        pol = make_policy(cfg, ds_meta=ds.meta, rename_map=rmap).to("cuda").eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=path)
        n = min(a.steps, len(ds) - a.start)
        print(f"=== {label}: episode {a.episode}, frames {a.start}..{a.start+n} of {len(ds)}", flush=True)

        preds, gts = [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(a.start, a.start + n):
                raw = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
                       for k, v in ds[i].items()}
                batch = _preprocess_dataset_batch(raw, ds.meta.camera_keys, rmap, pre)
                batch = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                pol.reset()
                chunk = pol.predict_action_chunk(dict(batch))       # (1, T, D)
                preds.append(chunk[0, 0].float().cpu().numpy())     # first action of the chunk
                gts.append(batch["action"][0, 0].float().cpu().numpy())
        P, G = np.stack(preds), np.stack(gts)
        mae = float(np.abs(P - G).mean())
        res["traces"][label] = {"pred": P.tolist(), "mae": round(mae, 4),
                                "seconds": round(time.perf_counter() - t0, 1)}
        # Both checkpoints must see byte-identical targets, or the MAEs are not comparable:
        # each policy normalizes `action` with its OWN stats, so a stats mismatch silently
        # compares two different quantities. Fail loudly instead.
        if "ground_truth" in res and not np.allclose(res["ground_truth"], G, atol=1e-6):
            raise SystemExit(f"ABORT: {label} sees different ground truth than the previous "
                             f"checkpoint -- normalization stats differ, MAEs incomparable")
        res["ground_truth"] = G.tolist()
        res["dims"] = int(G.shape[1])
        res["start"] = a.start
        print(f"    MAE vs teleop actions: {mae:.4f}  ({P.shape[0]} steps)", flush=True)
        del pol; torch.cuda.empty_cache()

    json.dump(res, open(a.out, "w"))
    print("RESULT written", a.out, flush=True)


if __name__ == "__main__":
    main()
