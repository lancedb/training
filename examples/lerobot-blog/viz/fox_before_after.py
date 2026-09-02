#!/usr/bin/env python
"""Foxglove server replaying a held-out DROID episode with BOTH policies' predictions.

Four channels: the camera, the teleoperator's actual actions, the base policy's predictions,
and the finetuned policy's. Scrubbing the episode shows the base policy wandering while the
finetuned one tracks the human -- the same comparison as a trace plot, but you can drive it.

Frames come from the Lance blob column; the predictions come from a prior offline pass so this
server does no GPU work.
"""
import os
import argparse, json, time
import numpy as np, lance
import foxglove
from foxglove.channels import RawImageChannel
from foxglove.schemas import RawImage, Timestamp

ROOT = os.environ.get("LANCE_ROOT", "./data/droid_lance")
CAM = "observation.images.exterior_1_left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="out/before_after_ep226.json")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--loops", type=int, default=6)
    ap.add_argument("--dims", default="0,5,6",
                    help="which action dims to publish; all 8 overlaid is unreadable")
    a = ap.parse_args()

    d = json.load(open(a.traces))
    labels = list(d["traces"].keys())          # ["early", "trained"]
    G = np.array(d["ground_truth"])
    B = np.array(d["traces"][labels[0]]["pred"])
    A = np.array(d["traces"][labels[1]]["pred"])
    ep, n, start = d["episode"], len(G), d.get("start", 0)
    print(f"episode {ep}, frames {start}..{start+n}, {G.shape[1]} action dims", flush=True)

    # decode the camera frames for this episode once, straight from the blob column
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lerobot/droid_1.0.1", root=ROOT, episodes=[ep],
                        delta_timestamps={CAM: [0.0]}, return_uint8=True, tolerance_s=5e-3)
    imgs = []
    for i in range(start, start + n):
        x = ds[i][CAM]
        arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        if arr.ndim == 4: arr = arr[0]
        if arr.shape[0] in (1, 3): arr = np.transpose(arr, (1, 2, 0))
        imgs.append(np.ascontiguousarray(arr.astype(np.uint8)))
    print(f"decoded {len(imgs)} frames {imgs[0].shape}", flush=True)

    dims = [int(x) for x in a.dims.split(",")]
    print("publishing dims", dims, flush=True)
    server = foxglove.start_server(host="127.0.0.1", port=a.port)
    cam = RawImageChannel(topic="/camera")
    # Reuse lerobot's own scalar schema -- without a schema Lichtblick cannot resolve the
    # `.scalars[:]` plot path and the panels come up empty.
    from lerobot.utils.foxglove_visualization import _SCALARS_SCHEMA
    ch = {k: foxglove.Channel(topic=t, schema=_SCALARS_SCHEMA, message_encoding="json")
          for k, t in (("gt", "/action/teleoperator"),
                       ("before", "/action/policy_80_steps"),
                       ("after", "/action/policy_10k_steps"))}
    print("Started server", flush=True)
    time.sleep(2.0)

    dt = 1.0 / a.fps
    for loop in range(a.loops):
        for i in range(n):
            t = time.time()
            ts = Timestamp.from_epoch_secs(t)
            h, w, _ = imgs[i].shape
            cam.log(RawImage(data=imgs[i].tobytes(), width=w, height=h,
                             encoding="rgb8", step=w * 3), log_time=int(t * 1e9))
            for key, arr in (("gt", G), ("before", B), ("after", A)):
                ch[key].log({"scalars": [{"label": f"dim{j}", "value": float(arr[i, j])}
                                          for j in dims]},
                            log_time=int(t * 1e9))
            time.sleep(max(0.0, dt - (time.time() - t)))
        print(f"loop {loop+1}/{a.loops} done", flush=True)
    server.stop()


if __name__ == "__main__":
    main()
