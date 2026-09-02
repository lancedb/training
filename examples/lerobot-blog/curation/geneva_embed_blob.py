#!/usr/bin/env python
"""Geneva GPU UDF that reads the Lance tables directly -- no lerobot dataset, no DataLoader.

The frames table says which episode and frame; meta.lance carries the episodes parquet that
says which mp4 and at what offset; videos.lance holds the mp4 bytes in a blob column.
`take_blobs` hands back a file-like object supporting byte-range reads, and torchcodec decodes
straight out of it. That is the whole dependency chain, and it is all one dataset.
"""
import os
import argparse, io, json, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import lance
from geneva import connect, udf

MODEL_ID = "google/siglip2-base-patch16-224"
DIM = 768
CAM = "observation.images.exterior_1_left"


def episode_map(root: str, camera: str):
    """episode -> (video row id, first frame position inside that mp4). Read from meta.lance."""
    me = lance.dataset(f"{root}/meta.lance").to_table(columns=["path", "data"]).to_pydict()
    ep_tbl = None
    for p, d in zip(me["path"], me["data"]):
        if "episodes" in p and p.endswith(".parquet"):
            ep_tbl = pq.read_table(io.BytesIO(bytes(d)))
            break
    info = json.loads(bytes(dict(zip(me["path"], me["data"]))["info.json"]))
    fps = info["fps"]
    vi = lance.dataset(f"{root}/videos.lance").to_table(
        columns=["video_key", "chunk_index", "file_index"]).to_pydict()
    vrow = {(k, c, f): i for i, (k, c, f) in
            enumerate(zip(vi["video_key"], vi["chunk_index"], vi["file_index"]))}
    d = ep_tbl.to_pydict()
    out = {}
    for i, ep in enumerate(d["episode_index"]):
        key = (camera, d[f"videos/{camera}/chunk_index"][i], d[f"videos/{camera}/file_index"][i])
        from_ts = d[f"videos/{camera}/from_timestamp"][i]
        out[ep] = (vrow[key], int(round(from_ts * fps)))
    return out, fps


@udf(data_type=pa.list_(pa.float32(), DIM), num_gpus=1,
     input_columns=["episode_index", "frame_index"], max_checkpoint_size=1024)
class EmbedFromBlob:
    """Stateful GPU UDF: (episode, frame) -> SigLIP2 embedding, decoded from the blob column."""

    def __init__(self, root: str, epmap: dict, camera: str = CAM):
        self.root, self.epmap, self.camera = root, epmap, camera
        self.model = None

    def _setup(self):
        import os, glob, ctypes, torch
        # torchcodec dlopens libnppicc by soname; the wheel dir must be on the linker path
        # BEFORE the library loads. Preloading with ctypes alone does not satisfy it.
        npp = "$NPP_LIB"
        os.environ["LD_LIBRARY_PATH"] = npp + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        for so in glob.glob("$VENV/lib/python*/site-packages/nvidia/*/lib/lib*.so*"):
            try: ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except Exception: pass
        from transformers import AutoModel, AutoProcessor
        from torchcodec.decoders import VideoDecoder
        self.torch, self.VideoDecoder = torch, VideoDecoder
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to("cuda").eval()
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.videos = lance.dataset(f"{self.root}/videos.lance")
        self.dec = {}

    def _decoder(self, vrow: int):
        if vrow not in self.dec:
            blob = self.videos.take_blobs("video_bytes", indices=[vrow])[0]  # byte-range handle
            self.dec[vrow] = self.VideoDecoder(blob)
            if len(self.dec) > 4:                                     # bounded, like the reader
                self.dec.pop(next(iter(self.dec)))
        return self.dec[vrow]

    def __call__(self, episode_index: pa.Array, frame_index: pa.Array) -> pa.Array:
        if self.model is None:
            self._setup()
        torch = self.torch
        eps, fis = episode_index.to_pylist(), frame_index.to_pylist()
        imgs, keep = [], []
        for k, (ep, fi) in enumerate(zip(eps, fis)):
            loc = self.epmap.get(ep)
            if loc is None:
                continue
            vrow, base = loc
            try:
                fr = self._decoder(vrow).get_frames_at([base + fi]).data[0]   # CHW uint8
                imgs.append(fr.permute(1, 2, 0).cpu().numpy()); keep.append(k)
            except Exception as e:
                self.bad = getattr(self, "bad", 0) + 1
                if self.bad <= 3:
                    print(f"decode fail ep{ep} f{fi}: {type(e).__name__}: {e}"[:160], flush=True)
        out = np.zeros((len(eps), DIM), dtype="float32")
        if imgs:
            with torch.no_grad():
                px = self.proc(images=imgs, return_tensors="pt")
                px = {kk: v.to("cuda", dtype=torch.float16 if v.is_floating_point() else None)
                      for kk, v in px.items()}
                e = self.model.get_image_features(**px)
                if not torch.is_tensor(e):
                    e = e.pooler_output
                e = torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
            for k, row in zip(keep, e):
                out[k] = row
        return pa.FixedSizeListArray.from_arrays(pa.array(out.reshape(-1), pa.float32()), DIM)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lance-root", required=True)
    p.add_argument("--column", default="emb_blob")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--out", default="out/geneva_blob.json")
    a = p.parse_args()

    epmap, fps = episode_map(a.lance_root, CAM)
    print(f"episode map: {len(epmap)} episodes, fps {fps}", flush=True)

    tbl = connect(a.lance_root).open_table("frames")
    res = {"rows": tbl.count_rows(), "column": a.column, "concurrency": a.concurrency}
    if a.column in [f.name for f in tbl.schema]:
        tbl.drop_columns([a.column])
    t = time.perf_counter()
    tbl.add_columns({a.column: EmbedFromBlob(a.lance_root, epmap)})
    res["declare_s"] = round(time.perf_counter() - t, 3)
    print(f"declared in {res['declare_s']}s", flush=True)
    t = time.perf_counter()
    tbl.backfill(a.column, concurrency=a.concurrency)
    res["backfill_s"] = round(time.perf_counter() - t, 1)
    res["non_null"] = tbl.count_rows(f"{a.column} IS NOT NULL")
    res["frames_per_s"] = round(res["non_null"] / max(res["backfill_s"], 1e-9), 1)
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
