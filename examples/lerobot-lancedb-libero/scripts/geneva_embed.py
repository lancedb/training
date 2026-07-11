#!/usr/bin/env python
"""Backfill SigLIP2 frame embeddings with Geneva — LanceDB's feature-engineering engine.

This is the canonical way to add a feature column to a Lance table at scale:
register a virtual column backed by a UDF, then run a distributed, checkpointed
backfill job. The UDF here is stateful (loads SigLIP2 once per worker) and GPU
scheduled (num_gpus=1); it decodes pixels directly from the Lance video blob
table via LeRobotLanceVideoDataset, so the frames table itself stays video-sized.

    python geneva_embed.py --lance-root ~/work/data/libero_lance_video \
        [--column emb_image] [--concurrency 4]
"""

import argparse

import pyarrow as pa
from geneva import connect, udf

MODEL_ID = "google/siglip2-base-patch16-256"
DIM = 768


@udf(data_type=pa.list_(pa.float32(), DIM), num_gpus=1, input_columns=["index"],
     max_checkpoint_size=1024)
class EmbedAgentview:
    """Stateful GPU UDF: frame index -> SigLIP2 embedding of the agentview camera."""

    def __init__(self, lance_root: str, camera: str = "observation.images.image"):
        self.lance_root = lance_root
        self.camera = camera
        self.model = None

    def _setup(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        from lerobot_lancedb import LeRobotLanceVideoDataset

        self.torch = torch
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.ds = LeRobotLanceVideoDataset(root=self.lance_root, return_uint8=True)

    def __call__(self, index: pa.Array) -> pa.Array:
        if self.model is None:
            self._setup()
        torch = self.torch
        items = self.ds.__getitems__(index.to_pylist())
        imgs = [it[self.camera].permute(1, 2, 0).numpy() for it in items]
        with torch.no_grad():
            pixel = self.processor(images=imgs, return_tensors="pt")
            pixel = {k: v.to("cuda", dtype=torch.float16 if v.is_floating_point() else None)
                     for k, v in pixel.items()}
            emb = self.model.get_image_features(**pixel)
            if not torch.is_tensor(emb):
                emb = emb.pooler_output
            emb = torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy()
        return pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), pa.float32()), DIM)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lance-root", required=True)
    p.add_argument("--column", default="emb_image")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    db = connect(args.lance_root)
    tbl = db.open_table("libero")
    if args.column in [f.name for f in tbl.schema]:
        print(f"dropping existing column {args.column}")
        tbl.drop_columns([args.column])
    tbl.add_columns({args.column: EmbedAgentview(args.lance_root)})
    job = tbl.backfill(args.column, concurrency=args.concurrency)
    print("backfill job:", job)
    print("column ready:", args.column, tbl.count_rows(f"{args.column} IS NOT NULL"), "rows")


if __name__ == "__main__":
    main()
