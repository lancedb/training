import io
from functools import partial

import boto3
import botocore
import lancedb
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import torch
from PIL import Image
from lancedb.permutation import Permutation
from torchvision import transforms


_to_tensor = transforms.ToTensor()


def _decode_jpegs(bytes_list, image_size):
    n = len(bytes_list)
    out = torch.empty((n, 3, image_size, image_size), dtype=torch.float32)
    for i, b in enumerate(bytes_list):
        out[i].copy_(_to_tensor(Image.open(io.BytesIO(b)).convert("RGB")))
    return out


class LanceArrowDataset(torch.utils.data.Dataset):
    def __init__(self, uri, table_name, **connect_kwargs):
        self.uri = uri
        self.table_name = table_name
        self.connect_kwargs = connect_kwargs
        self._perm = None

        db = lancedb.connect(uri, **connect_kwargs)
        self.length = len(db.open_table(table_name))

    def __len__(self):
        return self.length

    def __getstate__(self):
        # Permutation holds Rust async state — zero it so each worker reopens its own
        state = self.__dict__.copy()
        state["_perm"] = None
        return state

    def _ensure_open(self):
        if self._perm is None:
            db = lancedb.connect(self.uri, **self.connect_kwargs)
            self._perm = (
                Permutation.identity(db.open_table(self.table_name))
                .select_columns(["image_bytes", "label"])
                .with_format("arrow")
            )

    def __getitem__(self, idx):
        self._ensure_open()
        return self._perm[idx]

    def __getitems__(self, indices):
        self._ensure_open()
        return self._perm.__getitems__(indices)


def _lance_collate(batch, image_size):
    # __getitems__ with with_format("arrow") returns a pa.RecordBatch
    if isinstance(batch, pa.RecordBatch):
        bytes_list = batch["image_bytes"].to_pylist()
        labels = torch.from_numpy(batch["label"].to_numpy(zero_copy_only=False).copy()).long()
    else:
        bytes_list = [row["image_bytes"] for row in batch]
        labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)
    return _decode_jpegs(bytes_list, image_size), labels


def make_lance_loader(uri, table_name, batch_size, image_size, num_workers, prefetch_factor, **connect_kwargs):
    dataset = LanceArrowDataset(uri, table_name, **connect_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(_lance_collate, image_size=image_size),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


class Boto3S3Dataset(torch.utils.data.Dataset):
    def __init__(self, bucket, keys, access_key, secret_key, region):
        self.bucket = bucket
        self.keys = keys
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.labels = [i % 1000 for i in range(len(keys))]
        self._client = None

    def _init_client(self):
        if self._client is None:
            config = botocore.config.Config(max_pool_connections=64, retries={"max_attempts": 3})
            self._client = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            ).client("s3", region_name=self.region, config=config)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_client()
        img_bytes = self._client.get_object(Bucket=self.bucket, Key=self.keys[idx])["Body"].read()
        return _to_tensor(Image.open(io.BytesIO(img_bytes)).convert("RGB")), self.labels[idx]


def list_s3_keys(bucket, prefix, access_key, secret_key, region, max_keys=10000):
    s3 = boto3.client("s3", region_name=region,
                      aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png")):
                keys.append(obj["Key"])
                if len(keys) >= max_keys:
                    return keys
    return keys


def make_boto_loader(bucket, keys, batch_size, access_key, secret_key, region, num_workers, prefetch_factor):
    dataset = Boto3S3Dataset(bucket, keys, access_key, secret_key, region)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )


class ParquetS3Dataset(torch.utils.data.Dataset):
    def __init__(self, bucket, key, total_rows, access_key, secret_key, region):
        # PyArrow ds.dataset expects "bucket/key" (no s3:// prefix)
        self.s3_uri = f"{bucket}/{key}"
        self.total_rows = total_rows
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self._scanner = None

    def __len__(self):
        return self.total_rows

    def _init_scanner(self):
        if self._scanner is None:
            s3_fs = fs.S3FileSystem(
                region=self.region,
                access_key=self.access_key,
                secret_key=self.secret_key,
            )
            dataset = ds.dataset(self.s3_uri, format="parquet", filesystem=s3_fs)
            self._scanner = dataset.scanner(columns=["image_bytes", "label"])

    def __getitem__(self, idx):
        self._init_scanner()
        table = self._scanner.take([idx])
        return {
            "image_bytes": table.column("image_bytes")[0].as_py(),
            "label": table.column("label")[0].as_py(),
        }

    def __getitems__(self, indices):
        self._init_scanner()
        # Random-access over S3 Parquet is where it usually struggles vs LanceDB
        table = self._scanner.take(indices)
        return [
            {"image_bytes": b, "label": l}
            for b, l in zip(table.column("image_bytes").to_pylist(), table.column("label").to_pylist())
        ]


def _parquet_collate(batch, image_size):
    bytes_list = [row["image_bytes"] for row in batch]
    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)
    return _decode_jpegs(bytes_list, image_size), labels


def make_parquet_loader(bucket, key, total_rows, batch_size, image_size, access_key, secret_key, region, num_workers, prefetch_factor):
    dataset = ParquetS3Dataset(bucket, key, total_rows, access_key, secret_key, region)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(_parquet_collate, image_size=image_size),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
