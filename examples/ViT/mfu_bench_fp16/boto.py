import os
import io
import multiprocessing

import torch
import boto3
import botocore
from PIL import Image
from torchvision import transforms
import torchvision.models as models

MODEL_NAME = "vit_h_14"  # Options: "vit_h_14", "vit_l_16", "vit_b_16"
EXPECT_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 350
WARMUP_STEPS = 5
BENCH_STEPS = 50
NUM_WORKERS = 8
PREFETCH_FACTOR = 4

# H100 / H200 Peak bfloat16 Dense Compute
PEAK_FLOPS = 989e12


def calculate_dynamic_flops(model_name, img_height, img_width):
    patch_size = int(model_name.split("_")[-1])
    n_patches = (img_height // patch_size) * (img_width // patch_size)
    seq_len = n_patches + 1

    if model_name == "vit_h_14":
        attention_flops = 163840 * (seq_len ** 2)   # 4 * 32 * 1280
        linear_flops    = 1259796480 * seq_len       # ~24 * 32 * 1280^2 + patch embed
        head_flops      = 2560000                    # 2 * 1280 * 1000
    elif model_name == "vit_l_16":
        attention_flops = 98304 * (seq_len ** 2)     # 4 * 24 * 1024
        linear_flops    = 603979776 * seq_len        # 24 * 24 * 1024^2
        head_flops      = 2048000                    # 2 * 1024 * 1000
    elif model_name == "vit_b_16":
        attention_flops = 36864 * (seq_len ** 2)     # 4 * 12 * 768
        linear_flops    = 169869312 * seq_len        # 24 * 12 * 768^2
        head_flops      = 1536000                    # 2 * 768 * 1000
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    forward_flops = attention_flops + linear_flops + head_flops
    return forward_flops * 3  # FWD + 2x BWD

FLOPS_PER_IMAGE = calculate_dynamic_flops(MODEL_NAME, EXPECT_IMAGE_SIZE[0], EXPECT_IMAGE_SIZE[1])
print(f"[{MODEL_NAME} @ {EXPECT_IMAGE_SIZE[0]}x{EXPECT_IMAGE_SIZE[1]}] Calculated FLOPs/Image: {FLOPS_PER_IMAGE / 1e9:.2f} GFLOPs")

S3_BUCKET = "lancedb-datasets-dev-us-east-2-devrel"
S3_PREFIX = f"training/mfu_test_{EXPECT_IMAGE_SIZE[0]}"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_ID                                                                                                                                           "
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_ACCESS_KEY                                                                                                                   "

standard_transform = transforms.Compose([
    transforms.ToTensor()
])


class Boto3S3Dataset(torch.utils.data.Dataset):
    def __init__(self, bucket, keys, labels):
        self.bucket = bucket
        self.keys = keys
        self.labels = labels
        self.client = None

    def _init_client(self):
        if self.client is None:
            config = botocore.config.Config(
                max_pool_connections=64,
                retries={'max_attempts': 3},
            )
            session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            self.client = session.client('s3', region_name=AWS_REGION, config=config)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_client()

        response = self.client.get_object(Bucket=self.bucket, Key=self.keys[idx])
        img_bytes = response['Body'].read()

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return standard_transform(img), self.labels[idx]

def get_s3_keys(bucket, prefix, max_keys=10000):
    print(f"Fetching up to {max_keys} image keys from s3://{bucket}/{prefix}...")
    s3 = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys = []
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png')):
                keys.append(obj['Key'])
                if len(keys) >= max_keys:
                    return keys
    return keys

def make_s3_loader(bucket, keys):
    labels = [i % 1000 for i in range(len(keys))]
    dataset = Boto3S3Dataset(bucket, keys, labels)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        multiprocessing_context="spawn" if NUM_WORKERS > 0 else None
    )

def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    print(f"\nInitializing {MODEL_NAME} natively in bfloat16 and compiling...")

    model_class = getattr(models, MODEL_NAME)
    model = model_class(weights=None, image_size=EXPECT_IMAGE_SIZE[0]).to(device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    criterion = torch.nn.CrossEntropyLoss()

    @torch.compile(mode="max-autotune")
    def compiled_train_step(images, labels):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        return loss

    def fetch_next_batch(it, loader):
        try:
            return it, next(it)
        except StopIteration:
            it = iter(loader)
            return it, next(it)

    def train_exact_steps(it, loader, steps):
        model.train()
        for _ in range(steps):
            it, (images, labels) = fetch_next_batch(it, loader)
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            compiled_train_step(images, labels)
        return it

    def print_metrics(time_sec):
        images_per_sec = (BENCH_STEPS * BATCH_SIZE) / time_sec
        achieved_flops = images_per_sec * FLOPS_PER_IMAGE
        mfu = 100.0 * achieved_flops / PEAK_FLOPS
        print(f"Time Taken:     {time_sec:.3f} sec")
        print(f"Throughput:     {images_per_sec:.2f} images/sec")
        print(f"Achieved FLOPS: {achieved_flops / 1e12:.3f} TFLOPS")
        print(f"GPU MFU:        {mfu:.2f}%")

    def profile_training(loader, tag):
        print(f"\n--- {tag} ---")
        print(f"batch_size={BATCH_SIZE}, workers={NUM_WORKERS}, steps={BENCH_STEPS}")

        it = train_exact_steps(iter(loader), loader, WARMUP_STEPS)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_evt.record()
        train_exact_steps(it, loader, BENCH_STEPS)
        end_evt.record()
        torch.cuda.synchronize()

        print_metrics(start_evt.elapsed_time(end_evt) / 1000.0)

    s3_keys = get_s3_keys(S3_BUCKET, S3_PREFIX, max_keys=10000)

    if not s3_keys:
        print(f"\nERROR: No images found in s3://{S3_BUCKET}/{S3_PREFIX}")
        print("Boto3 needs a folder of loose .jpg files, not a .lance folder.")
        return

    s3_loader = make_s3_loader(S3_BUCKET, s3_keys)
    profile_training(s3_loader, f"Boto3 S3 Training ({MODEL_NAME}) / {EXPECT_IMAGE_SIZE}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
