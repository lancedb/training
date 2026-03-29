import os
import io
import multiprocessing

import torch
import pyarrow.dataset as ds
import pyarrow.fs as fs
from PIL import Image
from torchvision import transforms
import torchvision.models as models

MODEL_NAME = "vit_h_14"
EXPECT_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 350
WARMUP_STEPS = 5
BENCH_STEPS = 50
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
TOTAL_DATASET_ROWS = 10000

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
S3_PARQUET_KEY = f"training/mfu_test_parquet/images_{EXPECT_IMAGE_SIZE[0]}.parquet"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_ID                                                                                                                                           "
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_ACCESS_KEY                                                                                                                   "


standard_transform = transforms.Compose([
    transforms.ToTensor()
])


def decode_collate(batch):
    bytes_list = [row["image_bytes"] for row in batch]
    labels = torch.tensor([row["label"] for row in batch], dtype=torch.long)

    n = len(bytes_list)
    images = torch.empty((n, 3, EXPECT_IMAGE_SIZE[0], EXPECT_IMAGE_SIZE[1]), dtype=torch.float32)

    for i, b in enumerate(bytes_list):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        images[i].copy_(standard_transform(img))

    return images, labels

class ParquetS3Dataset(torch.utils.data.Dataset):
    def __init__(self, bucket, key, total_rows):
        # PyArrow ds.dataset with s3fs expects "bucket/key" (no s3:// prefix)
        self.s3_uri = f"{bucket}/{key}"
        self.total_rows = total_rows
        self.scanner = None

    def __len__(self): return self.total_rows

    def _init_client(self):
        if self.scanner is None:
            s3_fs = fs.S3FileSystem(
                region=AWS_REGION,
                access_key=AWS_ACCESS_KEY_ID,
                secret_key=AWS_SECRET_ACCESS_KEY
            )
            dataset = ds.dataset(self.s3_uri, format="parquet", filesystem=s3_fs)
            # Only pull the columns we need to save bandwidth
            self.scanner = dataset.scanner(columns=["image_bytes", "label"])

    def __getitem__(self, idx):
        self._init_client()
        table = self.scanner.take([idx])
        return {
            "image_bytes": table.column("image_bytes")[0].as_py(),
            "label": table.column("label")[0].as_py()
        }

    def __getitems__(self, indices):
        self._init_client()
        # Random-access over S3 Parquet is where it usually struggles vs LanceDB
        table = self.scanner.take(indices)
        bytes_list = table.column("image_bytes").to_pylist()
        labels_list = table.column("label").to_pylist()
        return [{"image_bytes": b, "label": l} for b, l in zip(bytes_list, labels_list)]

def make_parquet_loader(bucket, key):
    dataset = ParquetS3Dataset(bucket, key, TOTAL_DATASET_ROWS)
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        collate_fn=decode_collate, persistent_workers=(NUM_WORKERS > 0),
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
        try: return it, next(it)
        except StopIteration: return iter(loader), next(iter(loader))

    def train_exact_steps(it, loader, steps):
        model.train()
        for _ in range(steps):
            it, (images, labels) = fetch_next_batch(it, loader)
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            compiled_train_step(images, labels)
        return it

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

        time_sec = start_evt.elapsed_time(end_evt) / 1000.0
        images_per_sec = (BENCH_STEPS * BATCH_SIZE) / time_sec
        achieved_flops = images_per_sec * FLOPS_PER_IMAGE
        print(f"Time Taken:     {time_sec:.3f} sec")
        print(f"Throughput:     {images_per_sec:.2f} images/sec")
        print(f"Achieved FLOPS: {achieved_flops / 1e12:.3f} TFLOPS")
        print(f"GPU MFU:        {100.0 * achieved_flops / PEAK_FLOPS:.2f}%")

    try:
        loader = make_parquet_loader(S3_BUCKET, S3_PARQUET_KEY)
        profile_training(loader, f"S3 Parquet Training ({MODEL_NAME}) / {EXPECT_IMAGE_SIZE}")
    except Exception as e:
        print(f"Failed to run Parquet benchmark: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
