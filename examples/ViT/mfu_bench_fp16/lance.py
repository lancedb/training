import os
import io
import multiprocessing

import torch
import lancedb
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from lancedb.permutation import Permutation


MODEL_NAME = "vit_h_14"  # Options: "vit_h_14", "vit_l_16", "vit_b_16"
EXPECT_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 350

WARMUP_STEPS = 5
BENCH_STEPS = 50
NUM_WORKERS = 8  # High workers to stress IO
PREFETCH_FACTOR = 4

# H100 / H200 Peak bfloat16 Dense Compute
PEAK_FLOPS = 989e12


def calculate_dynamic_flops(model_name, img_height, img_width):
    patch_size = int(model_name.split("_")[-1])
    n_patches = (img_height // patch_size) * (img_width // patch_size)
    seq_len = n_patches + 1

    # attention: 4 * n_layers * d_model coefficients captures both QK^T and AV multiply
    # linear:   24 * n_layers * d_model^2 = (QKV=6 + out=2 + MLP_4x=16) * n_layers * d_model^2
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

TABLE_NAME = f"images_{EXPECT_IMAGE_SIZE[0]}"
S3_URI = "s3://lancedb-datasets-dev-us-east-2-devrel/training/"
ENTERPRISE_URI = "db://training"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

STORAGE_OPTIONS = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_ID                                                                                                                                           "),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_ACCESS_KEY                                                                                                                   "),
    "aws_region": AWS_REGION
}
LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY", "YOUR_LANCE_API_KEY                                                                                                                                           ")
LANCEDB_HOST = "https://ayush:YOUR_PASSWORD  @devrelaws@3-129-160-110.sslip.io"


standard_transform = transforms.Compose([
    transforms.ToTensor()
])

def decode_collate(batch):
    bytes_list = batch["image_bytes"].to_pylist()
    labels = torch.from_numpy(batch["label"].to_numpy(zero_copy_only=False).copy()).long()

    n = len(bytes_list)
    images = torch.empty((n, 3, EXPECT_IMAGE_SIZE[0], EXPECT_IMAGE_SIZE[1]), dtype=torch.float32)

    for i, b in enumerate(bytes_list):
        img = Image.open(io.BytesIO(b)).convert("RGB")
        images[i].copy_(standard_transform(img))

    return images, labels

class LanceArrowDataset(torch.utils.data.Dataset):
    def __init__(self, uri, table_name, **connect_kwargs):
        self.uri = uri
        self.table_name = table_name
        self.connect_kwargs = connect_kwargs

        db = lancedb.connect(uri, **connect_kwargs)
        self.length = len(db.open_table(table_name))
        self._perm = None

    def __len__(self): return self.length

    def __getstate__(self):
        # Permutation holds Rust async state — zero it so each worker reopens its own connection
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

def make_loader(uri, **connect_kwargs):
    dataset = LanceArrowDataset(uri, TABLE_NAME, **connect_kwargs)
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

    def print_metrics(time_sec):
        images_per_sec = (BENCH_STEPS * BATCH_SIZE) / time_sec
        achieved_flops = images_per_sec * FLOPS_PER_IMAGE
        mfu = 100.0 * achieved_flops / PEAK_FLOPS
        print(f"Time Taken:     {time_sec:.3f} sec")
        print(f"Throughput:     {images_per_sec:.2f} images/sec")
        print(f"Achieved FLOPS: {achieved_flops / 1e12:.3f} TFLOPS")
        print(f"GPU MFU:        {mfu:.2f}%")

    def profile_synthetic_baseline(tag):
        print(f"\n--- {tag} ---")
        print(f"batch_size={BATCH_SIZE}, warmup={WARMUP_STEPS}, steps={BENCH_STEPS}")
        images = torch.randn(BATCH_SIZE, 3, EXPECT_IMAGE_SIZE[0], EXPECT_IMAGE_SIZE[1],
                             device=device, dtype=torch.bfloat16)
        labels = torch.randint(0, 1000, (BATCH_SIZE,), device=device)

        model.train()
        for _ in range(WARMUP_STEPS): compiled_train_step(images, labels)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_evt.record()
        for _ in range(BENCH_STEPS): compiled_train_step(images, labels)
        end_evt.record()
        torch.cuda.synchronize()

        print_metrics(start_evt.elapsed_time(end_evt) / 1000.0)

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

    profile_synthetic_baseline(f"Synthetic Pure-GPU Baseline ({MODEL_NAME}) / {EXPECT_IMAGE_SIZE}")

    s3_loader = make_loader(S3_URI, storage_options=STORAGE_OPTIONS)
    profile_training(s3_loader, f"LanceDB OSS Training ({MODEL_NAME}) / {EXPECT_IMAGE_SIZE}")

    try:
        ent_loader = make_loader(ENTERPRISE_URI, api_key=LANCEDB_API_KEY, host_override=LANCEDB_HOST, region=AWS_REGION)
        profile_training(ent_loader, f"LanceDB Enterprise Training ({MODEL_NAME}) / {EXPECT_IMAGE_SIZE}")
    except Exception as e:
        print(f"Failed to run Enterprise benchmark: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
