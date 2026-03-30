import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models.vision_transformer import VisionTransformer

from dataloaders import list_s3_keys, make_lance_loader, make_boto_loader, make_parquet_loader


# Architecture — uncomment one:
ARCH = dict(n_layers=32, d_model=2048, n_heads=32, patch_size=16, mlp_dim=8192)  # ~1.6B custom
# ARCH = "vit_h_14"   # 632M params, patch=14, seq=257
# ARCH = "vit_l_16"   # 307M params, patch=16, seq=197
# ARCH = "vit_b_16"   #  86M params, patch=16, seq=197

# Backend for dict architectures (ignored for named presets):
#   "custom"      — fused QKV, bias=False, direct F.scaled_dot_product_attention
#   "torchvision" — torchvision VisionTransformer
MODEL_BACKEND = "custom"
# MODEL_BACKEND = "torchvision"

EXPECT_IMAGE_SIZE = 224
BATCH_SIZE        = 256   # 350 works for named presets; 256 is the safe limit for the ~1.6B custom arch
WARMUP_STEPS      = 5
BENCH_STEPS       = 50
NUM_WORKERS       = 8
PREFETCH_FACTOR   = 4

# H100 / H200 peak bfloat16 dense compute
PEAK_FLOPS = 989e12

AWS_ACCESS_KEY_ID     = "YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION            = "us-east-2"

S3_BUCKET      = "YOUR_S3_BUCKET"
S3_JPEG_PREFIX = f"training/mfu_test_{EXPECT_IMAGE_SIZE}"
S3_PARQUET_KEY = f"training/mfu_test_parquet/images_{EXPECT_IMAGE_SIZE}.parquet"
PARQUET_ROWS   = 10000

LANCE_S3_URI      = f"s3://{S3_BUCKET}/training/"
LANCE_ENT_URI     = "db://training"
LANCE_TABLE_NAME  = f"images_{EXPECT_IMAGE_SIZE}"
LANCE_API_KEY     = "YOUR_LANCE_API_KEY"
LANCE_ENT_HOST    = "YOUR_LANCE_ENT_HOST"
LANCE_STORAGE_OPTIONS = {
    "aws_access_key_id":     AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_region":            AWS_REGION,
}


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.norm2   = nn.LayerNorm(d_model)
        self.fc1     = nn.Linear(d_model, mlp_dim, bias=False)
        self.fc2     = nn.Linear(mlp_dim, d_model, bias=False)
        self.n_heads = n_heads

    def forward(self, x):
        B, S, D = x.shape
        H = self.n_heads
        qkv = self.qkv(self.norm1(x)).reshape(B, S, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, D))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class CustomViT(nn.Module):
    def __init__(self, image_size, patch_size, d_model, n_heads, mlp_dim, n_layers, num_classes=1000):
        super().__init__()
        n_patches        = (image_size // patch_size) ** 2
        self.patch_size  = patch_size
        self.patch_embed = nn.Linear(3 * patch_size ** 2, d_model, bias=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.blocks      = nn.ModuleList([EncoderBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)])
        self.norm        = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * P * P)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


MODEL_PRESETS = {
    "vit_h_14": dict(n_layers=32, d_model=1280, n_heads=16, patch_size=14, mlp_dim=5120),
    "vit_l_16": dict(n_layers=24, d_model=1024, n_heads=16, patch_size=16, mlp_dim=4096),
    "vit_b_16": dict(n_layers=12, d_model=768,  n_heads=12, patch_size=16, mlp_dim=3072),
}

def _arch_cfg():
    return MODEL_PRESETS[ARCH] if isinstance(ARCH, str) else ARCH

def _arch_label():
    if isinstance(ARCH, str):
        return ARCH
    cfg = _arch_cfg()
    suffix = f"_{MODEL_BACKEND}" if MODEL_BACKEND != "custom" else ""
    return f"custom_{cfg['d_model']}d_{cfg['n_layers']}L{suffix}"

def calculate_flops(arch, image_size):
    n, d, p, m = arch["n_layers"], arch["d_model"], arch["patch_size"], arch["mlp_dim"]
    seq_len = (image_size // p) ** 2 + 1
    # attention: QK^T + AV  → 4 * n * d * seq^2
    # linear:   QKV+out+MLP → (8d + 4m) * n * d * seq
    # head:                  → 2 * d * 1000
    forward = (
        4 * n * d * seq_len ** 2
        + (8 * d + 4 * m) * n * d * seq_len
        + 2 * d * 1000
    )
    return forward * 3  # FWD + 2x BWD

def build_model(image_size, device):
    arch = _arch_cfg()
    if isinstance(ARCH, str):
        model = getattr(tv_models, ARCH)(weights=None, image_size=image_size)
    elif MODEL_BACKEND == "custom":
        model = CustomViT(
            image_size=image_size,
            patch_size=arch["patch_size"],
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            mlp_dim=arch["mlp_dim"],
            n_layers=arch["n_layers"],
        )
    else:
        model = VisionTransformer(
            image_size=image_size,
            patch_size=arch["patch_size"],
            num_layers=arch["n_layers"],
            num_heads=arch["n_heads"],
            hidden_dim=arch["d_model"],
            mlp_dim=arch["mlp_dim"],
            num_classes=1000,
        )
    return model.to(device, dtype=torch.bfloat16)


def main():
    arch            = _arch_cfg()
    label           = _arch_label()
    seq_len         = (EXPECT_IMAGE_SIZE // arch["patch_size"]) ** 2 + 1
    flops_per_image = calculate_flops(arch, EXPECT_IMAGE_SIZE)

    print(f"\n[{label} @ {EXPECT_IMAGE_SIZE}x{EXPECT_IMAGE_SIZE}]")
    print(f"  arch:           {arch['n_layers']}L x d{arch['d_model']} x h{arch['n_heads']} x p{arch['patch_size']}  mlp={arch['mlp_dim']}")
    print(f"  backend:        {MODEL_BACKEND if not isinstance(ARCH, str) else 'torchvision (preset)'}")
    print(f"  seq_len:        {seq_len}")
    print(f"  batch×seq (M):  {BATCH_SIZE * seq_len:,}")
    print(f"  FLOPs/image:    {flops_per_image / 1e12:.3f} TFLOPs")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    print(f"\nInitializing {label} in bfloat16...")
    model = build_model(EXPECT_IMAGE_SIZE, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters:     {n_params / 1e9:.2f}B")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    criterion = nn.CrossEntropyLoss()

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
        images_per_sec  = (BENCH_STEPS * BATCH_SIZE) / time_sec
        achieved_flops  = images_per_sec * flops_per_image
        print(f"Time Taken:     {time_sec:.3f} sec")
        print(f"Throughput:     {images_per_sec:.2f} images/sec")
        print(f"Achieved FLOPS: {achieved_flops / 1e12:.3f} TFLOPS")
        print(f"GPU MFU:        {100.0 * achieved_flops / PEAK_FLOPS:.2f}%")

    def profile_synthetic_baseline():
        tag = f"Synthetic Pure-GPU Baseline ({label}) / {EXPECT_IMAGE_SIZE}"
        print(f"\n--- {tag} ---")
        print(f"batch_size={BATCH_SIZE}, warmup={WARMUP_STEPS}, steps={BENCH_STEPS}")

        images = torch.randn(BATCH_SIZE, 3, EXPECT_IMAGE_SIZE, EXPECT_IMAGE_SIZE,
                             device=device, dtype=torch.bfloat16)
        labels = torch.randint(0, 1000, (BATCH_SIZE,), device=device)
        model.train()

        print("  warming up (first step includes compilation)...")
        for step in range(WARMUP_STEPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record(); compiled_train_step(images, labels); e.record()
            torch.cuda.synchronize()
            print(f"  warmup {step + 1}/{WARMUP_STEPS}: {s.elapsed_time(e) / 1000:.2f}s")

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
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
        end_evt   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        train_exact_steps(it, loader, BENCH_STEPS)
        end_evt.record()
        torch.cuda.synchronize()
        print_metrics(start_evt.elapsed_time(end_evt) / 1000.0)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE, image_size=EXPECT_IMAGE_SIZE,
        num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    )

    profile_synthetic_baseline()

    s3_loader = make_lance_loader(
        LANCE_S3_URI, LANCE_TABLE_NAME,
        **loader_kwargs,
        storage_options=LANCE_STORAGE_OPTIONS,
    )
    profile_training(s3_loader, f"LanceDB OSS ({label}) / {EXPECT_IMAGE_SIZE}")

    try:
        ent_loader = make_lance_loader(
            LANCE_ENT_URI, LANCE_TABLE_NAME,
            **loader_kwargs,
            api_key=LANCE_API_KEY, host_override=LANCE_ENT_HOST, region=AWS_REGION,
        )
        profile_training(ent_loader, f"LanceDB Enterprise ({label}) / {EXPECT_IMAGE_SIZE}")
    except Exception as e:
        print(f"LanceDB Enterprise unavailable: {e}")

    s3_keys = list_s3_keys(S3_BUCKET, S3_JPEG_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    if not s3_keys:
        print(f"\nERROR: No images found at s3://{S3_BUCKET}/{S3_JPEG_PREFIX}")
        print("Boto3 benchmark requires loose .jpg files, not a .lance folder.")
    else:
        boto_loader = make_boto_loader(
            S3_BUCKET, s3_keys,
            batch_size=BATCH_SIZE,
            access_key=AWS_ACCESS_KEY_ID, secret_key=AWS_SECRET_ACCESS_KEY, region=AWS_REGION,
            num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
        )
        profile_training(boto_loader, f"Boto3 S3 ({label}) / {EXPECT_IMAGE_SIZE}")

    parquet_loader = make_parquet_loader(
        S3_BUCKET, S3_PARQUET_KEY, PARQUET_ROWS,
        access_key=AWS_ACCESS_KEY_ID, secret_key=AWS_SECRET_ACCESS_KEY, region=AWS_REGION,
        **loader_kwargs,
    )
    profile_training(parquet_loader, f"S3 Parquet ({label}) / {EXPECT_IMAGE_SIZE}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
