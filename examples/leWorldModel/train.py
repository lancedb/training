"""
leWorldModel trainer with LanceDB data backend.

Drop-in replacement for le-wm/train.py that swaps the HDF5 data pipeline for
a LanceDB-backed DataLoader while keeping the model, loss, and Lightning
training loop identical to the original.

Usage:
  # Local LanceDB store (defaults come from config)
  python train.py --config config/lewm_pusht.yaml

  # S3-backed store, credentials via CLI
  python train.py --config config/lewm_pusht.yaml \\
    --lance-uri s3://my-bucket/lewm \\
    --aws-region us-east-1 \\
    --aws-access-key-id AKIA... \\
    --aws-secret-access-key ...

  # S3-backed store, credentials via environment (AWS_ACCESS_KEY_ID etc.)
  python train.py --config config/lewm_pusht.yaml --lance-uri s3://my-bucket/lewm

  # Override table and columns without editing the config
  python train.py --config config/lewm_pusht.yaml \\
    --table-name lewm_reacher \\
    --columns pixels action observation
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytorch_lightning as pl
import timm
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger

from jepa import JEPA
from lewm_lance import make_train_val_loaders
from module import ARPredictor, Embedder, MLP, SIGReg
from stable_worldmodel.optim import LinearWarmupCosineAnnealingLR
from utils import ModelObjectCallBack


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LeWMLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper around the JEPA world model.

    Accepts the batch dict produced by make_train_val_loaders:
      "pixels"  : (B, T, C, H, W)  float32
      "action"  : (B, T, A)         float32
      ...additional columns...
    """

    def __init__(self, model: JEPA, sigreg: SIGReg, cfg: dict):
        super().__init__()
        self.model  = model
        self.sigreg = sigreg
        self.cfg    = cfg
        self.save_hyperparameters(ignore=["model", "sigreg"])

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        loss_pred = self.model.criterion(batch)["loss"]
        loss_reg  = self.sigreg(self.model.last_embeddings)
        loss = loss_pred + self.cfg["sigreg_weight"] * loss_reg
        self.log(f"{stage}/loss_pred", loss_pred, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/loss_reg",  loss_reg,  on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/loss",      loss,       on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: dict, _) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, _) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.cfg["warmup_epochs"],
            max_epochs=self.cfg["max_epochs"],
        )
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> tuple[JEPA, SIGReg]:
    m = cfg["model"]

    encoder = timm.create_model(
        m["encoder_name"],
        pretrained=False,
        img_size=m["image_size"],
        num_classes=0,
    )

    predictor = ARPredictor(
        embed_dim=m["embed_dim"],
        depth=m["predictor_depth"],
        num_heads=m["predictor_heads"],
        mlp_dim=m["predictor_mlp_dim"],
        max_seq_len=m["history_size"] + m["num_preds"],
    )

    action_encoder = Embedder(
        in_dim=m["action_dim"],
        out_dim=m["embed_dim"],
    )

    encoder_dim = encoder.embed_dim
    projector = MLP(encoder_dim, m["proj_hidden"], m["embed_dim"])
    pred_proj = MLP(m["embed_dim"], m["proj_hidden"], m["embed_dim"])

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        history_size=m["history_size"],
        num_preds=m["num_preds"],
    )

    sigreg = SIGReg(
        embed_dim=m["embed_dim"],
        knots=cfg["loss"]["sigreg_knots"],
        num_proj=cfg["loss"]["sigreg_num_proj"],
    )

    return model, sigreg


# ---------------------------------------------------------------------------
# S3 storage options
# ---------------------------------------------------------------------------

def build_storage_options(args: argparse.Namespace) -> dict:
    """
    Build the storage_options dict for lancedb.connect() from CLI args,
    falling back to standard AWS environment variables.

    LanceDB passes storage_options directly to the Rust object_store library,
    which accepts these keys for S3:
      aws_access_key_id, aws_secret_access_key, aws_session_token,
      region, endpoint_url, aws_virtual_hosted_style_request

    Environment variable fallbacks follow the standard AWS SDK convention:
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
      AWS_DEFAULT_REGION, AWS_ENDPOINT_URL

    Returns an empty dict for local URIs (no storage_options needed).
    """
    uri = args.lance_uri
    if not (uri.startswith("s3://") or uri.startswith("gs://") or uri.startswith("az://")):
        return {}

    opts: dict[str, str] = {}

    access_key = args.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = args.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = args.aws_session_token or os.environ.get("AWS_SESSION_TOKEN")
    region = args.aws_region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    endpoint = args.s3_endpoint or os.environ.get("AWS_ENDPOINT_URL")

    if access_key:
        opts["aws_access_key_id"] = access_key
    if secret_key:
        opts["aws_secret_access_key"] = secret_key
    if session_token:
        opts["aws_session_token"] = session_token
    if region:
        opts["region"] = region
    if endpoint:
        opts["endpoint_url"] = endpoint
        opts["aws_virtual_hosted_style_request"] = "false"

    return opts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train leWorldModel with LanceDB data backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config/lewm_pusht.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--lance-uri",
        default=None,
        help="LanceDB URI. Defaults to data.lance_uri in config. "
             "Use s3://bucket/prefix for cloud storage.",
    )
    parser.add_argument(
        "--table-name",
        default=None,
        help="LanceDB table name. Defaults to data.table_name in config.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Columns to load. Defaults to data.columns in config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="WandB run name. Defaults to <table_name>-<T>steps.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging.",
    )

    s3 = parser.add_argument_group(
        "S3 storage options",
        "Credentials for S3-backed LanceDB tables. All args fall back to "
        "standard AWS environment variables (AWS_ACCESS_KEY_ID, etc.).",
    )
    s3.add_argument("--aws-access-key-id",     default=None, metavar="KEY")
    s3.add_argument("--aws-secret-access-key", default=None, metavar="SECRET")
    s3.add_argument("--aws-session-token",     default=None, metavar="TOKEN",
                    help="Temporary session token (STS / IAM role assumed credentials).")
    s3.add_argument("--aws-region",            default=None, metavar="REGION",
                    help="AWS region, e.g. us-east-1.")
    s3.add_argument("--s3-endpoint",           default=None, metavar="URL",
                    help="Custom S3-compatible endpoint (MinIO, R2, etc.).")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg  = cfg.get("loss", {})

    # Resolve: CLI arg > config value > hardcoded fallback
    lance_uri  = args.lance_uri  or data_cfg.get("lance_uri",  "./lewm_lance")
    table_name = args.table_name or data_cfg.get("table_name")
    columns    = args.columns    or data_cfg["columns"]
    num_steps  = model_cfg["history_size"] + model_cfg["num_preds"]

    if table_name is None:
        parser.error("table_name is required: set data.table_name in config or pass --table-name")

    storage_options = build_storage_options(args)
    if storage_options:
        print(f"  S3 storage: region={storage_options.get('region', 'env')}"
              + (f"  endpoint={storage_options['endpoint_url']}" if "endpoint_url" in storage_options else ""))

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    # storage_options is only passed for cloud URIs; empty dict for local paths
    connect_kwargs = {"storage_options": storage_options} if storage_options else {}

    print(f"Building DataLoaders  ({lance_uri} / {table_name})...")
    train_loader, val_loader = make_train_val_loaders(
        uri=lance_uri,
        table_name=table_name,
        columns=columns,
        batch_size=train_cfg["batch_size"],
        num_steps=num_steps,
        img_size=model_cfg["image_size"],
        num_workers=data_cfg["num_workers"],
        prefetch_factor=data_cfg["prefetch_factor"],
        val_fraction=data_cfg["val_fraction"],
        seed=data_cfg["seed"],
        **connect_kwargs,
    )
    print(f"  Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print("Building model...")
    model_cfg["action_dim"] = _infer_action_dim(train_loader)
    model, sigreg = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params / 1e6:.1f}M")

    lightning_model = LeWMLightning(
        model=model,
        sigreg=sigreg,
        cfg={**train_cfg, **loss_cfg},
    )

    # ------------------------------------------------------------------ #
    # Logging & callbacks
    # ------------------------------------------------------------------ #
    run_name = args.run_name or f"{table_name}-{num_steps}T"

    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=cfg.get("wandb_project", "lewm-lancedb"),
            name=run_name,
            config={**cfg, "lance_uri": lance_uri, "table": table_name},
        )

    ckpt_dir = train_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelObjectCallBack(
            dirpath=ckpt_dir,
            filename=f"{table_name}_lewm",
            epoch_interval=train_cfg["save_every_n_epochs"],
        )
    ]

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        precision=train_cfg["precision"],
        gradient_clip_val=train_cfg["gradient_clip"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=train_cfg["log_every_n_steps"],
        enable_progress_bar=True,
    )

    print("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    print("Training complete.")


def _infer_action_dim(loader: torch.utils.data.DataLoader) -> int:
    batch = next(iter(loader))
    return batch["action"].shape[-1]


if __name__ == "__main__":
    main()
