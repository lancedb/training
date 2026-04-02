"""
leWorldModel trainer with LanceDB data backend.

Drop-in replacement for le-wm/train.py that swaps the HDF5 data pipeline for
a LanceDB-backed DataLoader while keeping the model, loss, and Lightning
training loop identical to the original.

jepa.py and module.py are vendored directly from https://github.com/lucas-maes/le-wm
(no git clone required).

Usage:
  # Local LanceDB store (defaults come from config)
  python train.py --config config/lewm_pusht.yaml

  # S3-backed store, credentials via CLI
  python train.py --config config/lewm_pusht.yaml \\
    --lance-uri s3://my-bucket/lewm \\
    --aws-region us-east-1 \\
    --aws-access-key-id AKIA... \\
    --aws-secret-access-key ...

  # Override table and columns without editing the config
  python train.py --config config/lewm_pusht.yaml \\
    --table-name lewm_reacher \\
    --columns pixels action observation
"""

import argparse
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
import yaml
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from jepa import JEPA
from lewm_loader import make_train_val_loaders
from module import ARPredictor, Embedder, MLP, SIGReg


# ---------------------------------------------------------------------------
# Encoder
#
# le-wm uses spt.backbone.utils.vit_hf() from stable_pretraining, which
# creates a HuggingFace ViTModel.  JEPA.encode() expects exactly that
# interface: output.last_hidden_state[:, 0] → CLS token.
# We build the same model directly with transformers.ViTConfig/ViTModel,
# matching the ViT-tiny spec from the le-wm paper.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Checkpoint callback (inlined to avoid stable_pretraining dependency)
# ---------------------------------------------------------------------------

class ModelObjectCallBack(Callback):
    """Save the raw model object (torch.save) at the end of every epoch_interval epochs."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            path = self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt"
            torch.save(pl_module.model, path)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LeWMLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper around the LeWM JEPA world model.

    Training loss (mirrors lejepa_forward in the original le-wm/train.py):
      1. Encode the pixel+action sequence into embedding space.
      2. Predict next-state embeddings autoregressively from the context window.
      3. Prediction loss: MSE between predicted and actual next embeddings.
      4. SIGReg loss: Sketch Isotropic Gaussian Regularizer — keeps the latent
         distribution well-shaped, preventing collapse and mode-dropping.

    Note: JEPA.criterion() is the MPC planning cost (comparing predicted rollouts
    to a goal embedding during evaluation). It is NOT the training loss and is
    never called here.
    """

    def __init__(self, model: JEPA, sigreg: SIGReg, cfg: dict):
        super().__init__()
        self.model  = model
        self.sigreg = sigreg
        self.cfg    = cfg
        self.save_hyperparameters(ignore=["model", "sigreg"])

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        ctx_len = self.cfg["history_size"]
        n_preds = self.cfg["num_preds"]

        # NaN occurs at sequence boundaries (padding); zero it out
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        # Encode pixels and actions → (B, T, embed_dim) each
        output  = self.model.encode(batch)
        emb     = output["emb"]      # (B, T, D)
        act_emb = output["act_emb"]  # (B, T, D)

        # Predict next states from the context window (first history_size frames)
        ctx_emb  = emb[:, :ctx_len]
        ctx_act  = act_emb[:, :ctx_len]
        tgt_emb  = emb[:, n_preds:]          # ground-truth targets (shifted by n_preds)
        pred_emb = self.model.predict(ctx_emb, ctx_act)

        # SIGReg expects (T, B, D) — transpose time and batch dims
        loss_pred = (pred_emb - tgt_emb).pow(2).mean()
        loss_reg  = self.sigreg(emb.transpose(0, 1))
        loss      = loss_pred + self.cfg["sigreg_weight"] * loss_reg

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
        # Replicate le-wm's LinearWarmupCosineAnnealingLR exactly:
        # warmup_steps = 1% of total steps (step-based, not epoch-based)
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.01 * total_steps))
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.0 + 1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(total_steps - warmup_steps, 1), eta_min=0
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
        )
        return [opt], [{"scheduler": sched, "interval": "step"}]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: dict, effective_act_dim: int) -> tuple[JEPA, SIGReg]:
    """
    Build the LeWM JEPA model from config.

    effective_act_dim = frameskip × raw_action_dim.
    With the default frameskip=5, five consecutive raw action steps are stacked
    into one frame-level action vector, so the Embedder sees a larger input.
    """
    wm   = cfg["wm"]
    pred = cfg["predictor"]

    # ViT-tiny — identical to spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224,
    #   pretrained=False, use_mask_token=False) from stable_pretraining.
    # vit_hf builds ViTModel(ViTConfig(**size_configs["tiny"]), add_pooling_layer=False,
    #   use_mask_token=False) where size_configs["tiny"] = {hidden_size:192, num_hidden_layers:12,
    #   num_attention_heads:3, intermediate_size:768}.
    vit_cfg = ViTConfig(
        hidden_size=wm["embed_dim"],        # 192
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=wm["embed_dim"] * 4,  # 768
        image_size=cfg["img_size"],         # 224
        patch_size=wm["patch_size"],        # 14
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    encoder    = ViTModel(vit_cfg, add_pooling_layer=False, use_mask_token=False)
    hidden_dim = wm["embed_dim"]   # ViT-tiny hidden_size: 192

    predictor = ARPredictor(
        num_frames=wm["history_size"],
        input_dim=wm["embed_dim"],
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=pred["depth"],
        heads=pred["heads"],
        mlp_dim=pred["mlp_dim"],
        dim_head=pred["dim_head"],
        dropout=pred["dropout"],
        emb_dropout=pred["emb_dropout"],
    )

    action_encoder = Embedder(
        input_dim=effective_act_dim,
        emb_dim=wm["embed_dim"],
    )

    # MLP(input_dim, hidden_dim, output_dim, ...) — norm_fn=BatchNorm1d matches le-wm defaults
    projector = MLP(hidden_dim,         wm["proj_hidden"], wm["embed_dim"], norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(wm["embed_dim"],    wm["proj_hidden"], wm["embed_dim"], norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )

    # SIGReg: knots and num_proj only — no embed_dim
    sigreg_cfg = cfg["loss"]["sigreg"]
    sigreg = SIGReg(
        knots=sigreg_cfg["kwargs"]["knots"],
        num_proj=sigreg_cfg["kwargs"]["num_proj"],
    )

    return model, sigreg


# ---------------------------------------------------------------------------
# S3 storage options
# ---------------------------------------------------------------------------

def build_storage_options(args: argparse.Namespace, uri: str) -> dict:
    if not (uri.startswith("s3://") or uri.startswith("gs://") or uri.startswith("az://")):
        return {}
    opts: dict[str, str] = {}
    access_key    = args.aws_access_key_id     or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key    = args.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    session_token = args.aws_session_token     or os.environ.get("AWS_SESSION_TOKEN")
    region        = args.aws_region            or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    endpoint      = args.s3_endpoint           or os.environ.get("AWS_ENDPOINT_URL")
    if access_key:    opts["aws_access_key_id"]     = access_key
    if secret_key:    opts["aws_secret_access_key"] = secret_key
    if session_token: opts["aws_session_token"]     = session_token
    if region:        opts["region"]                = region
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
    parser.add_argument("--config",     default="config/lewm_pusht.yaml")
    parser.add_argument("--lance-uri",  default=None)
    parser.add_argument("--table-name", default=None)
    parser.add_argument("--columns",    nargs="+", default=None)
    parser.add_argument("--run-name",      default=None)
    parser.add_argument("--no-wandb",      action="store_true")
    parser.add_argument("--fast-dev-run",  action="store_true",
                        help="Run 1 train+val batch then exit (smoke test)")
    parser.add_argument("--precision",     default=None,
                        help="Override trainer.precision (e.g. 32, 16-mixed, bf16-mixed)")
    s3 = parser.add_argument_group("S3 storage options")
    s3.add_argument("--aws-access-key-id",     default=None, metavar="KEY")
    s3.add_argument("--aws-secret-access-key", default=None, metavar="SECRET")
    s3.add_argument("--aws-session-token",     default=None, metavar="TOKEN")
    s3.add_argument("--aws-region",            default=None, metavar="REGION")
    s3.add_argument("--s3-endpoint",           default=None, metavar="URL")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg    = cfg["data"]
    loader_cfg  = cfg["loader"]
    trainer_cfg = cfg["trainer"]
    opt_cfg     = cfg["optimizer"]
    wm_cfg      = cfg["wm"]

    lance_uri  = args.lance_uri  or data_cfg.get("lance_uri",  "./lewm_lance")
    table_name = args.table_name or data_cfg.get("table_name")
    columns    = args.columns    or data_cfg["columns"]
    num_steps  = wm_cfg["history_size"] + wm_cfg["num_preds"]
    frameskip  = data_cfg.get("frameskip", 1)

    if table_name is None:
        parser.error("table_name required: set data.table_name in config or pass --table-name")

    storage_options = build_storage_options(args, lance_uri)
    connect_kwargs  = {"storage_options": storage_options} if storage_options else {}

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    print(f"Building DataLoaders  ({lance_uri} / {table_name})...")
    train_loader, val_loader = make_train_val_loaders(
        uri=lance_uri,
        table_name=table_name,
        columns=columns,
        batch_size=loader_cfg["batch_size"],
        num_steps=num_steps,
        frameskip=frameskip,
        img_size=cfg["img_size"],
        num_workers=loader_cfg["num_workers"],
        prefetch_factor=loader_cfg["prefetch_factor"],
        val_fraction=data_cfg["val_fraction"],
        seed=cfg["seed"],
        **connect_kwargs,
    )
    print(f"  Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    # Infer effective_act_dim from the first batch (action shape: B, T, eff_dim)
    sample_batch      = next(iter(train_loader))
    effective_act_dim = sample_batch["action"].shape[-1]
    print(f"  effective_act_dim={effective_act_dim}  "
          f"(frameskip={frameskip} × raw_action_dim={effective_act_dim // max(frameskip,1)})")

    print("Building model...")
    model, sigreg = build_model(cfg, effective_act_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params / 1e6:.1f}M")

    # Flat dict passed into LeWMLightning for optimizer/scheduler and loss
    lightning_cfg = {
        "lr":             opt_cfg["lr"],
        "weight_decay":   opt_cfg["weight_decay"],
        "max_epochs":     trainer_cfg["max_epochs"],
        "sigreg_weight":  cfg["loss"]["sigreg"]["weight"],
        "history_size":   wm_cfg["history_size"],
        "num_preds":      wm_cfg["num_preds"],
    }

    lightning_model = LeWMLightning(model=model, sigreg=sigreg, cfg=lightning_cfg)

    # ------------------------------------------------------------------ #
    # Logging & callbacks
    # ------------------------------------------------------------------ #
    run_name = args.run_name or f"{table_name}-{num_steps}T"
    logger   = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=cfg.get("wandb_project", "lewm-lancedb"),
            name=run_name,
            config={**cfg, "lance_uri": lance_uri, "table": table_name},
        )

    ckpt_dir = trainer_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        ModelObjectCallBack(
            dirpath=ckpt_dir,
            filename=f"{table_name}_lewm",
            epoch_interval=trainer_cfg["save_every_n_epochs"],
        )
    ]

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    precision = args.precision or trainer_cfg["precision"]
    trainer = pl.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        precision=precision,
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        num_sanity_val_steps=1,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    print("Starting training...")
    trainer.fit(lightning_model, train_loader, val_loader)
    print("Training complete.")


if __name__ == "__main__":
    main()
