import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import BinaryAccuracy

from image_segmentation.models.unet import UNet
from image_segmentation.models.dice_loss import DiceLoss

class BinarySegmentator(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        input_channels: int = 3,
        ndim: int = 2,
        num_layers: int = 5,
        features_start: int = 64,
        channels: list[int] = None,
        bilinear: bool = True,
        norm_op: str = 'batch',
        warmup_epochs: int = 5,
        dropout: float = 0.0,
        kernel_size: int = 3
    ):
        super().__init__()
        self.save_hyperparameters()

        if channels is None:
            channels = [features_start * (2**i) for i in range(num_layers)]

        self.net = UNet(
            input_channels=input_channels,
            num_classes=1,
            spatial_dims=ndim,
            channels=channels,
            bilinear=bilinear,
            norm_op=norm_op,
            dropout=dropout,
            kernel_size=kernel_size
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        self.train_metrics = nn.ModuleDict({
            "dice": DiceScore(num_classes=2, input_format="index", include_background=False),
            "accuracy": BinaryAccuracy(),
        })
        self.val_metrics = nn.ModuleDict({
            "dice": DiceScore(num_classes=2, input_format="index", include_background=False),
            "accuracy": BinaryAccuracy(),
        })
        self.test_metrics = nn.ModuleDict({
            "dice": DiceScore(num_classes=2, input_format="index", include_background=False),
            "accuracy": BinaryAccuracy(),
        })

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, stage: str):
        img, mask = batch
        img = img.float()
        mask = (mask > 0.5).float()

        # Forward pass
        logits = self(img)
        probs = torch.sigmoid(logits)

        # Compute losses (grad tracked)
        bce_loss = self.bce(logits, mask)
        dice_loss = self.dice_loss(probs, mask)
        loss = 0.5 * bce_loss + 0.5 * dice_loss

        # Compute metrics (grad NOT tracked)
        with torch.no_grad():
            preds = (probs > 0.5).long().squeeze(1) # (N, H, W) or (N, D, H, W), values {0, 1}
            targets = mask.long().squeeze(1)        # (N, H, W) or (N, D, H, W), values {0, 1}

            metrics = (self.train_metrics if stage == "train" else
                       self.val_metrics if stage == "val" else
                          self.test_metrics)
            
            for name, metric in metrics.items():
                if name == "dice":
                    metric.update(preds, targets)
                else:
                    metric.update(probs, mask)
                self.log(f"{stage}_{name}", metric, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        # Logging
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        return loss


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure = None):
        if epoch < self.hparams.warmup_epochs:
            lr_scale = float(epoch + 1) / float(self.hparams.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        optimizer.step(closure=optimizer_closure)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }