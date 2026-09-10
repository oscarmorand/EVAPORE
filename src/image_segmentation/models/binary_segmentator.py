import logging

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from tqdm import tqdm
import gc
import os
import torchio as tio

from image_segmentation.models.unet import UNet
from image_segmentation.models.dice_loss import DiceLoss
from image_segmentation.data.io_utils import save_array

logger = logging.getLogger(__name__)


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
        kernel_size: int = 3,
        dice_loss_ratio: float = 0.5,
        val_patch_size: int = 64,
        val_patch_overlap: int = 16,
        val_sw_batch_size: int = 4
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

        self.register_buffer("use_sliding_window", torch.tensor(False), persistent=True)

        self.train_metrics = nn.ModuleDict({
            "dice": DiceScore(num_classes=2, input_format="index", include_background=False),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        })
        self.val_metrics = nn.ModuleDict({
            "dice": DiceScore(num_classes=2, input_format="index", include_background=False),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        })
        self.test_metrics = nn.ModuleDict({
            "dice": DiceScore(num_classes=2, input_format="index", include_background=False),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        })

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward compatibility for checkpoints saved before `use_sliding_window` was added as a persistent buffer.
        """
        if "use_sliding_window" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["use_sliding_window"] = self.use_sliding_window.clone()
            logger.info(
                "Checkpoint predates the sliding-window fallback buffer; "
                "defaulting use_sliding_window=False"
            )
        return super().load_state_dict(state_dict, strict=strict)


    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, stage: str):
        img, mask = batch
        img = img.float()
        mask = (mask > 0.5).float()

        if stage == "train":
            logits = self(img)
        else:
            logits = self._predict(img)

        probs = torch.sigmoid(logits)

        if self.hparams.dice_loss_ratio >= 1.0:
            loss = self.dice_loss(probs, mask)
        elif self.hparams.dice_loss_ratio <= 0.0:
            loss = self.bce(logits, mask)
        else:
            bce_loss = self.bce(logits, mask)
            dice_loss = self.dice_loss(probs, mask)
            loss = self.hparams.dice_loss_ratio * dice_loss + (1.0 - self.hparams.dice_loss_ratio) * bce_loss

        with torch.no_grad():
            preds = (probs > 0.5).long().squeeze(1)
            targets = mask.long().squeeze(1)

            metrics = (self.train_metrics if stage == "train" else
                    self.val_metrics if stage == "val" else
                        self.test_metrics)

            for name, metric in metrics.items():
                if name == "dice":
                    metric.update(preds, targets)
                else:
                    metric.update(probs, mask)
                self.log(f"{stage}_{name}", metric, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def _predict(self, img: torch.Tensor) -> torch.Tensor:
        """
        Full-image forward pass when it fits in memory, sliding-window
        inference otherwise. The very first time a full-image pass OOMs,
        `use_sliding_window` flips permanently (for the rest of the run,
        and across resumes since it's a persistent buffer) so later calls
        skip straight to sliding-window instead of re-attempting and
        re-failing the full pass every step.
        """
        if not self.use_sliding_window:
            try:
                return self(img)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if img.is_cuda:
                    torch.cuda.empty_cache()
                gc.collect()
                self.use_sliding_window.fill_(True)
                logger.warning(
                    "Full-image forward pass ran out of memory (shape=%s); "
                    "switching to sliding-window inference for the rest of the run.",
                    tuple(img.shape),
                )

        return self._sliding_window_predict(
            img, self.hparams.val_patch_size, self.hparams.val_patch_overlap, self.hparams.val_sw_batch_size
        )

    def _sliding_window_predict(self, img: torch.Tensor, patch_size=64, overlap=16, sw_batch_size=4):
        """img: (1, C, D, H, W) tensor already on device. Returns logits (1, num_classes, D, H, W)."""
        patch_size = tuple([patch_size] * self.hparams.ndim)
        overlap = tuple([overlap] * self.hparams.ndim)
        subject = tio.Subject(image=tio.ScalarImage(tensor=img.squeeze(0).cpu()))
        grid_sampler = tio.inference.GridSampler(subject, patch_size, overlap)
        aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=sw_batch_size)

        with torch.no_grad():
            for patches_batch in patch_loader:
                patch_imgs = patches_batch["image"][tio.DATA].to(self.device)
                locations = patches_batch[tio.LOCATION]
                logits = self(patch_imgs)
                aggregator.add_batch(logits.cpu(), locations)

        return aggregator.get_output_tensor().unsqueeze(0).to(self.device)


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


    def inference_on_dataset(self, 
                  dataset, 
                  pred_dir,
                  device
    ) -> None:
        with torch.no_grad():
            for i, img_path in enumerate(tqdm(dataset.img_paths)):
                img, mask = dataset[i]
                print(img.shape)
                img = img.unsqueeze(0).float().to(device)
                print(img.shape)

                logits = self(img)
                print(logits.shape)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                print(probs.shape)

                pred = (probs > 0.5).astype('uint8') * 255
                print(pred.shape)

                base_name = os.path.basename(img_path)
                pred_path = os.path.join(pred_dir, base_name)
                
                save_array(pred, pred_path)

                del logits, probs, pred, img, mask

        torch.cuda.empty_cache()
        gc.collect()