# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%
from graph_neural_networks.models.binary_segmentator import BinarySegmentator
from graph_neural_networks.data.datamodules.image_datamodule import ImageDatamodule
from graph_neural_networks.data.dataset.image_dataset import ImageDataset

# %%
train_transforms = A.Compose([
    A.CropNonEmptyMaskIfExists(height=512, width=512, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

dataset = ImageDataset("/home/morand/afs/EVAPORE/data/FIVES/")

datamodule = ImageDatamodule(
    dataset=dataset,
    val_split=0.2,
    test_split=0.1,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    num_workers=16,
    train_batch_size=8,
    val_batch_size=1,
    seed=42
)

# %%
from pytorch_lightning import Trainer
import torch
import pytorch_lightning as pl

callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}"
    ),
    pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.00,
        verbose=True,
        mode="min"
    )
]

trainer = Trainer(accelerator='gpu', devices="auto", max_epochs=100, precision='16-mixed', callbacks=callbacks)
torch.set_float32_matmul_precision("medium")

# %%
import pytorch_lightning as pl
import torch.nn as nn
import torch
from graph_neural_networks.models.path_encoders.path_encoder import PathEncoder
import logging
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryPrecision

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ReducedPipelineModel(pl.LightningModule):
    def __init__(
        self,
        features_generator: nn.Module,
        path_encoder: PathEncoder,
        path_classifier: nn.Module,
        edge_classification_loss_fn: nn.Module,
        lr = 1e-3,
    ):
        super().__init__()
        self.features_generator = features_generator
        self.path_encoder = path_encoder
        self.path_classifier = path_classifier

        self.edge_classification_loss_fn = edge_classification_loss_fn

        self.lr = lr

        # Metrics
        self.train_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
        })
        self.val_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
        })
        self.test_metrics = nn.ModuleDict({
            "accuracy": BinaryAccuracy(),
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision(),
        })
    
    def forward(self, 
                img: torch.Tensor,
                paths: list[torch.Tensor]
    ) -> torch.Tensor:
    
        # Compute the feature maps
        feature_maps = self.features_generator(img) # shape (1, channels, H, W)
        logger.debug(f"Feature maps: {feature_maps.shape}")

        # Sample features along paths
        edges_pred = []
        for i, path in enumerate(paths):
            # Back to GPU (if GPU is used)
            path = path.type(torch.long).squeeze(dim=0)  # shape (path_length, 2)
            logger.debug(f"Path {i}: {path.shape}")

            path_features = feature_maps[:, :, path[:, 0], path[:, 1]]  # shape (1, channels, path_length)
            inv_path_features = torch.flip(path_features, dims=[2])  # shape (1, channels, path_length)
            logger.debug(f"Path {i} features: {path_features.shape}")
        
            # Encode path
            encoded_path = self.path_encoder(path_features) # shape (1, out_channels)
            inv_encoded_path = self.path_encoder(inv_path_features) # shape (1, out_channels)
            logger.debug(f"Encoded path {i}: {encoded_path.shape}")
        
            # Classify path
            path_class = self.path_classifier(encoded_path)
            inv_path_class = self.path_classifier(inv_encoded_path)
            logger.debug(f"Path {i} class: {path_class.shape}")

            final_path_class = (path_class + inv_path_class) / 2.0

            edges_pred.append(final_path_class)
            
        edges_pred = torch.cat(edges_pred, dim=1).squeeze()  # shape (num_edges,)
        logger.debug(f"Edges predictions: {edges_pred.shape}")

        return edges_pred, feature_maps

    def _shared_step(self, 
                     batch: tuple[torch.Tensor, list[list[tuple[int, int]]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        img, paths, edges_classes, _, _, _, _ = batch
        edges_pred, _ = self.forward(img, paths)
        edges_classes = edges_classes.squeeze().type(torch.float)

        loss = self.edge_classification_loss_fn(edges_pred, edges_classes)

        return loss, edges_pred, edges_classes

    def training_step(self, batch, batch_idx):
        loss, edges_pred, edges_classes = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for name, metric in self.train_metrics.items():
            metric(edges_pred, edges_classes.int())
            self.log(f"train_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, edges_pred, edges_classes = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for name, metric in self.val_metrics.items():
            metric(edges_pred, edges_classes.int())
            self.log(f"val_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        loss, edges_pred, edges_classes = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for name, metric in self.test_metrics.items():
            metric(edges_pred, edges_classes.int())
            self.log(f"test_{name}", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        img, _, _, edges, paths, _, _ = batch
        edges_scores, _ = self.forward(img, paths)
        return edges, edges_scores

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# %%
from graph_neural_networks.models.unet import UNet
from graph_neural_networks.models.path_encoders.conv_max_path_encoder import ConvMaxPathEncoder
from graph_neural_networks.models.path_classifiers.fcn_classifier import FCNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

classic_unet = BinarySegmentator.load_from_checkpoint("/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_35_train_unet_seg/checkpoints/best-checkpoint-epoch=19-val_loss=0.1323.ckpt", map_location=device)

features_generator_to_encoder = 32
encoder_to_classifier = 256
features_generator = UNet(input_channels=3, num_classes=features_generator_to_encoder, num_layers=4, features_start=32, bilinear=False)
path_encoder = ConvMaxPathEncoder(in_channels=features_generator_to_encoder, hidden_features=[64, 128], out_channels=encoder_to_classifier)
path_classifier = FCNClassifier(in_features=encoder_to_classifier, hidden_features=[128, 64], num_classes=1)
classification_loss_fn = nn.BCEWithLogitsLoss()
reduced_pipeline_model = ReducedPipelineModel.load_from_checkpoint(
    "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_127_train_reduced_pretrained_not_freezed/checkpoints/epoch=9-step=4200.ckpt",
    features_generator=features_generator,
    path_encoder=path_encoder,
    path_classifier=path_classifier,
    edge_classification_loss_fn=classification_loss_fn
)


# %%
first_unet_freeze_parameters = False
second_unet_freeze_parameters = True

first_unet = classic_unet.net
first_unet.layers[-1] = nn.Identity()
if first_unet_freeze_parameters:
    for param in first_unet.parameters():
        param.requires_grad = False

second_unet = reduced_pipeline_model.features_generator
second_unet.layers[7] = nn.Sequential(second_unet.layers[7], nn.ReLU(inplace=True))
if second_unet_freeze_parameters:
    for param in second_unet.parameters():
        param.requires_grad = False

segmentator = nn.Sequential(
    nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 1, kernel_size=(1,1), stride=(1,1))
)

# %%
print(first_unet)

# %%
print(second_unet)


# %%
class DoubleSegmentator(pl.LightningModule):
    def __init__(
        self,
        first_unet: nn.Module,
        second_unet: nn.Module,
        segmentator: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["first_unet", "second_unet", "segmentator"])

        self.first_unet = first_unet
        self.second_unet = second_unet
        self.segmentator = segmentator

        self.loss_fn = loss_fn

        self.lr = lr

    def forward(self, x):
        first_features = self.first_unet(x)
        second_features = self.second_unet(x)

        all_features = torch.cat((first_features, second_features), dim=1)

        res = self.segmentator(all_features)

        return res

    def _shared_step(self, batch, stage: str):
        img, mask = batch
        img = img.float()
        mask = mask.float()

        logits = self(img)
        probs = torch.sigmoid(logits)

        loss = self.loss_fn(probs, mask)

        # Soft Dice for logging
        with torch.no_grad():
            intersection = (probs * mask).sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1.0) / (
                probs.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1.0
            )
            dice = dice.mean()

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_dice", dice, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# %%
from graph_neural_networks.models.losses.dice_loss import DiceLoss

class MixedLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, probs, targets):
        res1 = self.bce(probs, targets)
        res2 = self.dice_loss(probs, targets)

        res = res1 + res2

        return res
    
loss_fn = MixedLoss()

double_segmentator = DoubleSegmentator(first_unet=first_unet,
                                       second_unet=second_unet,
                                       segmentator=segmentator,
                                       loss_fn=loss_fn,
                                       lr=3e-4)

# %%
trainer.fit(double_segmentator, datamodule=datamodule)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DoubleSegmentator.load_from_checkpoint(
    "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_140_train_double_unet_not_freezed/checkpoints/best-checkpoint-epoch=66-val_loss=0.8219.ckpt", 
    map_location=device,
    first_unet=first_unet,
    second_unet=second_unet,
    segmentator=segmentator,
    loss_fn=loss_fn
    )

# %%
trainer.test(model, datamodule=datamodule)

# %%
import matplotlib.pyplot as plt

with torch.no_grad():
    for batch in datamodule.test_dataloader():
        imgs, masks = batch

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        binary_preds = (probs > 0.5).float()
        print(probs.shape, binary_preds.shape)

        plt.figure(figsize=(12, 8))
        binary_pred = binary_preds[0, 0].cpu().numpy()
        mask = masks[0, 0].cpu().numpy()

        dice = (2 * (binary_pred * mask).sum()) / (binary_pred.sum() + mask.sum() + 1e-8)
        print("Dice Score:", dice)

        plt.subplot(1, 2, 1)
        plt.title("Predicted Mask")
        plt.imshow(binary_pred, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask, cmap='gray')
        plt.show()
        plt.close('all')
        
        del imgs, masks, logits, probs, binary_preds

torch.cuda.empty_cache()

# %%
