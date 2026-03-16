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
import cv2
import matplotlib.pyplot as plt
import numpy as np

# %%
from graph_neural_networks.models.binary_segmentator import BinarySegmentator
from graph_neural_networks.data.datamodules.image_datamodule import ImageDatamodule
from graph_neural_networks.data.dataset.image_dataset import ImageDataset

# %%
dataset = "FIVES_clean"
data_dir = f"/home/morand/afs/EVAPORE/data/{dataset}/"

# %% [markdown]
# # Show dataset without data augmentation

# %%
dataset = ImageDataset(data_dir)

train_transforms = A.Compose([
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

datamodule = ImageDatamodule(
    dataset=dataset,
    val_split=0.2,
    test_split=0.1,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    num_workers=0,
    train_batch_size=1,
    val_batch_size=1,
    seed=42,
    shuffle_train=False
)

datamodule.setup()
dataloader = datamodule.train_dataloader()

# %%
for i, batch in enumerate(dataloader):
    img, gt = batch
    img = img.squeeze(0).permute(1, 2, 0).numpy()  # (1, C, H, W) -> (H, W, C)
    gt = gt.squeeze(0).squeeze(0).numpy()  # (1, 1, H, W) -> (H, W)
    print(f"Image shape: {img.shape}, GT shape: {gt.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")
    plt.show()

    if i > 5:
        break


# %% [markdown]
# # Show dataset with data augmentation (but without normalization)

# %%
class AddGaussNoise:
    def __init__(self, std: float | tuple[float, float] = 0.01):
        self.std = std

    def __call__(self, image, **kwargs):
        if isinstance(self.std, tuple):
            std = np.random.uniform(self.std[0], self.std[1])
        else:
            std = self.std
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0)


# %%
dataset = ImageDataset(data_dir)

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),

    A.Affine(
        rotate=(-5, 5),
        scale=1.0,  # keep size
        translate_percent=(0, 0),  # no translation
        shear=0,  # no shear
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,  # important for masks
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,  # background fill for images
        fill_mask=0,  # background fill for masks
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=(-0.15, 0.15),
        contrast_limit=(-0.15, 0.15),
        p=0.5
    ),
    A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),

    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

datamodule = ImageDatamodule(
    dataset=dataset,
    val_split=0.2,
    test_split=0.1,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    num_workers=0,
    train_batch_size=1,
    val_batch_size=1,
    seed=42,
    shuffle_train=False
)

datamodule.setup()
dataloader = datamodule.train_dataloader()

# %%
img = np.random.rand(256, 256, 3).astype(np.float32)

out1 = train_transforms(image=img)["image"]
out2 = train_transforms(image=img)["image"]

print(torch.allclose(out1, out2))

# %%
for i, batch in enumerate(dataloader):
    img, gt = batch
    img = img.squeeze(0).permute(1, 2, 0).numpy()  # (1, C, H, W) -> (H, W, C)
    gt = gt.squeeze(0).squeeze(0).numpy()  # (1, 1, H, W) -> (H, W)
    print(f"Image shape: {img.shape}, GT shape: {gt.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap="gray")
    plt.axis("off")
    plt.show()

    if i > 5:
        break

# %% [markdown]
# # Show dataset with data augmentation and patches (without normalization)

# %%
dataset = ImageDataset(data_dir)

patch_size = 1024

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),

    A.Affine(
        rotate=(-5, 5),
        scale=1.0,  # keep size
        translate_percent=(0, 0),  # no translation
        shear=0,  # no shear
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,  # important for masks
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,  # background fill for images
        fill_mask=0,  # background fill for masks
        p=0.5
    ),
    
    A.CropNonEmptyMaskIfExists(
        height=patch_size,
        width=patch_size,
        p=1.0,
    ),

    A.RandomBrightnessContrast(
        brightness_limit=(-0.15, 0.15),
        contrast_limit=(-0.15, 0.15),
        p=0.5
    ),
    A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),

    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

datamodule = ImageDatamodule(
    dataset=dataset,
    val_split=0.2,
    test_split=0.1,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    num_workers=0,
    train_batch_size=16,
    val_batch_size=1,
    seed=42,
    shuffle_train=False
)

datamodule.setup()
dataloader = datamodule.train_dataloader()

# %%
for i, batch in enumerate(dataloader):
    img, gt = batch
    bs = img.shape[0]
    for b in range(bs):
        crop_img = img[b].permute(1, 2, 0).numpy()  # (bs, C, H, W) -> (H, W, C)
        crop_gt = gt[b].squeeze().numpy()  # (bs, 1, H, W) -> (H, W)
        print(f"Image shape: {crop_img.shape}, GT shape: {crop_gt.shape}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(crop_img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Ground Truth")
        plt.imshow(crop_gt, cmap="gray")
        plt.axis("off")
        plt.show()

        if b >= 5:
            break

    break

# %% [markdown]
# # Show dataset with data augmentation, crop and normalization

# %%
dataset = ImageDataset(data_dir)
mean = dataset.stats["mean"]
std = dataset.stats["std"]

print(f"Dataset: {dataset}")
print(f"Mean: {mean}")
print(f"Std: {std}")

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),

    A.Affine(
        rotate=(-5, 5),
        scale=1.0,  # keep size
        translate_percent=(0, 0),  # no translation
        shear=0,  # no shear
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,  # important for masks
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,  # background fill for images
        fill_mask=0,  # background fill for masks
        p=0.5
    ),

    A.CropNonEmptyMaskIfExists(
        height=patch_size,
        width=patch_size,
        p=1.0,
    ),

    A.RandomBrightnessContrast(
        brightness_limit=(-0.15, 0.15),
        contrast_limit=(-0.15, 0.15),
        p=0.5
    ),
    A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),

    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

datamodule = ImageDatamodule(
    dataset=dataset,
    val_split=0.2,
    test_split=0.1,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    num_workers=0,
    train_batch_size=16,
    val_batch_size=1,
    seed=42,
    shuffle_train=False
)

datamodule.setup()
dataloader = datamodule.train_dataloader()

# %%
for i, batch in enumerate(dataloader):
    img, gt = batch
    bs = img.shape[0]
    for b in range(bs):
        crop_img = img[b].permute(1, 2, 0).numpy()  # (bs, C, H, W) -> (H, W, C)
        crop_gt = gt[b].squeeze().numpy()  # (bs, 1, H, W) -> (H, W)
        print(f"Image shape: {crop_img.shape}, GT shape: {crop_gt.shape}")

        histogram, bin_edges = np.histogram(crop_img, bins=100)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(crop_img)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(crop_gt, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Pixel Intensity Histogram")
        plt.plot(bin_edges[:-1], histogram)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.show()

        if b >= 5:
            break

    break

# %% [markdown]
# # Train data augmentation

# %%
dataset = ImageDataset(data_dir)
mean = dataset.stats["mean"]
std = dataset.stats["std"]

print(f"Dataset: {dataset}")
print(f"Mean: {mean}")
print(f"Std: {std}")

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),

    A.Affine(
        rotate=(-5, 5),
        scale=1.0,  # keep size
        translate_percent=(0, 0),  # no translation
        shear=0,  # no shear
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,  # important for masks
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,  # background fill for images
        fill_mask=0,  # background fill for masks
        p=0.5
    ),

    A.CropNonEmptyMaskIfExists(
        height=patch_size,
        width=patch_size,
        p=1.0,
    ),

    A.RandomBrightnessContrast(
        brightness_limit=(-0.15, 0.15),
        contrast_limit=(-0.15, 0.15),
        p=0.5
    ),
    A.Lambda(image=AddGaussNoise(std=(0.005, 0.015)), p=0.5),

    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

val_transforms = A.Compose([
    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
    ToTensorV2(),
], additional_targets={"fg_mask": "mask"})

datamodule = ImageDatamodule(
    dataset=dataset,
    val_split=0.2,
    test_split=0.1,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
    num_workers=0,
    train_batch_size=4,
    val_batch_size=1,
    seed=42,
    shuffle_train=True
)

# %%
binary_segmentator = BinarySegmentator(
    lr=1e-3,
    input_channels=3,
    num_layers=4,
    features_start=32,
    bilinear=True,
    norm_op='instance',
    warmup_epochs=5,
    dropout=0.2,
    kernel_size=5
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
        patience=100,
        min_delta=0.00,
        verbose=True,
        mode="min"
    )
]

trainer = Trainer(accelerator='gpu', devices="auto", max_epochs=1000, precision='16-mixed', callbacks=callbacks)
torch.set_float32_matmul_precision("medium")

# %%
trainer.fit(binary_segmentator, datamodule=datamodule)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "/home/morand/afs/EVAPORE/notebooks/lightning_logs/version_202/checkpoints/best-checkpoint-epoch=65-val_loss=0.0685.ckpt"

model = BinarySegmentator.load_from_checkpoint(ckpt_path, map_location=device)

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
