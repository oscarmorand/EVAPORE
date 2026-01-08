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
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# %%
class PathEncoder(nn.Module, ABC):
    def __init__(self, ):
        super(PathEncoder, self).__init__()

    @abstractmethod
    def forward(self, paths):
        pass


# %%
class PathMaxValueEncoder(PathEncoder):
    def __init__(self, ):
        super(PathMaxValueEncoder, self).__init__()

    def forward(self, 
                paths: torch.Tensor # shape (num_paths, num_features, path_length)
    ) -> torch.Tensor:
        # Example implementation: return the maximum value in each path
        return torch.max(paths, dim=2)


# %%
class PathConvEncoder(PathEncoder):
    def __init__(self, in_channels: int, out_channels: int):
        super(PathConvEncoder, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, 
                paths: torch.Tensor # shape (num_paths, num_features, path_length)
    ) -> torch.Tensor:
        # Apply 1D convolution followed by ReLU activation
        conv_output = self.conv1d(paths)
        activated_output = self.relu(conv_output)
        # Pooling operation (e.g., max pooling) to get fixed-size representation
        pooled_output = torch.max(activated_output, dim=2)[0]
        return pooled_output


# %%
class SimpleFeatureMapsGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SimpleFeatureMapsGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.conv(x)
        return x


# %%
class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
from torch.nn import BCELoss
import matplotlib.pyplot as plt
import numpy as np

input_data = torch.randn((3, 100, 100), requires_grad=True) # Example input data (channels, height, width)

feature_maps_generator = SimpleFeatureMapsGenerator(in_channels=3, out_channels=32)

feature_maps = feature_maps_generator(input_data)  # shape (channels, height, width)

paths_centerlines_lengths = torch.randint(5, 100, (50,)) # Example path lengths
paths_centerlines = [torch.randint(0, 100, (length, 2)) for length in paths_centerlines_lengths] # Example paths

print("Feature maps shape:", feature_maps.shape)
print("Number of paths:", len(paths_centerlines))
print("First path length:", paths_centerlines_lengths[0])
print("First path coordinates:", paths_centerlines[0])

encoder = PathConvEncoder(in_channels=32, out_channels=64)
classifier = SimpleClassifier(in_features=64, num_classes=2)

loss = BCELoss()

targets_values = torch.randint(0, 2, (len(paths_centerlines),)).float()  # Example binary targets
targets = torch.zeros((len(paths_centerlines), 2))
targets[torch.arange(len(paths_centerlines)), targets_values.long()] = 1.0
print(len(targets), targets[0:10])

for i, path_centerline in enumerate(paths_centerlines):
    path_features = feature_maps[:, path_centerline[:, 0], path_centerline[:, 1]]  # shape (channels, path_length)
    print("Path features shape:", path_features.shape)

    x = np.arange(path_features.shape[1])
    plt.plot(x, path_features.detach().numpy().T)
    plt.title(f"Path {i} features")
    plt.xlabel("Path length")
    plt.ylabel("Feature values")
    plt.show()

    path_features = path_features.unsqueeze(0)  # shape (1, channels, path_length)
    encoded_path = encoder(path_features)  # shape (1, out_channels)
    print("Encoded path shape:", encoded_path.shape)

    classification = classifier(encoded_path)  # shape (1, num_classes)
    classification = torch.sigmoid(classification).squeeze(0)  # shape (num_classes)
    
    target = targets[i] # shape (num_classes)

    print(target.numpy(), classification.detach().numpy())

    l = loss(classification, target)
    print(l.item())

# %%
from PIL import Image
from graph.graph_creation import img_to_graph
from plotly import graph_objects as go

feature_maps_path = "/home/morand/afs/EVAPORE/data/FIVES/feature_maps/qtseg/FIVES_001.pt"
feature_maps = torch.load(feature_maps_path)

print(feature_maps.shape)

mask_path = "/home/morand/afs/EVAPORE/data/FIVES/gt/FIVES_001.png"
mask = np.array(Image.open(mask_path).convert("L").resize((feature_maps.shape[2], feature_maps.shape[1])))
print(mask.shape)

graph = img_to_graph(mask, clean=True, closing_radius=1, return_pixel_graph=False)
print("Number of edges in the graph:", graph.number_of_edges())
edges_centerlines = []
for e in graph.edges:
    edges_centerlines.append(np.array(graph.edges[e]['centerline']))
print("Number of edges centerlines:", len(edges_centerlines))
print("First edge centerline shape:", edges_centerlines[0].shape)

encoder = PathConvEncoder(in_channels=feature_maps.shape[0], out_channels=64)

for i, edge_centerline in enumerate(edges_centerlines):
    if i > 5:
        break
    path_features = feature_maps[:, edge_centerline[:, 0], edge_centerline[:, 1]]  # shape (channels, path_length)
    print("Path features shape:", path_features.shape)

    x = np.arange(path_features.shape[1])
    fig = go.Figure()
    for c in range(path_features.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=path_features[c].detach().numpy(), mode='lines', name=f'Channel {c}'))
    fig.update_layout(title=f'Edge {i} Features', xaxis_title='Path length', yaxis_title='Feature values')
    fig.show()

    encoded_path = encoder(path_features.unsqueeze(0))  # shape (1, out_channels)
    print("Encoded path shape:", encoded_path.shape)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(
            self,
            input_channels: int = 3,
            num_classes: int = 2,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        """
        Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
        <https://arxiv.org/abs/1505.04597>`_

        Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_
            - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

        Args:
            num_classes: Number of output classes required
            num_layers: Number of layers in each side of U-net (default 5)
            features_start: Number of features in first layer (default 64)
            bilinear (bool): Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
        """
        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# %%
import pytorch_lightning as pl

class BinarySegmentator(pl.LightningModule):
    def __init__(self,
                 datamodule: pl.LightningDataModule = None,
                 lr: float = 0.01,
                 input_channels: int = 3,
                 num_layers: int = 5,
                 features_start: int = 64,
                 bilinear: bool = False,
                 network: str = 'unet',
                 **kwargs
                 ):
        """
        Basic model for semantic segmentation. Uses UNet architecture by default.

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        Example::

            >>> from pl_bolts.models.vision import SemSegment
            >>> from pl_bolts.datamodules import KittiDataModule
            >>> import pytorch_lightning as pl
            ...
            >>> dm = KittiDataModule('path/to/kitt/dataset/', batch_size=4)
            >>> model = SemSegment(datamodule=dm)
            >>> trainer = pl.Trainer()
            >>> trainer.fit(model)

        Args:
            datamodule: LightningDataModule
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        assert datamodule
        self.datamodule = datamodule

        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(input_channels=input_channels,
                        num_classes=2,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.binary_cross_entropy_with_logits(out, mask.float(), ignore_index=250)
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.binary_cross_entropy_with_logits(out, mask.float(), ignore_index=250)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


# %%
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FIVESImageDataset(Dataset):

    IMAGE_PATH = 'img'
    GT_PATH = 'gt'
    FOREGROUND_MASK_PATH = 'foreground_masks'

    def __init__(
            self,
            data_dir: str,
            img_size: tuple = (2000, 2000),
            transform=None
    ):
        self.img_size = img_size
        self.transform = transform

        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)
        self.gt_path = os.path.join(self.data_dir, self.GT_PATH)
        self.foreground_mask_path = os.path.join(self.data_dir, self.FOREGROUND_MASK_PATH)

        self.img_list = self.get_filenames(self.img_path)
        self.gt_list = self.get_filenames(self.gt_path)
        self.foreground_mask_list = self.get_filenames(self.foreground_mask_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).resize(self.img_size)
        img = np.array(img).astype(np.float32)

        gt_path = self.gt_list[idx]
        gt = Image.open(gt_path).convert('L').resize(self.img_size)
        gt = np.array(gt)
        gt = (gt > 0).astype(np.float32)

        if len(self.foreground_mask_list) > 0:
            if len(self.foreground_mask_list) > 1:
                foreground_mask_path = self.foreground_mask_list[idx]
            else:
                foreground_mask_path = self.foreground_mask_list[0]
            foreground_mask = torch.load(foreground_mask_path)
            foreground_mask = F.interpolate(foreground_mask.unsqueeze(0).unsqueeze(0).float(), size=self.img_size, mode='nearest').squeeze().numpy()
            foreground_mask = (foreground_mask > 0).astype(np.float32)

            gt = gt * foreground_mask
            img = img * foreground_mask[:, :, np.newaxis]

            img = (img - np.mean(img[foreground_mask > 0], axis=0)) / np.std(img[foreground_mask > 0], axis=0)
        else:
            img = (img - np.mean(img, axis=(0,1))) / np.std(img, axis=(0,1))

        if self.transform:
            img = self.transform(img)

        return img, gt

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


# %%
import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split


class FIVESImageDatamodule(LightningDataModule):

    name = 'FIVES'

    def __init__(
            self,
            base_data_dir: str,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        FIVES train, validation and test dataloaders.

        Args:
            data_dir: where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
        """
        super().__init__(*args, **kwargs)
        self.data_dir = os.path.join(base_data_dir, self.name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.default_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        # split into train, val, test
        fives_dataset = FIVESImageDataset(self.data_dir, transform=self.default_transforms)

        val_len = round(val_split * len(fives_dataset))
        test_len = round(test_split * len(fives_dataset))
        train_len = len(fives_dataset) - val_len - test_len
        self.trainset, self.valset, self.testset = random_split(fives_dataset,
                                                                lengths=[train_len, val_len, test_len],
                                                                generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        loader = DataLoader(self.trainset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.testset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader


# %%
datamodule = FIVESImageDatamodule(base_data_dir='/home/morand/afs/EVAPORE/data/', batch_size=1, num_workers=7)

binary_segmentator = BinarySegmentator(datamodule=datamodule, input_channels=3, num_layers=5, features_start=64, bilinear=False)

# %%
test_input = datamodule.test_dataloader().dataset[0][0].unsqueeze(0)
print("Test input shape:", test_input.shape)

# %%
import matplotlib.pyplot as plt

plt.imshow(test_input.squeeze(0).permute(1, 2, 0).numpy()[:,:,0])
plt.colorbar()
plt.show()

plt.imshow(test_input.squeeze(0).permute(1, 2, 0).numpy()[:,:,1])
plt.colorbar()
plt.show()

plt.imshow(test_input.squeeze(0).permute(1, 2, 0).numpy()[:,:,2])
plt.colorbar()
plt.show()

# %%
out = binary_segmentator(test_input)
print("Output shape:", out.shape)

# %%
plt.imshow(out.squeeze(0).detach().numpy()[0,:,:])
plt.colorbar()
plt.show()

plt.imshow(out.squeeze(0).detach().numpy()[1,:,:])
plt.colorbar()
plt.show()

seg = torch.argmax(out, dim=1)
print("Segmentation shape:", seg.shape)
plt.figure(figsize=(30,30))
plt.imshow(seg.squeeze(0).detach().numpy(), cmap='gray')
plt.colorbar()
plt.show()

# %%
