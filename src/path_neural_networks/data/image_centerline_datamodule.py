from torch.utils.data import Subset
import albumentations as A

from path_neural_networks.data.image_centerline_dataset import ImageCenterlineDataset
from image_segmentation.data.image_datamodule import ImageDatamodule

class ImageCenterlineDatamodule(ImageDatamodule):
    def __init__(
            self,
            dataset: ImageCenterlineDataset,
            split_file_path: str = None,
            train_split_name: str = 'train',
            val_split_ratio: float = 0.2,
            train_transforms: A.Compose = None,
            val_transforms: A.Compose = None,
            test_transforms: A.Compose = None,
            seed: int = 42,
            shuffle_train: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            split_file_path=split_file_path,
            train_split_name=train_split_name,
            val_split_ratio=val_split_ratio,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=1,
            train_batch_size=1,
            val_batch_size=1,
            seed=seed,
            shuffle_train=shuffle_train,
            *args,
            **kwargs
        )

    def create_subsets(self):
        self.train_dataset = Subset(
            ImageCenterlineDataset(
                data_dir=self.full_dataset.data_dir,
                transforms=self.train_transforms,
                centerline_dirname=self.full_dataset.centerline_dirname),
            self.train_indices)
        
        self.val_dataset = Subset(
            ImageCenterlineDataset(
                data_dir=self.full_dataset.data_dir,
                transforms=self.val_transforms, 
                centerline_dirname=self.full_dataset.centerline_dirname),
            self.val_indices)
        
        self.test_dataset = Subset(
            ImageCenterlineDataset(
                data_dir=self.full_dataset.data_dir,
                transforms=self.test_transforms,
                centerline_dirname=self.full_dataset.centerline_dirname),
            self.test_indices)