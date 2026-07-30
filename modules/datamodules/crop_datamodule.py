from pathlib import Path

import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.crop_dataset import CropDataset
from utils.statistics import (
    calculate_statistics,
    MEANS,
    STDS,
)


class CropDataModule(L.LightningDataModule):
    """
    Lightning DataModule for multitemporal Sentinel-2 crop segmentation.

    Expected directory structure
    ----------------------------
    dataset3/
        images/
        masks/
    """

    def __init__(
        self,
        data_root,
        batch_size=8,
        num_workers=4,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        test_size=0.20,
        val_size=0.20,
        random_seed=42,
    ):
        super().__init__()

        self.data_root = Path(data_root)

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

    ####################################################################
    # Lightning
    ####################################################################

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        image_files = sorted((self.data_root / "images").glob("*.tif"))
        mask_files = sorted((self.data_root / "masks").glob("*.tif"))

        if len(image_files) == 0:
            raise RuntimeError("No image files found.")

        if len(image_files) != len(mask_files):
            raise RuntimeError("Different numbers of images and masks.")

        samples = []

        for image, mask in zip(image_files, mask_files):

            if image.stem != mask.stem:

                raise RuntimeError(
                    f"{image.name} does not match {mask.name}"
                )

            samples.append((image, mask))

        train_samples, test_samples = train_test_split(
            samples,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True,
        )

        train_samples, val_samples = train_test_split(
            train_samples,
            test_size=self.val_size,
            random_state=self.random_seed,
            shuffle=True,
        )

        train_images = [x[0] for x in train_samples]
        train_masks = [x[1] for x in train_samples]

        val_images = [x[0] for x in val_samples]
        val_masks = [x[1] for x in val_samples]

        test_images = [x[0] for x in test_samples]
        test_masks = [x[1] for x in test_samples]

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        if MEANS is None or STDS is None:
            means, stds = calculate_statistics(train_images)

            print("MEANS =")
            print(repr(means))

            print("STDS =")
            print(repr(stds))
        else:
            means, stds = MEANS, STDS

        self.train_dataset = CropDataset(
            train_images,
            train_masks,
            means,
            stds,
            transform=self.train_transform,
        )

        self.val_dataset = CropDataset(
            val_images,
            val_masks,
            means,
            stds,
            transform=self.val_transform,
        )

        self.test_dataset = CropDataset(
            test_images,
            test_masks,
            means,
            stds,
            transform=self.test_transform,
        )

    ####################################################################
    # DataLoaders
    ####################################################################

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )