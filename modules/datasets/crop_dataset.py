from pathlib import Path
import numpy as np
import rasterio
import torch
from torchgeo.datasets import NonGeoDataset

class CropDataset(NonGeoDataset):
    """
    Dataset for multitemporal Sentinel-2 crop segmentation matching Prithvi v2.
    Expected return shape: {"image": (6, 3, H, W), "mask": (H, W)}
    """
    num_classes = 16

    def __init__(self, image_files, mask_files, means, stds, transform=None):
        super().__init__()
        self.image_files = image_files
        self.mask_files = mask_files
        
        # Ensure stats match (Bands, Timesteps) -> (6, 3) configuration
        self.means = np.asarray(means, dtype=np.float32).reshape(6, 3)
        self.stds = np.asarray(stds, dtype=np.float32).reshape(6, 3)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self._read_image(self.image_files[index])
        mask = self._read_mask(self.mask_files[index])

        # 1. Apply Temporal Augmentations while still in NumPy format
        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        # 2. Normalize image using broadcasting array rules (6, 3, 1, 1)
        image = (image - self.means[:, :, None, None]) / self.stds[:, :, None, None]

        # 3. Structural conversions to PyTorch
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "filename": self.image_files[index].stem,
        }

    def _read_image(self, image_path):
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        image /= 10000.0  # Scale Sentinel-2 Reflectance
        H, W = image.shape[1:]
        
        # Input shape (18, H, W) -> Reshape to (3, 6, H, W) [Timestamps, Bands, H, W]
        image = image.reshape(3, 6, H, W)
        # Transpose to (6, 3, H, W) [Bands, Timestamps, H, W] as required by Prithvi v2
        image = np.transpose(image, (1, 0, 2, 3))
        return image

    def _read_mask(self, mask_path):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mask = np.nan_to_num(mask, nan=0, posinf=0, neginf=0).astype(np.int64)
        valid = np.isin(mask, np.arange(17))
        mask[~valid] = 0

        mask[mask == 0] = -1  # Model ignore index
        mask[mask > 0] -= 1   # Map classes 1..16 down to 0..15
        return mask