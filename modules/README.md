# Crop Dataset Datamodules

## Datamodules
### crop_datamodule.py
This is the central datamodule. It verifies the dataset, splits it into training, validation, and testing subsets, and computes normalisation statistics.

## Datasets
### crop_dataset.py
This is responsible for loading image-mask pairs, applying augmentations, normalising the imagery, and overall formatting for the Prithvi architecture.

## Transforms 
### augmentations.py
Temporarily flattens the temporal dimension, allowing augmentations to be applied.
### temporal.py
Additional classes that provide conversions between the different formats.

## Utils
### statistics.py
Computes normalisation statistics required. If you want to compute your own normalisation statistics replace MEANS and STDS with None.

