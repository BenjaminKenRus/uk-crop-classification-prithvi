import albumentations as A
from transforms.temporal import FlattenTemporalIntoChannels, UnflattenTemporalFromChannels

class TemporalCompose:
    """
    Wrapper that safely handles transformations on multi-temporal images.
    """
    def __init__(self, transforms, n_timesteps=3):
        self.flatten = FlattenTemporalIntoChannels()
        self.unflatten = UnflattenTemporalFromChannels(n_timesteps)
        self.transforms = A.Compose(transforms)

    def __call__(self, image, mask):
        image = self.flatten(image)
        sample = self.transforms(image=image, mask=mask)
        sample["image"] = self.unflatten(sample["image"])
        return sample

# Geometric transformations only (safe to perform across collapsed dimensions)
train_transform = TemporalCompose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ],
    n_timesteps=3, # 3 Timesteps total
)

val_transform = None
test_transform = None