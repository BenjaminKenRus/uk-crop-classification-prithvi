import numpy as np

class FlattenTemporalIntoChannels:
    """
    Convert (Bands, Timesteps, H, W) into (H, W, Bands * Timesteps) for Albumentations.
    """
    def __call__(self, image):
        # Move H and W to the front -> (H, W, Bands, Timesteps)
        image = np.transpose(image, (2, 3, 0, 1))
        H, W, C, T = image.shape
        # Collapse Channels and Timesteps into a single axis
        image = image.reshape(H, W, C * T)
        return image


class UnflattenTemporalFromChannels:
    """
    Convert (H, W, Bands * Timesteps) back into (Bands, Timesteps, H, W).
    """
    def __init__(self, n_timesteps=3):
        self.n_timesteps = n_timesteps

    def __call__(self, image):
        H, W, total_channels = image.shape
        n_bands = total_channels // self.n_timesteps

        # Reshape back to spatial dimensions
        image = image.reshape(H, W, n_bands, self.n_timesteps)
        # Transpose back to (Bands, Timesteps, H, W)
        image = np.transpose(image, (2, 3, 0, 1))
        return image