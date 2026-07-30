import numpy as np
import rasterio

def calculate_statistics(image_paths):
    """
    Calculates statistical values matching the (6, 3) matrix array alignment.
    """
    band_sum = np.zeros(18, dtype=np.float64)
    band_sq_sum = np.zeros(18, dtype=np.float64)
    pixel_count = 0

    for image_path in image_paths:
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        image /= 10000.0
        image = image.reshape(18, -1)

        band_sum += image.sum(axis=1)
        band_sq_sum += (image ** 2).sum(axis=1)
        pixel_count += image.shape[1]

    means = band_sum / pixel_count
    variance = (band_sq_sum / pixel_count) - (means ** 2)
    variance = np.maximum(variance, 1e-12)
    stds = np.sqrt(variance)

    # Convert the (18,) arrays to (3, 6) then transpose into (6, 3)
    means_fixed = means.reshape(3, 6).transpose(1, 0)
    stds_fixed = stds.reshape(3, 6).transpose(1, 0)

    return means_fixed.astype(np.float32), stds_fixed.astype(np.float32)

MEANS = np.array([
    [0.04559052, 0.05082432, 0.05554022],
    [0.0671344 , 0.08030113, 0.0849574 ],
    [0.06978969, 0.06684965, 0.09187673],
    [0.23877159, 0.3820966 , 0.32901233],
    [0.18698835, 0.20087947, 0.23944895],
    [0.11888084, 0.12409499, 0.15395162],
], dtype=np.float32)

STDS = np.array([
    [0.01535509, 0.02248588, 0.02342191],
    [0.02040194, 0.02749196, 0.02730082],
    [0.02851346, 0.04091552, 0.04886656],
    [0.08491492, 0.09553893, 0.08588763],
    [0.04362026, 0.05835127, 0.05934354],
    [0.03323045, 0.06138553, 0.05847415],
], dtype=np.float32)