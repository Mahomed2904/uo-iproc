import numpy as np


def auto_gamma_correction(img: np.ndarray, target_mean: float = 0.5) -> np.ndarray:
    """
    Final perceptual contrast/brightness pass: picks gamma so the image's
    mean luminance moves to target_mean, then applies it to every channel.
    Expects and returns an RGB image with values in [0, 1].
    """
    img = np.clip(img, 0, 1)
    mean_luminance = np.clip(img.mean(), 1e-4, 1 - 1e-4)
    gamma = np.log(target_mean) / np.log(mean_luminance)
    return np.power(img, gamma)
