import cv2
import numpy as np


def normalized_unsharp_mask(img: np.ndarray, sigma: float = 15) -> np.ndarray:
    """
    S = (I + N{I - blur(I)}) / 2, where N{} is min-max histogram stretching.
    Sharpens detail while the normalization keeps the added high-frequency
    term from blowing out contrast. Expects and returns an RGB image with
    values in [0, 1].
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    diff = img - blurred
    normalized_diff = _stretch(diff)
    return np.clip((img + normalized_diff) / 2.0, 0, 1)


def _stretch(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)
