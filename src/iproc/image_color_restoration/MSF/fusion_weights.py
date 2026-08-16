from typing import Tuple

import cv2
import numpy as np

_DELTA = 0.1


def aggregated_normalized_weights(
    img_a: np.ndarray, img_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines Laplacian-contrast, saliency and saturation weight maps for each
    of the two fusion inputs into a single per-pixel weight, then normalizes
    the pair so they sum to 1 at every pixel (with a small regularizer delta
    so neither input is ever fully zeroed out). Expects RGB images with
    values in [0, 1]; returns two single-channel weight maps.
    """
    w_a = _combined_weight(img_a)
    w_b = _combined_weight(img_b)

    denom = w_a + w_b + 2 * _DELTA
    w_a_norm = (w_a + _DELTA) / denom
    w_b_norm = (w_b + _DELTA) / denom
    return w_a_norm, w_b_norm


def _combined_weight(img: np.ndarray) -> np.ndarray:
    return _laplacian_contrast_weight(img) + _saliency_weight(img) + _saturation_weight(img)


def _laplacian_contrast_weight(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return np.abs(laplacian)


def _saliency_weight(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    blurred = cv2.GaussianBlur(lab, (0, 0), sigmaX=5)
    mean_color = lab.reshape(-1, 3).mean(axis=0)
    diff = blurred - mean_color
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _saturation_weight(img: np.ndarray) -> np.ndarray:
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    luminance = (r + g + b) / 3.0
    return np.sqrt(((r - luminance) ** 2 + (g - luminance) ** 2 + (b - luminance) ** 2) / 3.0)
