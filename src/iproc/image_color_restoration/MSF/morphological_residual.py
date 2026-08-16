import cv2
import numpy as np
from skimage.morphology import remove_small_objects, reconstruction


def morphological_residual_enhance(
    img: np.ndarray, t: float = 0.001, s: float = 0.1, c: float = 2.4, sigma: float = 15
) -> np.ndarray:
    """
    Morphological Processed Residuals (MPR): splits the image into a smooth
    base layer and a high-frequency residual, keeps only the residual
    structures that are both significant (|residual| >= t) and span a
    non-trivial connected area (removes components smaller than s percent of
    the image area), then re-injects the surviving positive/negative detail
    back onto the base with contrast gain c. Suppresses noise-driven texture
    while avoiding the halos a plain unsharp mask would introduce.
    Expects and returns an RGB image with values in [0, 1].
    """
    base = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    residual = img - base

    positive = np.maximum(residual, 0)
    negative = np.maximum(-residual, 0)

    min_size = max(1, int((s / 100.0) * img.shape[0] * img.shape[1]))
    significance_mask = _significant_region_mask(residual, t, min_size)

    m_pos = _reconstruct_channels(positive, significance_mask)
    m_neg = _reconstruct_channels(negative, significance_mask)

    # m_pos/m_neg are both reconstructed as non-negative magnitudes, so
    # subtracting m_neg restores the negative branch's own sign — i.e. this
    # re-injects the residual's original local structure (dips stay dips,
    # bumps stay bumps), matching the paper's "significant residual segments
    # are included into the image to enhance visible edges".
    out = base + (m_pos - m_neg) * c
    return np.clip(out, 0, 1)


def _significant_region_mask(residual: np.ndarray, t: float, min_size: int) -> np.ndarray:
    magnitude = np.abs(residual).max(axis=2)
    mask = magnitude >= t
    if mask.any():
        mask = remove_small_objects(mask, min_size=min_size)
    return mask


def _reconstruct_channels(component: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Grayscale morphological reconstruction by dilation: marker keeps only the
    parts of `component` inside the significance mask, and reconstruction
    grows those seeds back out to fill their full connected region in
    `component`, discarding regions the marker never touched.
    """
    out = np.zeros_like(component)
    for ch in range(component.shape[2]):
        layer = component[:, :, ch]
        marker = np.where(mask, layer, 0.0)
        out[:, :, ch] = reconstruction(marker, layer, method="dilation")
    return out
