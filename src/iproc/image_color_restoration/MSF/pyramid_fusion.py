from typing import List

import cv2
import numpy as np


def multiscale_fusion(
    img_a: np.ndarray,
    img_b: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    levels: int = 5,
) -> np.ndarray:
    """
    Laplacian-pyramid fusion of two RGB images ([0, 1]) driven by a pair of
    per-pixel weight maps (already normalized to sum to 1). Decomposing into
    detail levels before blending, rather than blending the images directly,
    avoids the halos a naive per-pixel weighted average would introduce
    across sharp edges.
    """
    levels = _max_feasible_levels(img_a.shape[0], img_a.shape[1], levels)

    gauss_a = _gaussian_pyramid(img_a, levels)
    gauss_b = _gaussian_pyramid(img_b, levels)
    lap_a = _laplacian_pyramid(gauss_a)
    lap_b = _laplacian_pyramid(gauss_b)

    gauss_wa = _gaussian_pyramid(weight_a, levels)
    gauss_wb = _gaussian_pyramid(weight_b, levels)

    fused = [
        gauss_wa[l][:, :, None] * lap_a[l] + gauss_wb[l][:, :, None] * lap_b[l]
        for l in range(levels)
    ]

    result = fused[-1]
    for l in range(levels - 2, -1, -1):
        size = (fused[l].shape[1], fused[l].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + fused[l]
    return result


def _max_feasible_levels(height: int, width: int, requested: int) -> int:
    max_by_size = int(np.floor(np.log2(min(height, width)))) - 1
    return max(1, min(requested, max_by_size))


def _gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    pyramid = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _laplacian_pyramid(gauss_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    levels = len(gauss_pyramid)
    lap_pyramid = []
    for i in range(levels - 1):
        size = (gauss_pyramid[i].shape[1], gauss_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gauss_pyramid[i + 1], dstsize=size)
        lap_pyramid.append(gauss_pyramid[i] - expanded)
    lap_pyramid.append(gauss_pyramid[-1])
    return lap_pyramid
