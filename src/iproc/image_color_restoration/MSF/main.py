import numpy as np

from iproc.image_color_restoration.MSF.color_compensation import compensate_and_balance
from iproc.image_color_restoration.MSF.fusion_weights import aggregated_normalized_weights
from iproc.image_color_restoration.MSF.gamma_correction import auto_gamma_correction
from iproc.image_color_restoration.MSF.morphological_residual import morphological_residual_enhance
from iproc.image_color_restoration.MSF.pyramid_fusion import multiscale_fusion
from iproc.image_color_restoration.MSF.unsharp_masking import normalized_unsharp_mask
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class MSF_ICR(ImageAlgorithmInterface):
    """
    Fusion-based underwater color restoration: adaptive red/blue channel
    compensation and gray-world balancing produce a color-corrected base
    image, which is then routed down two paths — a morphological-residual
    detail layer and a normalized-unsharp-mask layer. Both are combined via
    Laplacian-pyramid multiscale fusion, weighted per pixel by local
    contrast, saliency and saturation, and a final auto-gamma pass adjusts
    perceptual brightness/contrast.

    Reimplements Lepcha, Goyal, Dogra, Vullikanti, Mayan, Sahu, Kumar &
    Siddaraj, "Underwater image enhancement using colour balancing and
    morphological residual processing through gamma correction",
    Sci. Rep. 16, 3182 (2026), https://doi.org/10.1038/s41598-025-33170-9
    (open access). Color compensation, unsharp masking, the Laplacian/
    saliency/saturation weights and the Laplacian-pyramid fusion all follow
    the paper's equations (11-18) and its published constants (t=0.001,
    s=0.1, c=2.4, sigma=15) directly. Two things aren't nailed down by the
    paper itself: it states its morphological-reconstruction operator only
    for a non-negative residual (Eq. 7), leaving the negative branch's sign
    handling to prose ("residual segments are included ... to enhance
    visible edges") rather than a formula — see the comment in
    morphological_residual.py for the reading used here. And the final
    gamma-correction step is named as a contribution but never given a
    formula anywhere in the paper (not even in the appendix/ablation), so
    auto_gamma_correction() is an independent, standard auto-gamma
    implementation consistent with the paper's own description of it
    ("ensures perceptually balanced contrast and brightness"), not a port
    of an equation that was never published.
    """

    T = 0.001
    S = 0.1
    C = 2.4
    SIGMA = 15
    LEVELS = 5

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float64) / 255.0

        balanced = compensate_and_balance(img_f)

        morphological = morphological_residual_enhance(
            balanced, t=MSF_ICR.T, s=MSF_ICR.S, c=MSF_ICR.C, sigma=MSF_ICR.SIGMA
        )
        sharpened = normalized_unsharp_mask(balanced, sigma=MSF_ICR.SIGMA)

        w_morph, w_sharp = aggregated_normalized_weights(morphological, sharpened)

        fused = multiscale_fusion(morphological, sharpened, w_morph, w_sharp, levels=MSF_ICR.LEVELS)
        fused = np.clip(fused, 0, 1)

        result = auto_gamma_correction(fused)

        result = np.clip(result * 255.0, 0, 255)
        return np.uint8(result)
