import cv2
import numpy as np

from iproc.image_color_restoration.MIX.contrast_correction.contrast_correction import ContrastCorrection
from iproc.image_color_restoration.MIX.color_correction.color_correction import ColorCorrection
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class MIX_ICR(ImageAlgorithmInterface):

    __contrast_correction = ContrastCorrection()
    __color_correction = ColorCorrection()

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        contrast_corrected_image = MIX_ICR.__contrast_correction.process(img)
        color_corrected_image = MIX_ICR.__color_correction.color_correct(contrast_corrected_image)
        image_bgr = cv2.cvtColor(color_corrected_image, cv2.COLOR_RGB2BGR)
        return image_bgr

