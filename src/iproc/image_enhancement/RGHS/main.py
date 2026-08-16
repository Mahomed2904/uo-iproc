
# encoding=utf-8
import numpy as np

from iproc.image_enhancement.RGHS.LabStretching import LABStretching
from iproc.image_enhancement.RGHS.global_stretching_RGB import stretching
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class RGHS_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        height = len(img)
        width = len(img[0])
        # sceneRadiance = RGB_equalisation(img)

        sceneRadiance = img
        # sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)

        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance = LABStretching(sceneRadiance)

        return sceneRadiance

