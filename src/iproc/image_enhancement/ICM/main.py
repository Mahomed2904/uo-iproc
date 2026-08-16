import numpy as np

from iproc.image_enhancement.ICM.global_histogram_stretching import stretching
from iproc.image_enhancement.ICM.hsvStretching import HSVStretching
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_enhancement.ICM.sceneRadiance import sceneRadianceRGB


class ICM_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        img = stretching(img)
        sceneRadiance = sceneRadianceRGB(img)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        return sceneRadiance
