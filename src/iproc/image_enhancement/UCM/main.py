import numpy as np

from iproc.image_enhancement.UCM.color_equalisation import RGB_equalisation
from iproc.image_enhancement.UCM.global_histogram_stretching import stretching
from iproc.image_enhancement.UCM.hsvStretching import HSVStretching
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_enhancement.UCM.sceneRadiance import sceneRadianceRGB


class UCM_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        return sceneRadiance

