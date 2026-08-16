import numpy as np

from iproc.image_enhancement.RayleighDistribution.color_equalisation import RGB_equalisation
from iproc.image_enhancement.RayleighDistribution.global_stretching_RGB import stretching
from iproc.image_enhancement.RayleighDistribution.hsvStretching import HSVStretching

from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_enhancement.RayleighDistribution.rayleighDistribution import rayleighStretching
from iproc.image_enhancement.RayleighDistribution.sceneRadiance import sceneRadianceRGB

e = np.e
esp = 2.2204e-16


class RayleighDistribution_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        height = len(img)
        width = len(img[0])

        sceneRadiance = RGB_equalisation(img, height, width)
        # sceneRadiance = stretching(img)
        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(sceneRadiance, height, width)

        sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2

        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)

        return sceneRadiance

