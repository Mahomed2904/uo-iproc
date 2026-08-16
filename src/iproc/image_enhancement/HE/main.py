import numpy as np

from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_enhancement.HE.sceneRadianceHE import RecoverHE


class HE_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        sceneRadiance = RecoverHE(img)
        return sceneRadiance
