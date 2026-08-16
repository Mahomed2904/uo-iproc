import numpy as np

from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class NONE_ALG(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        return img

