from abc import ABC, abstractmethod
import numpy as np


class ImageAlgorithmInterface(ABC):

    @staticmethod
    @abstractmethod
    def apply(img: np.ndarray) -> np.ndarray:
        pass

