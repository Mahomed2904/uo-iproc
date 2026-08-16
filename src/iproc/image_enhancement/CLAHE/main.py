from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_enhancement.CLAHE.sceneRadianceCLAHE import RecoverCLAHE


class CLAHE_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        sceneRadiance = RecoverCLAHE(img)
        return sceneRadiance

