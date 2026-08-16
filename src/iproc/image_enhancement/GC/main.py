from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_enhancement.GC.sceneRadianceGC import RecoverGC


class GC_IE(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        sceneRadiance = RecoverGC(img)
        return sceneRadiance
