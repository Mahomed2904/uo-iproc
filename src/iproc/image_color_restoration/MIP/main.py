from iproc.image_color_restoration.MIP.BL import getAtomsphericLight
from iproc.image_color_restoration.MIP.EstimateDepth import DepthMap
from iproc.image_color_restoration.MIP.getRefinedTramsmission import Refinedtransmission
from iproc.image_color_restoration.MIP.TM import getTransmission
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_color_restoration.MIP.sceneRadiance import sceneRadianceRGB


class MIP_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        blockSize = 9

        largestDiff = DepthMap(img, blockSize)
        transmission = getTransmission(largestDiff)
        transmission = Refinedtransmission(transmission, img)
        AtomsphericLight = getAtomsphericLight(transmission, img)
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

        return sceneRadiance

