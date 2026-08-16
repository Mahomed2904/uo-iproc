import numpy as np

from iproc.image_color_restoration.RoWS.RefinedTramsmission import Refinedtransmission
from iproc.image_color_restoration.RoWS.getAtomsphericLight import getAtomsphericLight
from iproc.image_color_restoration.RoWS.getRGBDarkChannel import getDarkChannel
from iproc.image_color_restoration.RoWS.getTM import getTransmission
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_color_restoration.RoWS.sceneRadiance import sceneRadianceRGB


class RoWS_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        blockSize = 9

        RGB_Darkchannel = getDarkChannel(img, blockSize)
        AtomsphericLight = getAtomsphericLight(RGB_Darkchannel, img)
        print('AtomsphericLight', AtomsphericLight)
        transmission = getTransmission(img, AtomsphericLight, blockSize)
        print('transmission', transmission)
        print('np.mean(transmission)', np.mean(transmission))
        transmission = Refinedtransmission(transmission, img)
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

        return sceneRadiance

