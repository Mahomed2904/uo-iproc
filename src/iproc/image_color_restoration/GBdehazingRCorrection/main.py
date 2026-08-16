import numpy as np
from iproc.image_color_restoration.GBdehazingRCorrection.DetermineDepth import determineDepth
from iproc.image_color_restoration.GBdehazingRCorrection.TransmissionEstimation import getTransmission
from iproc.image_color_restoration.GBdehazingRCorrection.getAdaptiveExposureMap import AdaptiveExposureMap
from iproc.image_color_restoration.GBdehazingRCorrection.getAdaptiveSceneRadiance import AdaptiveSceneRadiance
from iproc.image_color_restoration.GBdehazingRCorrection.getAtomsphericLight import getAtomsphericLight
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_color_restoration.GBdehazingRCorrection.refinedTransmission import refinedtransmission

from iproc.image_color_restoration.GBdehazingRCorrection.sceneRadianceGb import sceneRadianceGB
from iproc.image_color_restoration.GBdehazingRCorrection.sceneRadianceR import sceneradiance


class GBDehazingRCorrection_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray):
        img = (img - img.min()) / (img.max() - img.min()) * 255
        blockSize = 9
        largestDiff = determineDepth(img, blockSize)
        AtomsphericLight, AtomsphericLightGB, AtomsphericLightRGB = getAtomsphericLight(largestDiff, img)
        print('AtomsphericLightRGB', AtomsphericLightRGB)

        # transmission = getTransmission(img, AtomsphericLightRGB, blockSize=blockSize)
        transmission = getTransmission(img, AtomsphericLightRGB, blockSize)
        # print('transmission.shape',transmission.shape)
        # TransmissionComposition(folder, transmission, number, param='coarse')
        transmission = refinedtransmission(transmission, img)

        # TransmissionComposition(folder, transmission, number, param='refined_15_175_175')
        sceneRadiance_GB = sceneRadianceGB(img, transmission, AtomsphericLightRGB)

        # cv2.imwrite('OutputImages/' + prefix + 'GBDehazed.jpg', sceneRadiance_GB)

        # # print('sceneRadiance_GB',sceneRadiance_GB)
        sceneRadiance = sceneradiance(img, sceneRadiance_GB)
        # sceneRadiance= sceneRadiance_GB
        # cv2.imwrite('OutputImages/'+ prefix + 'GBDehazedRcoorectionUDCP.jpg', sceneRadiance)
        # # print('np.min(sceneRadiance)',np.min(sceneRadiance))
        # # print('sceneRadiance',sceneRadiance)

        S_x = AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
        # print('S_x',S_x)
        sceneRadiance = AdaptiveSceneRadiance(sceneRadiance, S_x)

        # print('sceneRadiance',sceneRadiance)
        return sceneRadiance



