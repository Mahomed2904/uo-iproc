from iproc.image_color_restoration.NewOpticalModel.DetermineDepth import determineDepth
from iproc.image_color_restoration.NewOpticalModel.getAtomsphericLight import getAtomsphericLight
from iproc.image_color_restoration.NewOpticalModel.getRefinedTramsmission import Refinedtransmission
from iproc.image_color_restoration.NewOpticalModel.getScatteringRate import ScatteringRateMap
from iproc.image_color_restoration.NewOpticalModel.getSceneRadiance import SceneRadiance
from iproc.image_color_restoration.NewOpticalModel.getTransmissionGB import TransmissionGB
from iproc.image_color_restoration.NewOpticalModel.getTransmissionR import TransmissionR
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class NewOpticalModel_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        blockSize = 9
        largestDiff = determineDepth(img, blockSize)
        AtomsphericLight = getAtomsphericLight(largestDiff, img)
        print('AtomsphericLight', AtomsphericLight)
        sactterRate = ScatteringRateMap(img, AtomsphericLight, blockSize)
        print('sactterRate', sactterRate)
        transmissionGB = TransmissionGB(sactterRate)
        transmissionR = TransmissionR(transmissionGB, img, blockSize)
        transmissionGB, transmissionR = Refinedtransmission(transmissionGB, transmissionR, img)
        sceneRadiance = SceneRadiance(img, transmissionGB, transmissionR, sactterRate, AtomsphericLight)

        return sceneRadiance

