from iproc.image_color_restoration.LowComplexityDCP.getAtomsphericLight import getAtomsphericLight
from iproc.image_color_restoration.LowComplexityDCP.getColorContrastEnhancement import ColorContrastEnhancement
from iproc.image_color_restoration.LowComplexityDCP.getRGBDarkChannel import getDarkChannel
from iproc.image_color_restoration.LowComplexityDCP.getSceneRadiance import SceneRadiance

from iproc.image_color_restoration.LowComplexityDCP.getTransmissionEstimation import getTransmissionMap
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class LowComplexityDCP_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        blockSize = 9

        imgGray = getDarkChannel(img, blockSize)
        AtomsphericLight = getAtomsphericLight(imgGray, img, meanMode=True, percent=0.001)
        transmission = getTransmissionMap(img, AtomsphericLight, blockSize)
        sceneRadiance = SceneRadiance(img, AtomsphericLight, transmission)
        sceneRadiance = ColorContrastEnhancement(sceneRadiance)

        return sceneRadiance



