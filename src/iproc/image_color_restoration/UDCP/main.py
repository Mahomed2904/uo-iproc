from iproc.image_color_restoration.UDCP.RefinedTramsmission import Refinedtransmission
from iproc.image_color_restoration.UDCP.getAtomsphericLight import getAtomsphericLight
from iproc.image_color_restoration.UDCP.getGbDarkChannel import getDarkChannel
from iproc.image_color_restoration.UDCP.getTM import getTransmission
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_color_restoration.UDCP.sceneRadiance import sceneRadianceRGB


class UDCP_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        blockSize = 9
        GB_Darkchannel = getDarkChannel(img, blockSize)
        AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)

        print('AtomsphericLight', AtomsphericLight)
        # print('img/AtomsphericLight', img/AtomsphericLight)

        # AtomsphericLight = [231, 171, 60]

        transmission = getTransmission(img, AtomsphericLight, blockSize)

        # cv2.imwrite('OutputImages/' + prefix + '_UDCP_Map.jpg', np.uint8(transmission))

        transmission = Refinedtransmission(transmission, img)
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
        # print('AtomsphericLight',AtomsphericLight)

        return sceneRadiance
