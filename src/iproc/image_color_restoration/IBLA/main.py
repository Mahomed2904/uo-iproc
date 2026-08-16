from iproc.image_color_restoration.IBLA.CloseDepth import closePoint
from iproc.image_color_restoration.IBLA.F_stretching import StretchingFusion
from iproc.image_color_restoration.IBLA.MapFusion import Scene_depth
from iproc.image_color_restoration.IBLA.MapOne import max_R
from iproc.image_color_restoration.IBLA.MapTwo import R_minus_GB
from iproc.image_color_restoration.IBLA.blurrinessMap import blurrnessMap
from iproc.image_color_restoration.IBLA.getAtomsphericLightFusion import ThreeAtomsphericLightFusion
from iproc.image_color_restoration.IBLA.getAtomsphericLightOne import getAtomsphericLightDCP_Bright
from iproc.image_color_restoration.IBLA.getAtomsphericLightThree import getAtomsphericLightLb
from iproc.image_color_restoration.IBLA.getAtomsphericLightTwo import getAtomsphericLightLv
from iproc.image_color_restoration.IBLA.getRGbDarkChannel import getRGB_Darkchannel
from iproc.image_color_restoration.IBLA.getRefinedTransmission import Refinedtransmission
from iproc.image_color_restoration.IBLA.getTransmissionGB import getGBTransmissionESt
from iproc.image_color_restoration.IBLA.getTransmissionR import getTransmission
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_color_restoration.IBLA.sceneRadiance import sceneRadianceRGB
from iproc.image_color_restoration.IBLA.sceneRadianceHE import RecoverHE


class IBLA_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img):
        blockSize = 9
        n = 5
        RGB_Darkchannel = getRGB_Darkchannel(img, blockSize)
        BlurrnessMap = blurrnessMap(img, blockSize, n)
        AtomsphericLightOne = getAtomsphericLightDCP_Bright(RGB_Darkchannel, img, percent=0.001)
        AtomsphericLightTwo = getAtomsphericLightLv(img)
        AtomsphericLightThree = getAtomsphericLightLb(img, BlurrnessMap)
        AtomsphericLight = ThreeAtomsphericLightFusion(AtomsphericLightOne, AtomsphericLightTwo, AtomsphericLightThree,
                                                       img)
        print('AtomsphericLight', AtomsphericLight)  # [b,g,r]

        R_map = max_R(img, blockSize)
        mip_map = R_minus_GB(img, blockSize, R_map)
        bluriness_map = BlurrnessMap

        d_R = 1 - StretchingFusion(R_map)
        d_D = 1 - StretchingFusion(mip_map)
        d_B = 1 - StretchingFusion(bluriness_map)

        d_n = Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
        d_0 = closePoint(img, AtomsphericLight)
        d_f = 8 * (d_n + d_0)

        # cv2.imwrite('OutputImages/' + prefix + '_IBLADepthMapd_D.jpg', np.uint8((d_D)*255))
        # cv2.imwrite('OutputImages/' + prefix + '_IBLADepthMap.jpg', np.uint8((d_f/d_f.max())*255))

        transmissionR = getTransmission(d_f)
        transmissionB, transmissionG = getGBTransmissionESt(transmissionR, AtomsphericLight)
        transmissionB, transmissionG, transmissionR = Refinedtransmission(transmissionB, transmissionG, transmissionR,
                                                                          img)

        # cv2.imwrite('OutputImages/' + prefix + '_IBLA_TM.jpg', np.uint8(np.clip(transmissionR * 255, 0, 255)))

        sceneRadiance = sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight)
        # cv2.imwrite('OutputImages/' + prefix + '_IBLA.jpg', sceneRadiance)

        sceneRadiance = RecoverHE(sceneRadiance)
        # cv2.imwrite('OutputImages/' + prefix + '_IBLA_HE.jpg', sceneRadiance)

        return sceneRadiance
