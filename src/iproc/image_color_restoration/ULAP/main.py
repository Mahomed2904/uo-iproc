import numpy as np

from iproc.image_color_restoration.ULAP.GuidedFilter import GuidedFilter
from iproc.image_color_restoration.ULAP.backgroundLight import BLEstimation
from iproc.image_color_restoration.ULAP.depthMapEstimation import depthMap
from iproc.image_color_restoration.ULAP.depthMin import minDepth
from iproc.image_color_restoration.ULAP.getRGBTransmission import getRGBTransmissionESt
from iproc.image_color_restoration.ULAP.global_Stretching import global_stretching
from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface
from iproc.image_color_restoration.ULAP.refinedTransmissionMap import refinedtransmissionMap

from iproc.image_color_restoration.ULAP.sceneRadiance import sceneRadianceRGB


class ULAP_ICR(ImageAlgorithmInterface):

    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        blockSize = 9
        gimfiltR = 50  # 引导滤波时半径的大小
        eps = 10 ** -3  # 引导滤波时epsilon的值

        DepthMap = depthMap(img)
        DepthMap = global_stretching(DepthMap)
        guided_filter = GuidedFilter(img, gimfiltR, eps)
        refineDR = guided_filter.filter(DepthMap)
        refineDR = np.clip(refineDR, 0, 1)

        AtomsphericLight = BLEstimation(img, DepthMap) * 255

        d_0 = minDepth(img, AtomsphericLight)
        d_f = 8 * (DepthMap + d_0)
        transmissionB, transmissionG, transmissionR = getRGBTransmissionESt(d_f)

        transmission = refinedtransmissionMap(transmissionB, transmissionG, transmissionR, img)
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

        return sceneRadiance



