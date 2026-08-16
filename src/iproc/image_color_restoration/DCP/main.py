from iproc.image_color_restoration.DCP.GuidedFilter import GuidedFilter

__author__ = 'wy'
import cv2
import numpy as np

from iproc.interface.ImageAlgorithmInterface import ImageAlgorithmInterface


class DCP_ICR(ImageAlgorithmInterface):

    # 获取最小值矩阵
    # 获取BGR三个通道的最小值
    @staticmethod
    def __getMinChannel(img):
        # 输入检查
        if len(img.shape) == 3 and img.shape[2] == 3:
            pass
        else:
            print("bad image shape, input must be color image")
            return None

        return img.min(axis=2).astype(np.uint8)

    @staticmethod
    # 获取暗通道
    def __getDarkChannel(img, blockSize):
        # 输入检查
        if len(img.shape) == 2:
            pass
        else:
            print("bad image shape, input image must be two demensions")
            return None

        # blockSize检查
        if blockSize % 2 == 0 or blockSize < 3:
            print('blockSize is not odd or too small')
            return None

        # Windowed min filter: cv2.erode with an all-ones kernel computes the
        # local minimum over each blockSize x blockSize neighborhood, exactly
        # matching the manual double loop this replaces. borderValue=255
        # reproduces the original's explicit padding with 255 (the max
        # representable value, so it never wins the min unless every real
        # neighbor is also 255 — i.e. it acts as "no effect" at the border,
        # same as the original's padded imgMiddle).
        kernel = np.ones((blockSize, blockSize), np.uint8)
        return cv2.erode(img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)

    # 获取全局大气光强度
    @staticmethod
    def __getAtomsphericLight(darkChannel, img, meanMode=False, percent=0.001):
        size = darkChannel.size
        # int(percent * size) == 0 (tiny images) falls back to k=1, which is
        # exactly the original's "only consider the single brightest
        # dark-channel pixel" special case.
        k = max(1, int(percent * size))

        # Full stable sort by descending value (ties broken by ascending
        # row-major index), not argpartition — argpartition picks an
        # arbitrary subset among tied boundary values, which can silently
        # select a different top-k set than the original's stable Python
        # sort and change the result (confirmed on real images for the
        # analogous IBLA case; fixed defensively here too).
        # darkChannel is uint8 — negating it directly wraps around instead of
        # producing negative values, so cast to a signed type first.
        flat_dark = darkChannel.reshape(-1).astype(np.int32)
        flat_img = img.reshape(-1, 3)
        order = np.argsort(-flat_dark, kind='stable')[:k]
        candidates = flat_img[order]

        if meanMode:
            return int(candidates.astype(np.int64).sum() / (k * 3))

        return int(candidates.max())

    # 恢复原图像
    # Omega 去雾比例 参数
    # t0 最小透射率值
    @staticmethod
    def __getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
        gimfiltR = 50  # 引导滤波时半径的大小
        eps = 10 ** -3  # 引导滤波时epsilon的值

        imgGray = DCP_ICR.__getMinChannel(img)
        # print('imgGray', imgGray)
        imgDark = DCP_ICR.__getDarkChannel(imgGray, blockSize=blockSize)
        atomsphericLight = DCP_ICR.__getAtomsphericLight(imgDark, img, meanMode=meanMode, percent=percent)

        imgDark = np.float64(imgDark)
        transmission = 1 - omega * imgDark / atomsphericLight

        guided_filter = GuidedFilter(img, gimfiltR, eps)
        transmission = guided_filter.filter(transmission)

        # 防止出现t小于0的情况
        # 对t限制最小值为0.1

        transmission = np.clip(transmission, t0, 0.9)

        sceneRadiance = np.zeros(img.shape)
        for i in range(0, 3):
            img = np.float64(img)
            sceneRadiance[:, :, i] = (img[:, :, i] - atomsphericLight) / transmission + atomsphericLight

        sceneRadiance = np.clip(sceneRadiance, 0, 255)
        sceneRadiance = np.uint8(sceneRadiance)

        return transmission, sceneRadiance

    @staticmethod
    def apply(image: np.ndarray):
        transmission, sceneRadiance = DCP_ICR.__getRecoverScene(image)
        return sceneRadiance


