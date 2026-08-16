import cv2
import numpy as np
# def getMinChannel(img,AtomsphericLight):
#     img = np.float32(img)/AtomsphericLight
#     imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(0, img.shape[0]):
#         for j in range(0, img.shape[1]):
#             localMin = 1
#             for k in range(0, 2):
#                 imgNormalization = img.item((i, j, k))
#                 if imgNormalization < localMin:
#                     localMin = imgNormalization
#             imgGrayNormalization[i, j] = localMin
#     return imgGrayNormalization
#
#
# # def getMinChannel(img,AtomsphericLight):
# #     imgGrayNormalization = np.zeros((img.shape[0], img.shape[1]))
# #     for i in range(0, img.shape[0]):
# #         for j in range(0, img.shape[1]):
# #             localMin = 1
# #             for k in range(0, 2):
# #                 imgNormalization = img.item((i, j, k)) / AtomsphericLight[k]
# #                 if imgNormalization < localMin:
# #                     localMin = imgNormalization
# #             imgGrayNormalization[i, j] = localMin
# #     return imgGrayNormalization
#
#
#
#
# def getTransmission(img,AtomsphericLight ,blockSize):
#     img = getMinChannel(img,AtomsphericLight)
#     addSize = int((blockSize - 1) / 2)
#     newHeight = img.shape[0] + blockSize - 1
#     newWidth = img.shape[1] + blockSize - 1
#     # 中间结果
#     imgMiddle = np.zeros((newHeight, newWidth))
#     imgMiddle[:, :] = 1
#     imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
#     # print('imgMiddle',imgMiddle)
#     imgDark = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(addSize, newHeight - addSize):
#         for j in range(addSize, newWidth - addSize):
#             localMin = 1
#             for k in range(i - addSize, i + addSize + 1):
#                 for l in range(j - addSize, j + addSize + 1):
#                     if imgMiddle.item((k, l)) < localMin:
#                         localMin = imgMiddle.item((k, l))
#             imgDark[i - addSize, j - addSize] = localMin
#     transmission  = 1- imgDark
#
#     transmission = np.clip(transmission, 0.1, 0.9)
#     return transmission




def getTransmission(normI,AtomsphericLight ,w):
    M, N, C = normI.shape #M are the rows, N are the columns, C is the bgr channel
    B = AtomsphericLight
    ratio = normI / B

    # Windowed min filter (channels 0,1 only), vectorized via cv2.erode with
    # an all-ones kernel — the exact equivalent of the manual np.ndindex
    # loop this replaces. borderValue=0 reproduces the original's
    # np.pad(..., 'constant') zero-padding exactly (even though 0 isn't a
    # domain-neutral fill for a min filter here — that's the original's own
    # behavior, preserved as-is rather than "fixed", since this is a
    # performance-only pass).
    # float64, matching the original's precision (normI/B here is computed
    # without an explicit downcast, same as the original's np.pad(normI/B,...)).
    kernel = np.ones((w, w), np.uint8)
    ch0 = cv2.erode(ratio[:, :, 0].astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ch1 = cv2.erode(ratio[:, :, 1].astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    transmission = np.zeros((M, N, 2))
    transmission[:, :, 0] = 1 - ch0
    transmission[:, :, 1] = 1 - ch1
    return transmission