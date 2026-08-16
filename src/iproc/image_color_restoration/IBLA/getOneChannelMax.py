import cv2
import numpy as np


def getMaxChannel(img, blockSize):
    # Windowed MAX filter: cv2.dilate with an all-ones kernel is the exact
    # vectorized equivalent of the manual double loop this replaces.
    # borderValue=0 reproduces the original's explicit 0-padding.
    kernel = np.ones((blockSize, blockSize), np.uint8)
    return cv2.dilate(img.astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)