import cv2
import numpy as np


def getMinChannel(img):
	return img.min(axis=2).astype(np.float32)

def getDarkChannel(img, blockSize):
    img = getMinChannel(img)
    # Windowed min filter via cv2.erode with an all-ones kernel — the exact
    # vectorized equivalent of the manual double loop this replaces.
    # borderValue=255 reproduces the original's explicit 255-padding.
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.erode(img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
    return imgDark.astype(np.uint8)