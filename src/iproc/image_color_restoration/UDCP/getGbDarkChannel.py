import cv2
import numpy as np


def getMinChannel(img):
	# Only channels 0,1 ("GB" — this algorithm's whole premise is excluding the
	# fastest-attenuating channel from the dark-channel estimate) — preserved
	# exactly as the original indexed them.
	return img[:, :, :2].astype(np.float32).min(axis=2)

def getDarkChannel(img, blockSize):
    img = getMinChannel(img)
    # Windowed min filter: cv2.erode with an all-ones kernel is the exact
    # vectorized equivalent of the manual double loop this replaces.
    # borderValue=255 reproduces the original's explicit 255-padding (neutral
    # for a min filter, same as DCP's identical technique).
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.erode(img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=255)
    return imgDark.astype(np.uint8)