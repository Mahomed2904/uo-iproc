import cv2
import numpy as np


def getMAxChannel(img):
    # Channels 0,1 only, matching the original's range(0,2) exactly (same
    # BGR-assuming index choice noted in GBdehazingRCorrection — preserved
    # as-is, not "corrected", since this is a performance-only pass).
    return img[:, :, :2].max(axis=2).astype(np.float64)



def getMaxChannel(img, blockSize):
    # Windowed MAX filter (despite this being called from "Dark"-named
    # callers elsewhere): cv2.dilate with an all-ones kernel is the exact
    # vectorized equivalent of the manual double loop this replaces.
    # borderValue=0 reproduces the original's explicit 0-padding (neutral
    # for a max filter).
    kernel = np.ones((blockSize, blockSize), np.uint8)
    return cv2.dilate(img.astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def DepthMap(img, blockSize):
    img = img/255
    img_GB = getMAxChannel(img)
    Max_GB = getMaxChannel(img_GB, blockSize)
    Max_R  = getMaxChannel(img[:,:,2], blockSize)
    largestDiff = Max_R  - Max_GB

    return largestDiff