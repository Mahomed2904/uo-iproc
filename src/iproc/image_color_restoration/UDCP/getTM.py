import cv2
import numpy as np

def getMinChannel(img,AtomsphericLight):
    # Channels 0,1 ("GB"), each normalized against the matching atmospheric-
    # light component, then the per-pixel min of the two ratios — preserved
    # exactly as the original indexed/ordered these operations.
    ratio = img[:, :, :2].astype(np.float64) / np.asarray(AtomsphericLight[:2], dtype=np.float64)
    return ratio.min(axis=2)

def getTransmission(img,AtomsphericLight ,blockSize):
    img = getMinChannel(img,AtomsphericLight)
    # Windowed min filter, vectorized the same way as getGbDarkChannel's —
    # borderValue=1 here (not 255) to match this function's own imgMiddle
    # fill value, which is the correct neutral value for these ~[0,1]-ranged
    # normalized ratios rather than raw 0-255 intensities.
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.erode(img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1)
    transmission  = 1- imgDark

    transmission = np.clip(transmission, 0.1, 0.9)


    return transmission