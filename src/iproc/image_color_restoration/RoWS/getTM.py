import cv2
import numpy as np
def getMinChannel(img,AtomsphericLight):
    product = img * AtomsphericLight
    return np.minimum(product.min(axis=2), 1)

def getTransmission(img,AtomsphericLight ,blockSize):
    img = img/255
    AtomsphericLight = AtomsphericLight/255
    img = getMinChannel(img,AtomsphericLight)
    # Windowed min filter via cv2.erode, matching getDarkChannel's approach —
    # borderValue=1 reproduces the original's explicit 1-padding.
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.erode(img, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1)
    transmission  = 1- imgDark
    transmission = np.clip(transmission, 0.1, 0.9)

    return transmission