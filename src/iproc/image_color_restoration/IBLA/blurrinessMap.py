import cv2
import numpy as np


def getMaxDarkChannel(img, blockSize):
    # Windowed MAX filter (localMin/`>`, despite the name): cv2.dilate with
    # an all-ones kernel is the exact vectorized equivalent; borderValue=0
    # reproduces the original's explicit 0-padding (neutral for a max filter).
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.dilate(img.astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.uint8(imgDark)


def blurrnessMap(img,blockSize,n):
    B = np.zeros((img.shape))
    for i in range(1, n):

        r = 2 ** i * (n - 1) + 1
        # print('r', r)
        img = np.uint8(img)
        blur = cv2.GaussianBlur(img, (r, r), r)
        blur = np.float32(blur)
        img = np.float32(img)
        B = np.absolute((img - blur)) + B
    B_Map = B / (n - 1)
    B_Map = np.uint8(B_Map)

    B_Map_dark = cv2.cvtColor((B_Map), cv2.COLOR_BGR2GRAY)

    Roughdepthmap = getMaxDarkChannel(B_Map_dark, blockSize)
    Refinedepthmap = cv2.bilateralFilter(Roughdepthmap, 9, 75, 75)
    return Refinedepthmap