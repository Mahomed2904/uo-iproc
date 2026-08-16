import cv2
import numpy as np

def getMAxChannel(B_Dark,G_Dark):
    return np.maximum(B_Dark, G_Dark).astype(np.float16)



def getDarkChannel(img, blockSize):
    # Despite the name, this is a windowed MIN filter (localMax/`<`).
    # cv2.erode with an all-ones kernel is the exact vectorized equivalent;
    # borderValue=1 reproduces the original's explicit 1-padding (neutral
    # for a min filter over these ~[0,1]-ranged normalized values).
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.erode(img.astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1)
    return imgDark.astype(np.float16)


def determineDepth(img, blockSize):
    img = np.float16(img)
    img = img/255

    R_Dark = getDarkChannel(img[:, :, 2], blockSize)
    G_Dark = getDarkChannel(img[:, :, 1], blockSize)
    B_Dark = getDarkChannel(img[:,:,0], blockSize)
    GB_Max = getMAxChannel(B_Dark,G_Dark)
    largestDiff = R_Dark  - GB_Max
    return largestDiff
