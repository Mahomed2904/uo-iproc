import cv2
import numpy as np

def GetMaxR(img,blockSize):
    # Windowed MAX filter (localMin/`>`, despite the name): cv2.dilate with
    # an all-ones kernel is the exact vectorized equivalent; borderValue=0
    # reproduces the original's explicit 0-padding (neutral for a max filter).
    kernel = np.ones((blockSize, blockSize), np.uint8)
    return cv2.dilate(img.astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def TransmissionR(transmissionGB,img,blockSize):
    img = np.float16(img)/255
    MaxRChannel = GetMaxR(img[:,:,2],blockSize)
    # print('np.average(MaxRChannel)',np.average(MaxRChannel))
    # print('np.average(transmissionGB)',np.average(transmissionGB))
    alpha  = np.average(transmissionGB)/ np.average(MaxRChannel)
    transmissionR = alpha * MaxRChannel
    # transmissionR = np.clip(transmissionR, 0.1, 0.9)
    transmission = transmissionR
    # print('np.average(transmission)',np.average(transmission))
    return transmission





