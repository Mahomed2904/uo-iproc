import numpy as np
import cv2

def getTransmissionMap(img,AtomsphericLight ,blockSize):

    AtomsphericLight = np.array(AtomsphericLight)
    img = np.float64(img)
    imgGrayNormalization = img / AtomsphericLight

    # The original's `for k in range(0, 3): ...` normalization loop leaves
    # `k` at 2 once it ends (Python loop variables persist), so the
    # median-blur rescale below only ever touched channel index 2 — channels
    # 0,1 kept their raw (un-blurred) ratio. Preserved exactly as a
    # performance-only pass, not "fixed", since fixing it would change
    # the algorithm's actual output.
    k = 2
    imgUint8 = np.uint8((imgGrayNormalization[:, :, k] / np.max(imgGrayNormalization[:, :, k])) * 255)
    imgGrayNormalization[:, :, k] = np.float32(cv2.medianBlur(imgUint8, blockSize)) / 255

    # The original initializes imgDark to 0 and only overwrites a pixel when
    # some channel's ratio is < the running localMin (started at 1) — so a
    # pixel whose true 3-channel min is >= 1 is never touched and stays at 0,
    # not at its real min. Reproduced here rather than using a plain min().
    channel_min = imgGrayNormalization.min(axis=2)
    imgDark = np.where(channel_min < 1, channel_min, 0.0)

    transmission  = 1- imgDark
    transmission = np.clip(transmission, 0.1, 0.9)
    return transmission