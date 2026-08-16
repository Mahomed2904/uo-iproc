import numpy as np


def getAtomsphericLight(darkChannel, img):
    img = np.float32(img)
    # np.argmin returns the first occurrence (row-major) on ties, matching
    # the original's stable ascending sort by value -> nodes[0].
    y, x = np.unravel_index(np.argmin(darkChannel), darkChannel.shape)

    atomsphericLight = np.mean([img[y, x, 0], img[y, x, 1]])
    atomsphericLightGB = img[y, x, 0:2]
    atomsphericLightRGB = img[y, x, :]
    return atomsphericLight, atomsphericLightGB, atomsphericLightRGB
