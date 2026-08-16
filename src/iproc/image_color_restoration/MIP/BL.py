import numpy as np


def getAtomsphericLight(transmission, img):
    # np.argmin returns the first occurrence (row-major) on ties, matching
    # the original's stable ascending sort by value -> nodes[0].
    y, x = np.unravel_index(np.argmin(transmission), transmission.shape)
    return img[y, x, :]

