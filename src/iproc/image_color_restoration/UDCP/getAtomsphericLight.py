import numpy as np


def getAtomsphericLight(darkChannel, img):
    # np.argmax returns the first occurrence (row-major) on ties, matching
    # the original's stable sort by value descending -> nodes[0].
    y, x = np.unravel_index(np.argmax(darkChannel), darkChannel.shape)
    return img[y, x, :]

