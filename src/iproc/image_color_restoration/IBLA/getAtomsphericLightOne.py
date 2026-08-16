import numpy as np


def getAtomsphericLightDCP_Bright(darkChannel, img,percent):
    size = darkChannel.size
    k = int(percent * size)
    img = np.float32(img)

    # Top-k dark-channel-brightest pixels. This is a sum/mean over the whole
    # candidate SET, so unlike a single-pixel argmax/argmin, WHICH k pixels
    # get selected matters when there's a tie at the k-th boundary value —
    # argpartition picks an arbitrary subset among ties, which silently
    # changed the result on some real test images. A full stable sort by
    # descending value (ties broken by ascending row-major index, matching
    # the original's stable Python sort over its row-major-appended node
    # list) reproduces the exact original candidate set.
    flat_dark = darkChannel.reshape(-1)
    flat_img = img.reshape(-1, 3)
    order = np.argsort(-flat_dark, kind='stable')[:k]
    return flat_img[order].sum(axis=0) / k

