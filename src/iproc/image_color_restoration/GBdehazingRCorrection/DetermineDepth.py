import cv2
import numpy as np

def getMAxChannel(img):
    # Channels 0,1 only, matching the original's range(0,2) exactly (this
    # algorithm's index choices only make sense for BGR input, not the RGB
    # this codebase actually feeds it — a pre-existing possible correctness
    # issue, out of scope for this performance-only pass; indices preserved
    # as-is rather than "corrected").
    return img[:, :, :2].max(axis=2).astype(np.float32)


def getDarkChannel(img, blockSize):
    # Despite the name, this is a windowed MAX filter (localMax/`>`).
    # cv2.dilate with an all-ones kernel is the exact vectorized equivalent;
    # borderValue=0 reproduces the original's explicit 0-padding (neutral
    # for a max filter).
    # Computed at float64 (matching the original's imgMiddle, which is
    # float64 by numpy's default dtype) — only the final return value is
    # truncated to float16, same as the original's final assignment.
    kernel = np.ones((blockSize, blockSize), np.uint8)
    imgDark = cv2.dilate(img.astype(np.float64), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return imgDark.astype(np.float16)


def determineDepth(img, blockSize):
    img2 = img/255
    img_GB = getMAxChannel(img2)
    Max_GB = getDarkChannel(img_GB, blockSize)
    Max_R  = getDarkChannel(img2[:,:,2], blockSize)
    largestDiff = Max_R  - Max_GB

    return largestDiff