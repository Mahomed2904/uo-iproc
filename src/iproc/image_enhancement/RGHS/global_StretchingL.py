import numpy as np

def global_stretching(img_L,height, width):
    length = height * width
    R_rray = np.sort(img_L.flatten())
    I_min = int(R_rray[int(length / 100)])
    I_max = int(R_rray[-int(length / 100)])

    below = img_L < I_min
    above = img_L > I_max
    mid = ~below & ~above

    result = np.zeros((height, width))
    result[above] = 100
    # int() truncation, matching the original's int(...) on the mid branch
    # (safe to truncate-toward-zero via astype here since these values are
    # always >= 0 in this branch).
    result[mid] = ((img_L[mid] - I_min) * (100 / (I_max - I_min))).astype(int)
    return result

